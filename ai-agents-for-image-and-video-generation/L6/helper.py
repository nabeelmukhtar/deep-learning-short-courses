import os
import functools
import time
import base64
from io import BytesIO

import google.auth
import google.auth.credentials
import google.auth.transport.requests
from dotenv import load_dotenv
from google.auth import impersonated_credentials
from google.oauth2 import service_account
import numpy as np
import matplotlib.pyplot as plt
import copy


def authenticate(
    location: str | None = None,
) -> tuple[google.auth.credentials.Credentials, str]:
    load_dotenv(override=True)

    # 1. Locate the JSON key file
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    # SAFETY CHECK: Ensure the file actually exists
    if not key_path or not os.path.exists(key_path):
        # Fallback: Try to look in the current directory if env var is missing
        if os.path.exists("credentials.json"):
            key_path = "credentials.json"
        elif os.path.exists("../credentials.json"):
            key_path = "../credentials.json"
        else:
            raise ValueError(
                "Could not find credentials.json or GOOGLE_APPLICATION_CREDENTIALS env var."
            )

    # 2. Load the file EXPLICITLY with scopes
    source_credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    # 3. Create the HTTP Request object
    request = google.auth.transport.requests.Request()

    # 4. Refresh the source credential (The 1-hour token)
    source_credentials.refresh(request)

    # 5. Create Impersonated Credentials (The 2-hour token)
    # This allows the service account to "impersonate" itself to get a
    # fresh token with a longer lifetime (up to 12 hours, set to 2 here).
    # This is often used in notebook environments to avoid re-authenticating.
    target_principal = source_credentials.service_account_email

    credentials = impersonated_credentials.Credentials(
        source_credentials=source_credentials,
        target_principal=target_principal,
        target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        lifetime=7200,  # 2 Hours
        iam_endpoint_override=os.getenv("DLAI_GOOGLE_IAM_ENDPOINT")
    )

    # 6. Refresh to get the final token
    credentials.refresh(request)

    # Set Env Vars
    os.environ["GOOGLE_CLOUD_PROJECT"] = source_credentials.project_id
    if location:
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    return credentials, source_credentials.project_id


# @title Helper: radar chart visualization
def plot_radar_comparison(
    CRITERIA,
    aligned_image,
    aligned_gemini,
    aligned_siglip,
    misaligned_image,
    misaligned_gemini,
    misaligned_siglip,
):
    angles = np.linspace(
        0, 2 * np.pi, len(CRITERIA),
        endpoint=False,
    ).tolist() + [0]
    labels = [
        c.replace('_', ' ').title()
        for c in CRITERIA
    ]

    def radar_vals(result):
        v = [
            result[k]["score"]
            for k in CRITERIA
        ]
        return v + v[:1]

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        "Gemini evaluation:"
        " aligned (blue)"
        " vs misaligned (red)",
        fontsize=10, color="gray",
    )

    pairs = [
        (
            aligned_image,
            aligned_gemini,
            aligned_siglip,
            "#2563EB", "Aligned",
        ),
        (
            misaligned_image,
            misaligned_gemini,
            misaligned_siglip,
            "#DC2626", "Misaligned",
        ),
    ]

    for idx, (
        img, res, sc, col, lbl
    ) in enumerate(pairs):
        ax_img = fig.add_subplot(
            2, 2, idx * 2 + 1
        )
        ax_img.imshow(img)
        ax_img.set_title(
            f"{lbl}\nSigLIP: {sc:.4f}",
            fontsize=10,
            fontweight="bold",
            pad=8,
        )
        ax_img.axis("off")

        ax_rad = fig.add_subplot(
            2, 2, idx * 2 + 2,
            polar=True,
        )
        vals = radar_vals(res)
        ax_rad.fill(
            angles, vals,
            color=col, alpha=0.25,
        )
        ax_rad.plot(
            angles, vals,
            color=col, linewidth=2,
            marker="o", markersize=6,
        )
        ax_rad.set_xticks(angles[:-1])
        ax_rad.set_xticklabels(
            labels, fontsize=9,
        )
        ax_rad.set_ylim(0, 5)
        ax_rad.set_yticks(
            [1, 2, 3, 4, 5]
        )
        ax_rad.set_yticklabels(
            ["1", "2", "3", "4", "5"],
            fontsize=7,
        )
        overall = res['overall_score']
        ax_rad.set_title(
            f"Gemini: {overall}/5",
            fontsize=10,
            fontweight="bold",
            pad=20,
        )

    plt.tight_layout()
    plt.show()


# @title Helper: build a Gecko row from a local image
def make_gecko_row(prompt, image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")
    return {
        "prompt": prompt,
        "response": {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": b64,
                    }
                }
            ],
            "role": "model",
        },
    }


# @title Helper: extract image from response
def extract_image(response):
    from io import BytesIO
    from PIL import Image as _PILImage
    for part in (
        response.candidates[0].content.parts
    ):
        if (
            part.inline_data
            and part.inline_data.mime_type
            .startswith("image/")
        ):
            return _PILImage.open(
                BytesIO(part.inline_data.data)
            )
    return None


# @title Helper: clean non-printable characters
def clean(s):
    """Remove non-printable characters."""
    return "".join(c for c in str(s) if c.isprintable())


# @title Helper: tool wrapper with timing
def make_display_tool(fn):
    """Wrap a tool to log timing."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"\n>> {fn.__name__}")
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"   Done in {elapsed:.1f}s")
        return result
    return wrapper

def show_result_clean(result):
    clone = copy.deepcopy(result)

    # Scrub each case's response
    for case in clone.eval_case_results:
        if hasattr(case, "response"):
            case.response = "[image hidden]"

    # Scrub the embedded dataset too (in case .show() reads from it)
    for ds in clone.evaluation_dataset:
        if hasattr(ds, "eval_dataset_df") and ds.eval_dataset_df is not None:
            ds.eval_dataset_df = ds.eval_dataset_df.drop(
                columns=["response"], errors="ignore"
            )

    clone.show()

def show_clean(eval_dataset):
    clone = copy.deepcopy(eval_dataset)
    clone.eval_dataset_df = clone.eval_dataset_df.drop(
        columns=["response"]
    )
    clone.show()
