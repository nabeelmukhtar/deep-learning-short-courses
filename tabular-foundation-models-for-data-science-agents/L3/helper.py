import logging
import os
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from IPython.display import display
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

DATA_PATH = (
    Path(__file__).resolve().parent
    / "ro_shared_data"
    / "heart_disease.csv"
)

# A small ensemble keeps lesson API usage modest. In production,
# keep n_estimators at its default unless tuning it is needed.
N_ESTIMATORS = 4
RANDOM_STATE = 42

RENAME = {
    "age": "Age",
    "sex": "Sex",
    "chest": "Chest pain type",
    "resting_blood_pressure": "Resting BP",
    "serum_cholestoral": "Cholesterol",
    "fasting_blood_sugar": "High fasting sugar",
    "resting_electrocardiographic_results": "Resting ECG",
    "maximum_heart_rate_achieved": "Max heart rate",
    "exercise_induced_angina": "Exercise angina",
    "oldpeak": "ST depression",
    "slope": "ST slope",
    "number_of_major_vessels": "Blocked vessels",
    "thal": "Thallium scan",
}

DECODE = {
    "Sex": {0: "female", 1: "male"},
    "Chest pain type": {
        1: "typical angina",
        2: "atypical angina",
        3: "non-anginal",
        4: "asymptomatic",
    },
    "High fasting sugar": {0: "no", 1: "yes"},
    "Resting ECG": {
        0: "normal",
        1: "ST-T abnormality",
        2: "LV hypertrophy",
    },
    "Exercise angina": {0: "no", 1: "yes"},
    "ST slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "Thallium scan": {
        3: "normal",
        6: "fixed defect",
        7: "reversible defect",
    },
}

CATEGORICAL = [
    "Sex",
    "Chest pain type",
    "High fasting sugar",
    "Resting ECG",
    "Exercise angina",
    "ST slope",
    "Thallium scan",
]

CONTINUOUS = [
    "Age",
    "Resting BP",
    "Cholesterol",
    "Max heart rate",
    "ST depression",
    "Blocked vessels",
]

def load_env():
    _ = load_dotenv(find_dotenv())


def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key


def load_heart_disease(train_size=200, test_size=70):
    """Load readable Heart Statlog features and a binary target."""
    df = pd.read_csv(DATA_PATH).rename(columns=RENAME)
    X = df.drop(columns="target")
    y = df["target"]
    return train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def sample_heart_disease_rows(n_rows=8, *, rng=None):
    """Draw intact patient rows for an inference benchmark."""
    if n_rows < 1:
        raise ValueError("n_rows must be at least 1")

    rng = np.random.default_rng(rng)
    features = (
        pd.read_csv(DATA_PATH)
        .rename(columns=RENAME)
        .drop(columns="target")
    )
    positions = rng.choice(
        len(features),
        size=n_rows,
        replace=n_rows > len(features),
    )
    return features.iloc[positions].reset_index(drop=True)


def categorical_indices(feature_names):
    """Return positions of categorical columns."""
    return [feature_names.index(column) for column in CATEGORICAL]


def show_value(name, value):
    """Render a raw feature value as a human-readable label."""
    if name in DECODE:
        return DECODE[name].get(int(value), str(value))
    return f"{value:g}"


def describe_patient(
    shap_row,
    patient,
    feature_names,
    label,
    top_k=5,
):
    """Print the top drivers behind one patient's predicted risk."""
    risk = float(shap_row.values.sum() + shap_row.base_values)
    base = float(shap_row.base_values)
    print(
        f"[{label}] predicted risk = {risk:.0%}  "
        f"(reference average {base:.0%})"
    )
    for index in np.argsort(-np.abs(shap_row.values))[:top_k]:
        name = feature_names[index]
        if shap_row.values[index] > 0:
            direction = "raises"
        else:
            direction = "lowers"
        value = show_value(name, patient.iloc[index])
        print(
            f"    {name:<18} = {value:<18} "
            f"{direction} risk by "
            f"{abs(shap_row.values[index]) * 100:4.1f} points"
        )
    print()


def decoded_display_data(patient, feature_names):
    """Return human-readable feature values for SHAP plots."""
    return np.array(
        [
            show_value(name, patient.iloc[index])
            for index, name in enumerate(feature_names)
        ],
        dtype=object,
    )


def use_lesson_style():
    """Use legible, high-contrast plotting defaults."""
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5),
            "figure.dpi": 110,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.frameon": False,
        }
    )


def configure_lesson(*, plots=False):
    """Apply compact display defaults shared by lesson notebooks."""
    from tabpfn_client.options import get_opts

    options = get_opts()
    if hasattr(options, "TABPFN_CLIENT_CI_MODE"):
        os.environ["TABPFN_CLIENT_CI_MODE"] = "true"
        options.TABPFN_CLIENT_CI_MODE = True
    for name in ("tabpfn_client", "tabpfn_client.client"):
        logging.getLogger(name).setLevel(logging.WARNING)
    pd.set_option("display.max_columns", 7)
    if plots:
        use_lesson_style()


def quiet_client_call(method, *args, **kwargs):
    """Call the client without retaining animated spinner output."""
    with redirect_stdout(StringIO()):
        return method(*args, **kwargs)


def timed_client_call(method, *args, **kwargs):
    """Return a quiet client result and its elapsed time."""
    start = time.perf_counter()
    result = quiet_client_call(method, *args, **kwargs)
    return result, time.perf_counter() - start


def show_table(frame, caption, formats=None):
    """Display a consistently formatted lesson table."""
    styled = frame.style
    if formats:
        styled = styled.format(formats)
    display(styled.set_caption(caption))


def show_split_summary(X_train, X_test):
    """Show the dimensions of the training and test tables."""
    summary = pd.DataFrame(
        {
            "rows": [len(X_train), len(X_test)],
            "features": [X_train.shape[1], X_test.shape[1]],
        },
        index=["training set", "test set"],
    )
    show_table(summary, "Training and test split")


def show_fit_comparison(fit_times, predict_times, aucs, n_rows):
    """Display the measured timings and scores."""
    predict_label = f"request {n_rows} rows (s)"
    comparison = pd.DataFrame(
        {
            "fit (s)": fit_times,
            predict_label: predict_times,
            "test AUC": aucs,
        },
        index=["preprocessing only", "KV cache"],
    )
    formats = {
        "fit (s)": "{:.2f}",
        predict_label: "{:.2f}",
        "test AUC": "{:.3f}",
    }
    show_table(comparison, "One fit and one prediction", formats)


def show_latency_comparison(timings):
    """Keep server computation and whole-request timing separate."""
    labels = ["preprocessing only", "KV cache"]
    for column, caption in [
        ("server_ms", "Server prediction (ms)"),
        ("request_ms", "Whole request (ms)"),
    ]:
        grouped = timings.groupby("model")[column]
        summary = pd.DataFrame({
            "median": grouped.median(),
            "25th percentile": grouped.quantile(0.25),
            "75th percentile": grouped.quantile(0.75),
        }).reindex(labels)
        show_table(summary, caption, "{:.0f}")
        baseline, cached = summary["median"]
        if np.isfinite([baseline, cached]).all() and cached > 0:
            ratio = baseline / cached
            print(f"{caption}: uncached / cached = {ratio:.2f}×")
        else:
            print(f"{caption}: timing unavailable")


def benchmark_cache_comparison(
    baseline,
    cached,
    rows,
    n_queries=20,
    rows_per_query=8,
):
    """Compare warm models on matched batches; alternate order."""
    rng = np.random.default_rng(RANDOM_STATE)
    batches = [
        rows.iloc[
            rng.choice(len(rows), rows_per_query, replace=False)
        ].to_numpy()
        for _ in range(n_queries + 2)
    ]
    models = [("preprocessing only", baseline), ("KV cache", cached)]
    for _, model in models:
        for batch in batches[-2:]:
            quiet_client_call(model.predict_proba, batch)

    records = []
    for index in tqdm(
        range(n_queries), desc="Requests", leave=False
    ):
        order = models if index % 2 == 0 else models[::-1]
        for label, model in order:
            _, elapsed = timed_client_call(
                model.predict_proba, batches[index]
            )
            server = model.last_predict_timings or {}
            prediction = server.get("predict_s")
            queue = server.get("predict_queue_wait_s")
            records.append({
                "model": label,
                "query": index,
                "request_ms": elapsed * 1000,
                "server_ms": (
                    prediction * 1000
                    if prediction is not None else np.nan
                ),
                "queue_ms": (
                    queue * 1000 if queue is not None else np.nan
                ),
            })
    return pd.DataFrame(records)


def make_shap_explainer(model, X_train, y_train):
    """Build the lesson's SHAP explainer with a fixed reference."""
    import shap

    background, _ = train_test_split(
        X_train,
        train_size=20,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    def predict_risk(rows):
        probabilities = quiet_client_call(
            model.predict_proba, np.asarray(rows)
        )
        return probabilities[:, 1]

    return shap.PermutationExplainer(
        predict_risk, background, seed=RANDOM_STATE
    )


def plot_beeswarm(shap_values, feature_count):
    """Draw the lesson's compact global SHAP summary."""
    import shap

    shap.plots.beeswarm(
        shap_values,
        max_display=feature_count,
        show=False,
    )
    figure = plt.gcf()
    figure.set_size_inches(9, 5.5)
    figure.suptitle(
        "Global drivers of coronary-disease risk\n"
        "right = toward disease; red = high feature value",
        fontsize=13,
    )
    figure.tight_layout()
    plt.show()


def top_continuous_features(
    shap_values,
    feature_names,
    continuous,
    count=4,
):
    """Return indices of the strongest continuous features."""
    importance = np.abs(shap_values.values).mean(axis=0)
    ranking = np.argsort(-importance)
    return [
        index
        for index in ranking
        if feature_names[index] in continuous
    ][:count]


def plot_partial_dependence(
    model,
    data,
    feature_indices,
    feature_names,
):
    """Draw compact partial-dependence panels on a shared scale."""
    with redirect_stdout(StringIO()):
        result = PartialDependenceDisplay.from_estimator(
            model,
            data,
            features=list(feature_indices),
            grid_resolution=12,
            kind="average",
            target=1,
            response_method="predict_proba",
            method="brute",
            n_cols=2,
        )

    figure = result.figure_
    figure.set_size_inches(9, 5.5)
    axes = [
        axis
        for axis in np.ravel(result.axes_)
        if axis is not None
    ]
    values = [
        line.get_ydata()
        for axis in axes
        for line in axis.get_lines()
    ]
    low = min(value.min() for value in values)
    high = max(value.max() for value in values)
    padding = 0.05 * (high - low)

    for position, (axis, index) in enumerate(
        zip(axes, feature_indices)
    ):
        axis.set_ylim(low - padding, high + padding)
        axis.set_xlabel(feature_names[index])
        label = "Predicted P(disease)" if position % 2 == 0 else ""
        axis.set_ylabel(label)

    figure.suptitle(
        "Average predicted risk by measurement\n"
        "panels share the same y-axis",
        fontsize=13,
    )
    figure.tight_layout()
    plt.show()


def plot_patient_waterfall(
    shap_values,
    patient_index,
    patients,
    feature_names,
    label,
):
    """Describe one patient and draw a compact SHAP waterfall."""
    import shap

    row = shap_values[patient_index]
    patient = patients.iloc[patient_index]
    row.display_data = decoded_display_data(
        patient,
        feature_names,
    )
    describe_patient(row, patient, feature_names, label)
    predicted = row.values.sum() + row.base_values
    shap.plots.waterfall(row, show=False)
    figure = plt.gcf()
    figure.set_size_inches(9, 5)
    figure.suptitle(
        f"{label}: predicted risk {predicted:.0%}",
        fontsize=13,
    )
    figure.tight_layout()
    plt.show()
    
# ---------------------------------------------------------------
# L2 helpers: prediction, distributions, and forecasting
# ---------------------------------------------------------------

SHARED_DATA = Path(__file__).resolve().parent / "ro_shared_data"
CO2_PATH = SHARED_DATA / "co2_monthly.csv"
HOUSING_PATH = SHARED_DATA / "house_prices.csv"
ELECTRICITY_PATH = SHARED_DATA / "electricity_prices.csv"
HEART_FULL_PATH = SHARED_DATA / "heart_disease_full.csv"


def load_heart_disease_full(train_size=700, test_size=220):
    """Split the pooled four-hospital heart data (920 rows)."""
    df = pd.read_csv(HEART_FULL_PATH).rename(columns=RENAME)
    X = df.drop(columns="target")
    y = df["target"]
    return train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def plot_actual_vs_predicted(y_true, y_pred, unit=""):
    """Scatter actual against predicted with the ideal line."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    low = min(y.min(), p.min())
    high = max(y.max(), p.max())
    pad = 0.04 * (high - low)
    ends = [low - pad, high + pad]

    plt.figure(figsize=(6.2, 6.2))
    plt.plot(
        ends,
        ends,
        color="gray",
        ls="--",
        lw=1.5,
        label="perfect prediction",
    )
    plt.scatter(y, p, s=18, alpha=0.6)
    plt.xlim(ends)
    plt.ylim(ends)
    plt.gca().set_aspect("equal")
    plt.xlabel(f"actual {unit}".strip())
    plt.ylabel(f"predicted {unit}".strip())
    plt.title("Closer to the diagonal is better")
    plt.legend()
    plt.tight_layout()
    plt.show()


def init_tabpfn():
    """Authenticate the client, preferring a token in the env."""
    import tabpfn_client

    load_env()
    token = os.getenv("TABPFN_TOKEN")
    if token:
        tabpfn_client.set_access_token(token)
    elif hasattr(tabpfn_client, "interactive_login"):
        # tabpfn-client >= 0.6: init() no longer prompts for a login.
        # interactive_login() reuses a cached token or opens the browser.
        tabpfn_client.interactive_login()
    else:
        tabpfn_client.init()


def load_housing(train_size=1200, test_size=260):
    """Split Ames houses to predict the sale price in dollars."""
    df = pd.read_csv(HOUSING_PATH)
    X = df.drop(columns="Sale price")
    y = df["Sale price"]
    return train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )


def plot_quantile_band(
    y_true,
    lower,
    median,
    upper,
    ylabel,
    unit="test samples",
):
    """Plot actual values inside a predicted quantile band."""
    order = np.argsort(np.asarray(median))
    y = np.asarray(y_true)[order]
    low = np.asarray(lower)[order]
    mid = np.asarray(median)[order]
    high = np.asarray(upper)[order]
    inside = (y >= low) & (y <= high)
    x = np.arange(len(y))

    plt.figure()
    plt.fill_between(
        x,
        low,
        high,
        alpha=0.25,
        label="10th to 90th percentile",
    )
    plt.plot(x, mid, lw=1.5, label="predicted median")
    plt.scatter(
        x[inside], y[inside], s=18, label="actual, inside band"
    )
    plt.scatter(
        x[~inside],
        y[~inside],
        s=26,
        color="crimson",
        label="actual, outside band",
    )
    plt.xlabel(f"{unit}, sorted by predicted median")
    plt.ylabel(ylabel)
    plt.title(f"One uncertainty band per {unit.rstrip('s')}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_predicted_distribution(
    full_output,
    row,
    actual,
    xlabel,
):
    """Draw the predicted density for one regression test row."""
    # The full output spans a huge support with log-probabilities
    # per bin; far-tail bins may arrive as NaN and carry no mass.
    logits = np.asarray(full_output["logits"])[row]
    borders = np.asarray(full_output["borders"])
    if borders.ndim > 1:
        borders = borders[row]

    probabilities = np.exp(np.nan_to_num(logits, nan=-np.inf))
    probabilities = probabilities / probabilities.sum()
    density = probabilities / np.diff(borders)
    window = np.ones(21) / 21
    density = np.convolve(density, window, mode="same")
    centers = (borders[:-1] + borders[1:]) / 2

    cdf = np.cumsum(probabilities)
    keep = (cdf > 0.001) & (cdf < 0.999)
    inner = (cdf > 0.1) & (cdf < 0.9)

    plt.figure()
    plt.fill_between(
        centers[keep],
        density[keep],
        color="tab:blue",
        alpha=0.2,
        label="predicted distribution",
    )
    plt.fill_between(
        centers[inner],
        density[inner],
        color="tab:blue",
        alpha=0.5,
        label="80% interval (10th to 90th)",
    )
    mean = float(np.asarray(full_output["mean"])[row])
    plt.axvline(
        mean,
        color="tab:blue",
        ls="--",
        label=f"mean {mean:,.0f}",
    )
    plt.axvline(
        actual,
        color="crimson",
        lw=2,
        label=f"actual {actual:,.0f}",
    )
    span = centers[keep]
    pad = 0.15 * (span.max() - span.min())
    plt.xlim(
        min(span.min(), actual) - pad,
        max(span.max(), actual) + pad,
    )
    plt.ylim(bottom=0)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.title("The model predicts a distribution, not a number")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _split_series(df, horizon):
    """Split one time series into history and held-out future."""
    history = df.iloc[:-horizon].reset_index(drop=True)
    future = df.iloc[-horizon:].reset_index(drop=True)
    return history.copy(), future.copy()


def load_co2(horizon=24):
    """Return monthly CO2 history plus a held-out future."""
    df = pd.read_csv(CO2_PATH, parse_dates=["month"])
    return _split_series(df, horizon)


def load_electricity(horizon=48):
    """Return hourly electricity prices plus a held-out future."""
    df = pd.read_csv(ELECTRICITY_PATH, parse_dates=["hour"])
    return _split_series(df, horizon)


def plot_co2(history, months=120):
    """Plot the tail of the CO2 series."""
    recent = history.iloc[-months:]
    plt.figure()
    plt.plot(recent["month"], recent["co2_ppm"])
    plt.ylabel("CO2 (ppm)")
    plt.title("Monthly CO2 at Mauna Loa: trend plus seasonality")
    plt.tight_layout()
    plt.show()


def plot_electricity(history, hours=336):
    """Plot the tail of the hourly electricity price series."""
    recent = history.iloc[-hours:]
    plt.figure()
    plt.plot(recent["hour"], recent["price_eur"])
    plt.ylabel("price (EUR/MWh)")
    plt.title("Hourly day-ahead electricity prices: a daily cycle")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_forecast_comparison(
    history,
    future,
    forecasts,
    time_col,
    target_col,
    ylabel,
    title,
    band=None,
    context=72,
):
    """Compare forecasts, optionally with a prediction band."""
    recent = history.iloc[-context:]
    plt.figure()
    plt.plot(
        recent[time_col],
        recent[target_col],
        color="gray",
        label="history",
    )
    plt.plot(
        future[time_col],
        future[target_col],
        color="black",
        lw=2,
        label="actual future",
    )
    for label, values in forecasts.items():
        plt.plot(
            future[time_col], values, "--", lw=2, label=label
        )
    if band is not None:
        low, high = band
        plt.fill_between(
            future[time_col],
            low,
            high,
            color="tab:blue",
            alpha=0.2,
            label="10th to 90th percentile",
        )
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

def display_precomputed():
    """Show the precomputed benchmark numbers (no re-run needed)."""
    labels = ["preprocessing only", "KV cache"]
    formats = {
        "median": "{:.0f}",
        "25th percentile": "{:.0f}",
        "75th percentile": "{:.0f}",
    }

    model_pred = pd.DataFrame({
        "median": [2625, 272],
        "25th percentile": [2609, 270],
        "75th percentile": [2642, 276],
    }, index=labels)
    show_table(model_pred, "Model prediction (ms)", formats)
    print(f"Model prediction (ms): uncached / cached = {2625 / 272:.2f}x")

    whole_request = pd.DataFrame({
        "median": [3853, 1524],
        "25th percentile": [3827, 1482],
        "75th percentile": [3889, 1584],
    }, index=labels)
    show_table(whole_request, "Whole request (ms)", formats)
    print(f"Whole request (ms): uncached / cached = {3853 / 1524:.2f}x")