'''Plumbing for pretrain_nanotabpfn.ipynb: inference, evaluation data and plots.
The notebook contains the parts worth reading: the prior, the model and the training loop.'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

NOISE_STD = 0.1  # must match the noise level of the prior in the notebook

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------- in-context inference
@torch.no_grad()
def predict_proba(model, X_train, y_train, X_test):
    '''One forward pass on one table (labeled rows + test rows). Returns P(class 1) for every test row.'''
    model.eval()
    as_batch = lambda a: torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)  # a batch of one table
    logits = model(as_batch(X_train), as_batch(y_train), as_batch(X_test)).squeeze(0)
    model.train()
    return F.softmax(logits, dim=-1)[:, 1].cpu().numpy()


# ---------------------------------------------------------------- evaluation data
EVAL_FUNCTIONS = {  # name: (intercept a, slope b, amplitude A, frequency w, phase phi) of the class boundary
    "line": (0.0, -0.3, 0.0, 0.0, 0.0),
    "sine wave": (1.0, 0.3, 1.5, 2.0, 0.0),
    "sine + fast wave": (-1.0, 0.5, 1.0, 4.0, 1.0),
}


def wave_function(x, a, b, amp, w, phi):
    return a + b * x + amp * np.sin(w * x + phi)


def load_eval_datasets(num_train=40, num_test=200, seed=42):
    '''Three fixed 2-d datasets: class 1 above a boundary x2 = f(x1) (a line, or a line plus a sine), class 0 below it.'''
    rng = np.random.RandomState(seed)
    datasets = {}
    for name, params in EVAL_FUNCTIONS.items():
        n = num_train + num_test
        x1 = rng.uniform(-2, 2, n)
        x2 = wave_function(x1, *params) + rng.randn(n)  # points spread around the boundary
        y = (x2 + NOISE_STD * rng.randn(n) > wave_function(x1, *params)).astype(int)
        datasets[name] = train_test_split(np.stack([x1, x2], 1), y, train_size=num_train, test_size=num_test, random_state=seed, stratify=y)
    return datasets


EVAL_DATASETS = load_eval_datasets()


def evaluate(predict_fn, datasets=EVAL_DATASETS):
    '''Returns {dataset: accuracy} for any predict_fn(X_train, y_train, X_test) -> P(class 1).'''
    return {name: accuracy_score(y_te, predict_fn(X_tr, y_tr, X_te) > 0.5) for name, (X_tr, X_te, y_tr, y_te) in datasets.items()}


def evaluate_model(model):
    '''Mean accuracy of the model over the three test datasets.'''
    return float(np.mean(list(evaluate(lambda X_tr, y_tr, X_te: predict_proba(model, X_tr, y_tr, X_te)).values())))


def compare_with_baseline(models):
    '''Prints the test accuracy per test dataset for each {name: model} and for two baselines fitted on the labeled points.'''
    predict_fns = {name: (lambda X_tr, y_tr, X_te, m=model: predict_proba(m, X_tr, y_tr, X_te)) for name, model in models.items()}
    predict_fns["logistic regression"] = lambda X_tr, y_tr, X_te: LogisticRegression().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    predict_fns["random forest"] = lambda X_tr, y_tr, X_te: RandomForestClassifier(random_state=0).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    widths = [len(d) + 3 for d in EVAL_DATASETS]  # narrow columns, so a row fits in the notebook without wrapping
    header = f"{'accuracy':20s}" + "".join(f"{d:>{w}s}" for d, w in zip(EVAL_DATASETS, widths)) + f"{'mean':>7s}"
    print(header); print("-" * len(header))
    for name, fn in predict_fns.items():
        scores = [accuracy_score(y_te, fn(X_tr, y_tr, X_te) > 0.5) for X_tr, X_te, y_tr, y_te in EVAL_DATASETS.values()]
        print(f"{name:20s}" + "".join(f"{s:{w}.3f}" for s, w in zip(scores, widths)) + f"{np.mean(scores):7.3f}")


# ---------------------------------------------------------------- plots
LEVELS = [0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1]  # probability bands: white in the middle means "unsure"


def _scatter_classes(ax, X, y, s=30):
    ax.scatter(*X[y == 1].T, s=s, color="red", marker="o", edgecolor="white", linewidth=0.5, label="class 1")
    ax.scatter(*X[y == 0].T, s=s, color="blue", marker="x", label="class 0")


def plot_prior_samples(x, y, boundary):
    '''boundary: function mapping x1 values (1, num_points) to the boundary of every table (n, num_points).'''
    grid = torch.linspace(x[..., 0].min().item(), x[..., 0].max().item(), 300).unsqueeze(0)
    curves = boundary(grid)
    fig, axes = plt.subplots(1, len(x), figsize=(3.2 * len(x), 3.4))  # one panel per table
    for i, ax in enumerate(axes):
        lo, hi = x[i, :, 0].min().item(), x[i, :, 0].max().item()
        inside = (grid[0] >= lo) & (grid[0] <= hi)  # draw the curve only over the range of this table's points
        ax.plot(grid[0, inside], curves[i, inside], color="black", linestyle="--", linewidth=1)
        _scatter_classes(ax, x[i].numpy(), y[i].numpy().astype(int), s=15)
        ax.set_title(f"table {i}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("Synthetic tables sampled from the prior (x1 vs x2, colored by class, dashed: boundary)")
    plt.tight_layout(); plt.show()


def plot_predictions(model, resolution=150):
    '''P(class 1) on a grid, with the labeled points the model gets as context and the true boundary.'''
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, name in zip(axes, EVAL_DATASETS):
        X_tr, X_te, y_tr, y_te = EVAL_DATASETS[name]
        X_all = np.concatenate([X_tr, X_te])
        g1, g2 = np.meshgrid(np.linspace(-2, 2, resolution), np.linspace(X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5, resolution))
        p = predict_proba(model, X_tr, y_tr, np.stack([g1.ravel(), g2.ravel()], 1)).reshape(g1.shape)
        cs = ax.contourf(g1, g2, p, levels=LEVELS, cmap="RdBu_r", alpha=0.8)
        ax.plot(g1[0], wave_function(g1[0], *EVAL_FUNCTIONS[name]), color="black", linestyle="--", linewidth=1, label="true boundary")
        _scatter_classes(ax, X_tr, y_tr)
        ax.set_title(name); ax.set_xlabel("x1")
    axes[0].set_ylabel("x2"); axes[0].legend(fontsize=8, loc="lower right")
    fig.colorbar(cs, ax=axes, label="P(class 1)", shrink=0.9)
    plt.suptitle("nanoTabPFN predicted class probabilities (white = unsure), with the 40 labeled points as context")
    plt.show()
