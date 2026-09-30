"""Metrics on the original dataset label scale; CI excludes tied labels."""
import math
from typing import Sequence
import numpy as np

def _finite_1d(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array

def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks

def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def _concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact O(n log n) CI; tied predictions contribute one half."""

    if y_true.size < 2:
        return 0.0
    order = np.argsort(y_true, kind="mergesort")
    labels = y_true[order]
    predictions = y_pred[order]
    unique_predictions = np.unique(predictions)
    ranks = np.searchsorted(unique_predictions, predictions) + 1
    tree = np.zeros(unique_predictions.size + 2, dtype=np.int64)

    def add(index: int) -> None:
        while index < tree.size:
            tree[index] += 1
            index += index & -index

    def prefix_sum(index: int) -> int:
        total = 0
        while index > 0:
            total += int(tree[index])
            index -= index & -index
        return total

    concordant = 0.0
    comparable = 0
    previous = 0
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and labels[end] == labels[start]:
            end += 1
        for rank in ranks[start:end]:
            less = prefix_sum(int(rank) - 1)
            equal = prefix_sum(int(rank)) - less
            concordant += less + 0.5 * equal
            comparable += previous
        for rank in ranks[start:end]:
            add(int(rank))
        previous += end - start
        start = end
    return float(concordant / comparable) if comparable else 0.0

def compute_native_affinity_metrics(
    y_true_native: Sequence[float] | np.ndarray,
    y_pred_normalized: Sequence[float] | np.ndarray,
    train_mean: float,
    train_std: float,
) -> dict[str, float]:
    """Return metrics in the dataset's original affinity-label units."""

    labels = _finite_1d(y_true_native, name="y_true_native")
    predictions_normalized = _finite_1d(
        y_pred_normalized, name="y_pred_normalized"
    )
    if labels.shape != predictions_normalized.shape:
        raise ValueError(
            "labels and predictions differ in shape: "
            f"{labels.shape} vs {predictions_normalized.shape}"
        )
    scale = float(train_std)
    mean = float(train_mean)
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("train label mean/std must be finite and std must be positive")
    predictions = predictions_normalized * scale + mean
    residual = predictions - labels
    mse = float(np.mean(residual * residual))
    total_variance = float(np.sum((labels - np.mean(labels)) ** 2))
    metrics = {
        "MSE": mse,
        "RMSE": float(math.sqrt(mse)),
        "MAE": float(np.mean(np.abs(residual))),
        "Pearson": _pearson(labels, predictions),
        "Spearman": _pearson(_average_ranks(labels), _average_ranks(predictions)),
        "CI": _concordance_index(labels, predictions),
        "R2": (
            0.0
            if total_variance == 0.0
            else float(1.0 - np.sum(residual * residual) / total_variance)
        ),
        "SD": float(np.std(residual, ddof=0)),
        "samples": float(labels.size),
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError(f"native affinity metrics are non-finite: {metrics}")
    return metrics
