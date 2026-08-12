"""
src/models/cnn.py
──────────────────────────────────────────────────────────────────────────────
1D CNN architectures for Models 2–4 of the complexity ladder, plus the shared
training recipe reused by Models 3–5.

One parameterised architecture, three variants
    A single `CNN1D` module produces the shallow / medium / deep variants purely
    by changing the list of per-block channel widths (`conv_channels`). Depth =
    len(conv_channels); width = the values. So Models 3 and 4 are just named
    configs in CNN_VARIANTS, not bespoke rebuilds. Complexity (parameter count) is
    varied deliberately by these configs — see DECISIONS_LOG.md for the ladder.

Architecture (per block): Conv1d → BatchNorm1d → ReLU → MaxPool1d(2). After the
    conv stack, Global Average Pooling collapses time to one value per channel,
    then a single Linear head maps to the classes. GAP keeps the head size
    independent of sequence length / depth and preserves a clean per-timestep
    story for later attribution. Kernel size 7 ≈ the ~5–10-timestep autocorrelation
    length of ECG200 (notebook 01b) — each filter sees roughly one coherent unit.

Dataset-agnostic: input length, channel count, and class count are constructor
    arguments read from the data (never hard-coded 96 / 1 / 2).

The shared training recipe (module constants below) is applied by `train_cnn`,
    which Models 3–5 reuse unchanged so the models differ in complexity, not in
    confounded training differences.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from harness.evaluation.metrics import classification_metrics

# ── Complexity ladder: per-block channel widths (depth = len, width = values) ──
CNN_VARIANTS: dict[str, list[int]] = {
    "shallow": [16, 32],            # Model 2 — low end of the ladder
    "medium":  [32, 64, 64],        # Model 3
    "deep":    [64, 128, 128, 256], # Model 4
}

# ── Shared training recipe (reused verbatim by Models 3–5) ────────────────────
SEED = 42
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
MAX_EPOCHS = 200
PATIENCE = 30            # early stopping: epochs without val-loss improvement
WEIGHT_DECAY = 1e-4      # mild L2, helps on the tiny (80-sample) train set
KERNEL_SIZE = 7


def set_seed(seed: int = SEED) -> None:
    """Seed python/numpy/torch for reproducible CPU training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CNN1D(nn.Module):
    """Parameterised 1D CNN: a stack of conv blocks + GAP + linear head.

    Parameters
    ----------
    conv_channels : list[int]
        Output channels for each conv block; its length is the depth.
    n_classes : int
        Number of output classes (read from the data).
    in_channels : int
        Input channel count (1 for ECG200; read from the data).
    kernel_size : int
        Conv kernel size (odd; 'same' padding is applied).
    """

    def __init__(self, conv_channels, n_classes, in_channels=1, kernel_size=KERNEL_SIZE):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in conv_channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(c_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, time)
        x = self.features(x)
        x = self.gap(x).squeeze(-1)     # (batch, channels)
        return self.head(x)             # (batch, n_classes) logits


def build_cnn(variant="shallow", n_classes=2, in_channels=1, kernel_size=KERNEL_SIZE) -> CNN1D:
    """Build a CNN1D for a named ladder variant ('shallow'/'medium'/'deep')."""
    if variant not in CNN_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {tuple(CNN_VARIANTS)}")
    return CNN1D(CNN_VARIANTS[variant], n_classes, in_channels, kernel_size)


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters — the complexity x-axis value."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _to_model_input(X: np.ndarray) -> np.ndarray:
    """Coerce (n, T) → (n, 1, T); pass (n, C, T) through. float32 for torch."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[:, None, :]
    return X


def torch_predict_proba(model: nn.Module, device: str = "cpu"):
    """Return the harness's predict_proba interface for a trained torch model.

    The returned callable maps a batch of raw signals — shape (n, T) or (n, C, T)
    — to class probabilities (n, n_classes), exactly the contract Model 1 exposes
    and the contract FeatureAblation / KernelSHAP / deletion curves call. So a
    trained CNN plugs into the existing harness unchanged.
    """
    def predict_proba(X):
        model.eval()
        X = _to_model_input(X)
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device))
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()
    return predict_proba


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency ('balanced') class weights for CrossEntropyLoss."""
    y = np.asarray(y)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = len(y) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def train_cnn(
    model: nn.Module,
    X_train, y_train,
    X_val, y_val,
    *,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = SEED,
    device: str = "cpu",
    monitor: str = "val_loss",
    min_delta: float = 1e-6,
    mode: str = "min",
) -> dict:
    """Shared training recipe (reused verbatim by Models 3–5).

    Adam + class-weighted cross-entropy (handles the ~65/35 imbalance), early
    stopping on a monitored validation metric with best-checkpoint restore, fixed seed.

    Early-stopping metric (additive; defaults reproduce the original behaviour exactly)
        monitor : {"val_loss", "val_balanced_accuracy"} — the validation quantity watched
            for improvement and used to pick the restored checkpoint. Default "val_loss".
        mode : {"min", "max"} — whether lower (loss) or higher (balanced accuracy) is
            better. Default "min".
        min_delta : float — an epoch counts as an improvement only if it beats the best
            so far by MORE than this margin, so noise doesn't keep resetting patience.
            Default 1e-6 (the original hard-coded loss threshold). With the defaults
            (monitor="val_loss", mode="min", min_delta=1e-6) the stop/restore logic is
            identical to the previous version.
    Both validation quantities are already computed every epoch, so no extra work is
    needed to monitor balanced accuracy.

    Returns
    -------
    dict with keys:
        "history"       : {"train_loss", "val_loss", "val_balanced_accuracy"} lists
        "best_epoch"    : int (1-indexed) of the restored checkpoint (best monitored value)
        "n_classes"     : int
        "monitor"       : str — the metric watched
        "best_metric"   : float — the best monitored value achieved
        "stopped_epoch" : int — the last epoch actually run
        "stop_reason"   : {"patience", "max_epochs"} — why training ended
    The model is left with the best-monitored-metric weights loaded in place.
    """
    if monitor not in ("val_loss", "val_balanced_accuracy"):
        raise ValueError(f"monitor must be 'val_loss' or 'val_balanced_accuracy'; got {monitor!r}")
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max'; got {mode!r}")
    set_seed(seed)
    model.to(device)
    n_classes = int(len(np.unique(y_train)))

    Xtr = torch.from_numpy(_to_model_input(X_train))
    ytr = torch.tensor(np.asarray(y_train), dtype=torch.long)
    Xva = torch.from_numpy(_to_model_input(X_val)).to(device)
    yva = torch.tensor(np.asarray(y_val), dtype=torch.long).to(device)

    loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, n_classes).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_balanced_accuracy": []}
    # Generic monitored-metric early stopping. Defaults (monitor="val_loss", mode="min",
    # min_delta=1e-6) reproduce the previous `val_loss < best_val_loss - 1e-6` behaviour.
    def _improved(current, best):
        return current < best - min_delta if mode == "min" else current > best + min_delta
    best_metric = float("inf") if mode == "min" else float("-inf")
    best_state, best_epoch, since_improved = None, 0, 0
    stopped_epoch, stop_reason = max_epochs, "max_epochs"

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(ytr)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_loss = criterion(val_logits, yva).item()
            val_pred = val_logits.argmax(1).cpu().numpy()
        val_bal = classification_metrics(yva.cpu().numpy(), val_pred)["balanced_accuracy"]

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_balanced_accuracy"].append(val_bal)

        current = val_loss if monitor == "val_loss" else val_bal
        if _improved(current, best_metric):
            best_metric, best_epoch, since_improved = current, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_improved += 1
            if since_improved >= patience:
                stopped_epoch, stop_reason = epoch, "patience"
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "n_classes": n_classes,
            "monitor": monitor, "best_metric": float(best_metric),
            "stopped_epoch": int(stopped_epoch), "stop_reason": stop_reason}
