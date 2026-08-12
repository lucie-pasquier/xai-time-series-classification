"""
sleep_edf/xai_bandpower.py
──────────────────────────────────────────────────────────────────────────────
Glue for applying the harness XAI methods to the band-power logistic baseline
(Model 1b), and computing CMI per method. Single source of truth so the notebook
imports rather than redefines.

The band-power model classifies a 5-dim vector of standardised band powers
(delta/theta/alpha/sigma/beta). We therefore attribute over those 5 BANDS: the
"signal" handed to the harness is the 5-band feature vector, and the RegionGrid
has 5 regions (one band each), so each region's attribution is one band's
importance — directly comparable to the model's own per-band coefficients (the
known ground truth, esp. N3 ↔ delta).

- Black-box methods (FeatureAblation, KernelSHAP) take a numpy predict_proba.
- Integrated Gradients needs a torch model, so `BandLinearTorch` is a torch view
  of the (multinomial) logistic regression. For a LINEAR model IG is near-trivial
  (it approximately reduces to feature × weight) — included for method consistency
  across the ladder, not because it adds insight here.

Nothing here touches harness internals; it only imports the harness public API.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from harness.xai.feature_ablation import feature_ablation
from harness.xai.kernel_shap import kernel_shap
from harness.xai.integrated_gradients import integrated_gradients
from harness.xai.deletion_curves import perturbation_curves
from harness.xai.cmi import compute_cmi
from harness.xai.concentration import concentration_from_importances


def band_predict_proba(coef, intercept):
    """Return pp(F) -> class probabilities for the multinomial band-power logreg.

    coef: (n_classes, n_bands), intercept: (n_classes,). F: (n, n_bands) standardised
    band features. Softmax(F·coefᵀ + intercept) reproduces sklearn's predict_proba
    for a multinomial (lbfgs) logistic regression, so no fitted sklearn object needed.
    """
    coef = np.asarray(coef, float)
    intercept = np.asarray(intercept, float)

    def pp(F):
        F = np.atleast_2d(np.asarray(F, float))
        z = F @ coef.T + intercept
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    return pp


class BandLinearTorch(nn.Module):
    """Torch view of the band-power logreg (for gradient-based IG).

    Input (n, 1, n_bands) or (n, n_bands) -> logits (n, n_classes). Weights copied
    from the trained logreg, so softmax(logits) == band_predict_proba.
    """

    def __init__(self, coef, intercept):
        super().__init__()
        coef = np.asarray(coef, float)
        intercept = np.asarray(intercept, float)
        n_class, n_feat = coef.shape
        self.lin = nn.Linear(n_feat, n_class).double()
        with torch.no_grad():
            self.lin.weight.copy_(torch.tensor(coef, dtype=torch.float64))
            self.lin.bias.copy_(torch.tensor(intercept, dtype=torch.float64))

    def forward(self, x):
        if x.dim() == 3:                      # (n, 1, n_bands) -> (n, n_bands)
            x = x.reshape(x.shape[0], -1)
        return self.lin(x)


def method_attributions_and_cmi(kind, signals, labels, grid, coef, intercept,
                                n_classes, pm="zero"):
    """Apply ONE attribution method over the band-features and compute its CMI.

    Parameters
    ----------
    kind : {"feature_ablation", "kernel_shap", "integrated_gradients"}
    signals : ndarray (n_epochs, n_bands)
        Standardised band-power vectors to attribute over.
    labels : ndarray (n_epochs,)
        True class per epoch (used to build the per-class attribution view).
    grid : RegionGrid
        Grid over the n_bands features (one region per band).
    coef, intercept : the trained logreg parameters.
    n_classes : int
    pm : str
        Perturbation method / ablation baseline ("zero" — for standardised features,
        zero == the mean, i.e. ablate the band's deviation).

    Returns
    -------
    dict:
        "per_class_attr" : (n_classes, n_bands) — mean attribution per band for each
                           stage, attributing toward that stage (the "which band"
                           view to compare against the model coefficients).
        "cmi"            : compute_cmi(...) dict {CMI, DDS, PES, dds_per_sample} over
                           the pooled sample (curves track the predicted class).
        "concentration"  : mean concentration_from_importances over the sample.
    """
    pp = band_predict_proba(coef, intercept)
    tmodel = BandLinearTorch(coef, intercept) if kind == "integrated_gradients" else None

    def attribute(sig, target_class):
        if kind == "feature_ablation":
            return feature_ablation(pp, sig, grid, pm, target_class=target_class)
        if kind == "kernel_shap":
            return kernel_shap(pp, sig, grid, pm, target_class=target_class)
        if kind == "integrated_gradients":
            return integrated_gradients(tmodel, sig, grid, pm, target_class=target_class)
        raise ValueError(f"unknown method kind {kind!r}")

    # (1) per-class attribution — attribute each epoch toward its OWN stage, average.
    per_class = np.full((n_classes, grid.n_regions), np.nan)
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        if len(idx):
            per_class[c] = np.mean([attribute(signals[i], c) for i in idx], axis=0)

    # (2) CMI + concentration over the pooled sample (curves follow predicted class).
    morf, lerf, conc = [], [], []
    for i in range(len(signals)):
        a = attribute(signals[i], None)
        cur = perturbation_curves(pp, signals[i], grid, a, method=pm)
        morf.append(cur["MoRF"]); lerf.append(cur["LeRF"])
        conc.append(concentration_from_importances(a))
    cmi = compute_cmi(morf, lerf)
    return {"per_class_attr": per_class, "cmi": cmi, "concentration": float(np.mean(conc))}


def plot_band_attribution(per_class_attr, coef, band_names, class_names,
                          method_name, save_path=None, highlight_row=None):
    """Two-panel heatmap: (left) this method's per-band attribution per stage,
    (right) the model's own coefficients (the known ground truth) — so one can SEE
    whether each stage's attribution lands on the physiologically-expected band
    (above all N3 → delta). Row-normalised for shape-comparability.

    highlight_row : int or None
        If given, outline that stage's row in both panels (use for the N3 primary
        ground-truth check, so N3 is easy to read).
    """
    import matplotlib.pyplot as plt

    def _rownorm(M):
        M = np.asarray(M, float)
        denom = np.nanmax(np.abs(M), axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return M / denom

    A = _rownorm(per_class_attr)
    C = _rownorm(coef)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, M, title in [(ax[0], A, f"{method_name}: attribution (per stage x band)"),
                        (ax[1], C, "model coefficients (ground truth)")]:
        im = a.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                a.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                       color="white" if abs(M[i, j]) > 0.6 else "black")
        a.set_xticks(range(len(band_names))); a.set_xticklabels(band_names)
        a.set_yticks(range(len(class_names))); a.set_yticklabels(class_names)
        a.set_xlabel("band"); a.set_title(title)
        if highlight_row is not None:
            a.add_patch(plt.Rectangle((-0.5, highlight_row - 0.5), M.shape[1], 1,
                                      fill=False, edgecolor="lime", lw=3))
    ax[0].set_ylabel("sleep stage")
    fig.colorbar(im, ax=ax, fraction=0.03, label="row-normalised weight")
    hl = f" (highlighted: {class_names[highlight_row]})" if highlight_row is not None else ""
    fig.suptitle(f"{method_name} vs model ground truth — does N3 land on delta?{hl}", y=1.02)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
