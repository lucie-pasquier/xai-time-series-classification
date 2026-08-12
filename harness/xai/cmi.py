"""
src/xai/cmi.py
──────────────────────────────────────────────────────────────────────────────
Layer 4 of the faithfulness harness: DDS, PES, and CMI.

Turns the layer-3 MoRF/LeRF perturbation curves into a single faithfulness score.

    DDS (Decaying Degradation Score)  — per sample, in [-1, 1]
        The cubic-weighted average gap (LeRF - MoRF) between the two curves. How
        MUCH faster the prediction falls when the most-relevant regions are hidden
        first. Early perturbation steps are weighted far more than later ones.
    PES (Perturbation Effect Size)    — dataset, in [-1, 1]
        How CONSISTENTLY DDS comes out positive across samples (fraction positive
        minus fraction negative). Magnitude on a few samples is not enough.
    CMI (Consistency-Magnitude-Index) — dataset, in [0, 1]
        Harmonic mean of |DDS| and |PES|; 0 if they disagree in sign. High only
        when the attribution separates important from unimportant regions both
        STRONGLY (DDS) and CONSISTENTLY (PES). Higher = more faithful.

──────────────────────────────────────────────────────────────────────────────
Adapted from the authors' reference implementation (Apache License 2.0)
    repo : https://github.com/perturbationeffect/cmi-am-validation-for-dl-ts-classifiers
    file : utils/res_utils.py
           - decaying_degradation_score(morf_values, lerf_values)
           - pes(dds_vals)
           - CMI(dds, pes)
    paper: Šimić, A., Veas, E., & Sabol, V. (2025). A comprehensive analysis of
           perturbation methods in explainable AI feature attribution validation
           for neural time series classifiers.
    Copyright the above authors; licensed under the Apache License, Version 2.0.
    A copy of the license and attribution is recorded in the repository NOTICE.

Modifications (Apache 2.0 requires stating changes):
    1. The three functions are lifted with their computation UNCHANGED (same
       cubic weighting, same normalisation, same sign-handling), only re-typed
       and re-documented here.
    2. The hard-coded normalisation constant 100 in DDS (the max possible
       per-step curve difference, which encodes a 0–100 probability scale) is
       exposed as a named parameter `prob_scale=100.0`. Default reproduces the
       original behaviour exactly. See "Prediction-scale reconciliation" below.
    3. Added a convenience wrapper `compute_cmi(morf_curves, lerf_curves)` (not in
       the original file) that chains per-sample DDS -> PES -> mean-DDS -> CMI,
       matching how the authors combine them in their results notebook
       (CMI(mean_of_per_sample_DDS, PES)).
    4. CMI returns a float 0.0 (was int 0) for type consistency.

Prediction-scale reconciliation (stated explicitly, not assumed)
    The reference DDS normalises by `max_diffed = 100`, i.e. it ASSUMES the two
    curves are the predicted-class probability on a 0–100 scale (their pipeline
    multiplies softmax outputs by 100). Our layer-3 `perturbation_curves` already
    emits curves on that same 0–100 scale (it multiplies probabilities by 100 for
    exactly this reason). => The scales ALREADY MATCH; we changed neither the
    curve values nor the constant. We only surfaced the constant as `prob_scale`
    (default 100.0) so the assumption is explicit and a future 0–1 pipeline can
    set prob_scale=1.0 rather than silently producing wrong-but-plausible scores.
"""

from __future__ import annotations

import numpy as np


def decaying_degradation_score(
    morf_values,
    lerf_values,
    prob_scale: float = 100.0,
) -> float:
    """Normalised, cubic-weighted gap between the LeRF and MoRF curves (per sample).

    Adapted verbatim (computation unchanged) from the reference
    utils/res_utils.py::decaying_degradation_score. See module header for licence.

    Parameters
    ----------
    morf_values, lerf_values : sequence of float
        The MoRF and LeRF perturbation curves for ONE sample, equal length, with
        index 0 = unperturbed prediction. Expected on a 0–`prob_scale` scale.
    prob_scale : float, default 100.0
        Maximum possible per-step curve difference = the probability scale. The
        reference hard-codes 100 (0–100 %). Our layer-3 curves are on that scale,
        so the default is correct; pass 1.0 for 0–1 probability curves.

    Returns
    -------
    float in [-1, 1]. Positive => MoRF falls faster than LeRF (faithful direction).
    """
    pc_morf = np.array(morf_values, dtype=float)
    pc_lerf = np.array(lerf_values, dtype=float)

    diffed = pc_lerf - pc_morf

    # Cubic, front-loaded weights: earliest perturbation steps count most.
    linear_weights = np.arange(len(diffed), 0, -1) / len(diffed)
    cubic_weights = linear_weights ** 3

    dds = np.average(diffed, weights=cubic_weights)

    # Maximum theoretical difference at each point is `prob_scale`, except index 0
    # (no perturbation there, so both curves are equal => max diff 0).
    max_diffed = np.zeros_like(diffed)
    max_diffed[1:] = prob_scale
    dds_max = np.average(max_diffed, weights=cubic_weights)

    return float(dds / dds_max)   # normalise into [-1, 1]


def pes(dds_vals) -> float:
    """Perturbation Effect Size: sign-consistency of DDS across samples.

    Adapted verbatim from the reference utils/res_utils.py::pes (Kerby's simple
    difference: fraction of positive DDS minus fraction of negative DDS).

    Parameters
    ----------
    dds_vals : sequence of float
        Per-sample DDS values.

    Returns
    -------
    float in [-1, 1]. +1 = every sample positive; 0 = evenly split (or all ties);
    -1 = every sample negative. Ties (DDS == 0) count toward neither.
    """
    dds_vals = np.asarray(dds_vals, dtype=float)
    f = len(dds_vals[dds_vals > 0]) / len(dds_vals)   # favourable fraction
    u = len(dds_vals[dds_vals < 0]) / len(dds_vals)   # unfavourable fraction
    return float(f - u)


def CMI(dds: float, pes: float) -> float:
    """Consistency-Magnitude-Index: harmonic mean of |DDS| and |PES|, else 0.

    Adapted verbatim from the reference utils/res_utils.py::CMI.

    Conflicting-signs rule: if DDS and PES disagree in sign (or either is 0),
    return 0 — the attribution is not reliably separating relevant from irrelevant
    regions, so no meaningful faithfulness value can be reported.

    Parameters
    ----------
    dds : float
        A scalar DDS (the mean of per-sample DDS values), in [-1, 1].
    pes : float
        The PES, in [-1, 1].

    Returns
    -------
    float in [0, 1]. Higher = more faithful.
    """
    if pes * dds <= 0:
        return 0.0
    return 2 / ((1 / abs(dds)) + (1 / abs(pes)))


def compute_cmi(morf_curves, lerf_curves, prob_scale: float = 100.0) -> dict:
    """Convenience wrapper: per-sample curves -> {CMI, DDS, PES, dds_per_sample}.

    NOT in the original reference file. Chains the reference functions the way the
    authors combine them in their results notebook: per-sample DDS, then PES over
    all samples, then CMI of the mean DDS with the PES.

    Parameters
    ----------
    morf_curves, lerf_curves : sequence of curves
        One (MoRF, LeRF) curve pair per sample; each pair equal length. A single
        pair (one sample) is allowed but PES is then trivially ±1 or 0.
    prob_scale : float, default 100.0
        Passed to DDS; see decaying_degradation_score. Layer-3 curves are 0–100,
        so the default is correct.

    Returns
    -------
    dict with keys:
        "CMI"            : float in [0, 1]
        "DDS"            : float in [-1, 1]  (mean of per-sample DDS)
        "PES"            : float in [-1, 1]
        "dds_per_sample" : ndarray of per-sample DDS
    """
    if len(morf_curves) != len(lerf_curves):
        raise ValueError("morf_curves and lerf_curves must have equal length")

    dds_per_sample = np.array([
        decaying_degradation_score(m, l, prob_scale=prob_scale)
        for m, l in zip(morf_curves, lerf_curves)
    ])
    dds_mean = float(np.mean(dds_per_sample))
    pes_value = pes(dds_per_sample)
    cmi_value = CMI(dds_mean, pes_value)

    return {
        "CMI": cmi_value,
        "DDS": dds_mean,
        "PES": pes_value,
        "dds_per_sample": dds_per_sample,
    }
