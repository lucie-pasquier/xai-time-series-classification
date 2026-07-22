## Decision: aeon over sktime (28 May 2026)
Choice: aeon for time series data loading.
Alternatives considered: sktime, custom .ts parser.
Reasoning: aeon is the actively-maintained fork of sktime focused on TSC.
Lighter dependency footprint. APIs are nearly identical so swap cost
is low if needed for paper-repo compatibility later.
Risk: some older paper repos use sktime; will translate calls or
install both as needed.

## Decision: per-sample z-score normalisation (28 May 2026)
Choice: each ECG trace normalised against its own mean and std.
Alternatives considered: per-dataset z-scoring (fit stats on train, apply
to test); no normalisation.
Reasoning: UCR archive convention for time series classification, used by
most benchmarks on ECG200. Removes inter-recording baseline drift.
Avoids the train/test leakage concern of per-dataset scoring because
no statistics are fit from training data.
Note: visual difference on ECG200 specifically is small because the
UCR data appears already centred — but normalisation remains the
correct defensive default and is required for cross-paper comparability.

## Decision: region grid for faithfulness harness (21 Jul 2026, revised 21 Jul 2026)
Choice: region size set as a percentage of input length — 10% primary
(data-justified anchor) and 5% secondary (fine grid). The percentage idea
follows Šimić, Veas & Sabol (2025), but the sizes are chosen from ECG200's
own scale (see Revision below), not inherited.
Non-integer rule: fix the region COUNT at round(100/pct), then split the
timeline into that many contiguous, near-equal regions with the remainder
distributed so sizes differ by at most 1. For length 96:
10% → 10 regions (4×9 + 6×10, avg 9.6); 5% → 20 regions (4×4 + 16×5, avg 4.8).
Alternatives considered: rounding the region LENGTH to whole timesteps.
Rejected: it drifts from the target percentage (e.g. a 2.5% region on length
96 rounds 2.4 → 2 or 3, i.e. 48 or 32 regions ≈ 2.08% / 3.125%).
Reasoning: fixing the count keeps every region as close to the intended
percentage as integer timesteps allow, and guarantees gapless,
non-overlapping coverage. The partition is deterministic per (length, pct).
Single source of truth: the grid lives in src/xai/regions.py and is the
one definition both the XAI relevance side and the CMI perturbation side
must reference — any divergence would silently corrupt faithfulness
results.
Revision (21 Jul 2026): the fine grid was initially set to 2.5% (inherited
from Šimić et al.). It was superseded by 5% after the autocorrelation
analysis in notebook 01b showed ECG200's natural scale is ~10 timesteps:
2.5% (~2–3 ts) falls ~4× below that and fragments coherent features, whereas
5% (~5 ts) is a data-matched fine resolution and 10% (~10 ts) sits right on
the correlation length. This deliberately prioritises data-justified sizes
over numeric label-comparability with Šimić — whose 2.5% was applied to much
longer signals and so was a different physical scale anyway. Recorded as a
revision (not a silent edit) to preserve the reasoning trail.

## Decision: perturbation methods for the harness (21 Jul 2026)
Choice: three perturbation methods (PMs) as the harness defaults, matching
Šimić, Veas & Sabol (2025, Table 4): Zero (replace region with 0),
SampleMean (replace with the region-signal's own mean), Laplace (replace
with a smoothed/convolved version of the signal). Built in
src/xai/perturbation.py; consumes the layer-1 RegionGrid (region identified
by grid index, no boundaries redefined). Dataset-agnostic: length read from
the grid/signal, no class count assumed (reusable for Sleep-EDF later).
SampleMean = sample-level, not training-set. Reasoning: matches the paper's
per-sample PM definitions and keeps each perturbation self-contained (no
leakage of a global/training statistic into a single-sample perturbation),
consistent with the per-sample z-scoring already used for the data.
Consequence on ECG200: because signals are per-sample z-scored (mean ≈ 0),
SampleMean ≈ Zero on this dataset; the two PMs diverge only on non-zero-mean
data (e.g. Sleep-EDF) — a reason to keep all three and stay dataset-agnostic.
Laplace interpretation (Table 4's exact kernel not transcribed — flagged, not
guessed silently): (a) kernel = normalised discrete Laplace/double-exponential
k[i] ∝ exp(-|i|/scale) — a low-pass smoother, NOT the Laplacian [1,-2,1]
sharpening operator; (b) scale/radius are explicit tunable params
(defaults 2.0 / 8), not read from the paper; (c) edges = reflect-pad the
signal to 'same' length; (d) the whole sample is smoothed and the smoothed
values copied into the region, so a short region is blurred using real
neighbouring context. If Table 4 specifies different values, change the
kernel/scale/radius/pad-mode in one place and nothing else moves.
Correction (22 Jul 2026): the Laplace interpretation above was WRONG. Layer 2
initially implemented Laplace as a double-exponential SMOOTHING kernel
(exp(-|i|/scale), low-pass) — a misreading of the ambiguous Table 4. After
cloning and inspecting the authors' official source
(github.com/perturbationeffect/cmi-am-validation-for-dl-ts-classifiers,
utils/subsequence_perturbation.py, class Laplace), the authoritative behaviour
is scipy.ndimage.laplace(original): convolution with the discrete Laplacian
SECOND-DERIVATIVE kernel [1, -2, 1], boundary mode 'reflect' — an edge/curvature
operator, not a smoother. src/xai/perturbation.py was corrected to call
scipy.ndimage.laplace(mode='reflect') and now matches their output exactly
(verified np.allclose on ECG200). Behavioural change: the buggy smoothing hugged
the original signal and hid little (we had wrongly called Laplace "the gentlest"
PM); the correct operator collapses a feature's amplitude toward ~0 (jagged
curvature), so it is roughly as aggressive as Zero. The layer-2 figure/result
narration were updated accordingly. This is cited reuse of Apache-2.0 code:
attribution + license preservation recorded in the NOTICE file and in a comment
block at the Laplace implementation.

## Decision: layer 3 — deletion / perturbation curves (22 Jul 2026)
Structure: src/xai/deletion_curves.py builds MoRF (most-relevant-first) and LeRF
(least-relevant-first) perturbation curves for a single signal, given a
predict_proba callable, a layer-1 RegionGrid, a per-region relevance vector, and
a layer-2 perturbation method. Regions are hidden cumulatively; each curve records
the originally-predicted class's probability after each step. Added a
perturb_regions() primitive to perturbation.py (background computed once from the
ORIGINAL signal, applied to all hidden regions) so cumulative perturbation matches
the reference; perturb_region() now delegates to it.
50% stop: n_steps = ceil(n_regions * 0.5), per Šimić et al.
Output format matched to reference CMI input (KEY): curves are the predicted-class
probability on a 0–100 scale, index 0 = unperturbed, MoRF/LeRF equal length —
exactly what utils/res_utils.py decaying_degradation_score(morf_values, lerf_values)
consumes, so the layer-4 metric adaptation needs no reformatting. (Their DDS
normalises by a max of 100, hence the ×100 scaling.)
Dataset-agnostic: length/region count from the grid, n_classes from predict_proba
width (tracks the predicted class, no binary assumption), 50% stop is a ratio.
Cited reuse (Apache 2.0): MoRF/LeRF construction adapted from their
interpret_model_regions.py; recorded in NOTICE and the module docstring.
Validation (dummy attribution, no real XAI yet): on a representative test beat a
QRS-focused dummy gives a steep MoRF vs shallow LeRF (gap ≈ +61 on the 10% grid,
Zero PM), while a random dummy gives tangled curves (gap ≈ 0) — confirming the
curve logic is correct.

## Decision: layer 4 — DDS / PES / CMI (22 Jul 2026)
Adapted from the authors' reference (Apache 2.0), utils/res_utils.py:
decaying_degradation_score, pes, CMI — lifted into src/xai/cmi.py with the
COMPUTATION UNCHANGED (cubic front-loaded weighting of the LeRF−MoRF gap;
sign-counting PES = f−u; CMI = harmonic mean of |DDS|,|PES| with conflicting-
signs/zero → 0). Modifications stated (Apache marking): (1) exposed the hard-coded
normalisation constant 100 as a named param prob_scale=100.0; (2) added a
compute_cmi(morf_curves, lerf_curves) wrapper chaining per-sample DDS → PES →
mean-DDS → CMI, matching how their results notebook combines them; (3) CMI returns
float 0.0. Attribution recorded in the module header and NOTICE.
Ranges: DDS,PES in [−1,1]; CMI in [0,1]; higher = more faithful.
Prediction-scale decision (explicit): the reference DDS normalises by max=100,
i.e. assumes 0–100 curves. Our layer-3 curves are ALREADY 0–100 (they ×100 the
probabilities), so scales already match — neither curve values nor the constant
were changed; the constant was only surfaced as prob_scale for visibility. No
silent rescale.
Validation (end-to-end, whole test set, 10% grid, Zero PM): oracle attribution
(relevance = true per-region single-deletion importance) → CMI≈0.65, DDS≈+0.48,
PES=+1.00 → the full stack works. Random control → CMI≈0.06 (≈0). NOTE: the
QRS-focused "sensible" dummy → CMI≈0.03, i.e. it did NOT beat random. This is not
a bug: CMI correctly withholds credit from a plausible-but-unfaithful attribution.
Cause = the dual feature-space issue — Model 1 decides in frequency-band space,
not on the QRS time-region, so a time-localised attribution is not its ground
truth. Only the oracle (the model's actual per-region reliance) scores high.
Range checks (DDS,PES∈[−1,1], CMI∈[0,1]) pass on real inputs.

## Decision: layer 5 — concentration measure (22 Jul 2026)
Purpose: a SUPPORTING measure (not a headline metric) reported alongside CMI, to
separate "faithfulness changed" from "the model's reliance became more/less
diffuse" when comparing CMI across models of increasing complexity. A concentrated
model gives a steep degradation curve and high CMI ceiling regardless of
attribution quality; concentration quantifies that confound.
Measure chosen: concentration = 1 − normalised Shannon entropy of the per-region
importance distribution (H / log(n_regions)). Range [0,1]: 0 = maximally diffuse
(reliance uniform across regions), 1 = maximally concentrated (one region carries
all reliance). Chosen over Gini because it is standard/interpretable, the log(n)
normalisation makes it comparable across grids/models with different region counts
(needed for cross-model comparison), and it hits both extremes exactly. Built in
src/xai/concentration.py (concentration_from_importances / region_reliance /
region_concentration / dataset_concentration). Dataset-agnostic (region count from
grid, n_classes from predict_proba width, no binary assumption).
Computed from the MODEL, not attributions: importance = the model's true
per-region reliance (single-deletion |Δ predicted-class prob|, Zero PM — the same
ground-truth the layer-4 oracle used). Deriving it from an XAI attribution would
measure the attribution, not the model, defeating the purpose.
Synthetic-extremes check: single active region → 1.0, perfectly uniform → 0.0,
two-equal → ~0.70 (guards: all-zero → 0.0, n=1 → NaN). Behaves correctly.
Model 1 calibration value: mean concentration ≈ 0.24 (std 0.15) on the 10% grid /
test set, i.e. effectively ~5.7 of 10 regions relied on, ~90% of samples < 0.4 —
Model 1 is DIFFUSE, as its whole-signal band-power features predict. This anchors
the cross-model comparison: a CMI change in Models 2–5 accompanied by a
concentration change is (partly) the diffuseness confound, not pure faithfulness.

## Decision: layer 6 — FeatureAblation (validation control) (22 Jul 2026)
Role: FeatureAblation is the FIRST real attribution method, wired in as a
VALIDATION CONTROL for the whole stack — not a headline method. Its scoring
(delete a region, measure prediction drop) is near-identical to how CMI evaluates
explanations, so it should score high by construction; a low score would mean
broken wiring, not a poor method.
Wiring (Captum, not hand-rolled): used captum.attr.FeatureAblation — the same
library the reference paper used — so Model 1 stays on the same code path as the
PyTorch Models 2–5. Model 1 is a scikit-learn pipeline, so its predict_proba is
wrapped in a small torch forward_func (FeatureAblation is gradient-free, so no
nn.Module is needed). Built in src/xai/feature_ablation.py. Captum cited (Kokhlikyan
et al. 2020, BSD-3-Clause) in the module docstring and NOTICE.
Grid-native: attribution grouped by the layer-1 RegionGrid — Captum feature_mask =
grid.labels, ablation baseline = the layer-2 PM background. Output is per-region
relevance, directly consumable by layer 3. No ad-hoc per-timestep pooling.
Consistency guarantee: computed in float64 so Captum FeatureAblation returns
bit-for-bit the same per-region reliance as the hand-rolled region_reliance
(layer 5) for ALL three PMs (max|diff| = 0.0). (float32 differed only by ~1e-7
rounding.) Guarantees Model 1 (Captum-wrapped sklearn) == Models 2–5 (native
Captum) semantics. dtype is a param (default float64; native-torch models can pass
float32). PM matched: FeatureAblation's baseline = the CMI deletion PM.
Dataset-agnostic (regions from grid, class from predict_proba width).
Validation outcome (Model 1, test set, per-sample attributions): FeatureAblation
CMI = 0.648 (SampleMean), 0.648 (Zero), 0.630 (Laplace), all PES = +1.00 — right at
the oracle (0.648) and ~10x above random (0.06) / QRS (0.03). FA(Zero) equals the
oracle exactly because with a zero baseline FeatureAblation computes the oracle's
per-region reliance. => the full 6-layer stack (grid → perturbation → curves →
DDS/PES/CMI + concentration) is validated end-to-end with a genuine XAI method.
Thesis note: this is the first cell of the model × method × CMI table
(Model 1 × FeatureAblation ≈ 0.65). Do NOT over-read: (a) FeatureAblation is a
near-circular control (upper reference, not competitive); (b) Model 1 is diffuse
(concentration ≈ 0.24, lower CMI ceiling — always report the pair) and lives in
band-power/frequency space, not the time-region space perturbed here (dual
feature-space), so its row is an anchor, not a like-for-like comparison with the
raw-signal Models 2–5.

## Decision: KernelSHAP as the Shapley-based through-line (TimeSHAP set aside) (22 Jul 2026)
Choice: the primary Shapley-based attribution run on all five models is KernelSHAP
via captum.attr.KernelShap with feature_mask = grid.labels — "KernelSHAP over
temporal regions" (Lundberg & Lee 2017). Built in src/xai/kernel_shap.py alongside
feature_ablation.py (new module, same patterns): grid-native (feature_mask =
grid.labels, baseline = layer-2 PM background, return_input_shape=False → one
Shapley value per region, no pooling); same sklearn predict_proba→torch
forward_func wrapper for Model 1, native for Models 2–5; dtype param (float64 for
Model 1). Dataset-agnostic (class from predict_proba width).
TimeSHAP considered and set aside (Bento et al. 2021, KDD — "KernelSHAP adapted for
sequences", originally named as the through-line in the background report). Deciding
factor: grid incompatibility — TimeSHAP imposes its own event (timestep) / feature /
cell segmentation with a pruning algorithm and offers NO user-defined region
grouping; pooling its per-timestep Shapley values into our regions is NOT equivalent
to Shapley values over our region coalitions (the attribution and CMI perturbation
must reference identical regions). Secondary: dependency risk (pins shap<=0.42.1,
issue #56; likely conflicts with our NumPy 2.x stack) and low maintenance; and
TimeSHAP is NOT benchmarked in Šimić et al., whereas KernelSHAP IS (they used
captum KernelShap too). Captum KernelShap: same library already in the stack, zero
new deps, uses the identical feature_mask mechanism as our FeatureAblation.
Methodological independence: KernelSHAP averages contributions over sampled
coalitions — a different computation from CMI's single ordered deletion (MoRF/LeRF)
— so it is NOT circular with the metric, unlike FeatureAblation (which with a zero
baseline computes the oracle quantity). Honest nuance recorded: it still perturbs
with a baseline, so not fully orthogonal to perturbation.
Stochasticity / reproducibility: KernelSHAP samples coalitions -> results vary.
kernel_shap() seeds BOTH torch and numpy (seed = 0) at every call and uses
n_samples = 200 coalitions; verified reproducible (identical CMI on rerun).
Dry-run gate (before building): captum KernelShap through the Model-1 wrapper,
feature_mask=grid.labels, PM baseline, return_input_shape=False → returned (1,10),
one coeff per region. NB Captum returns the coefficients in float32 regardless of
input dtype (its solver is float32) — immaterial for a stochastic estimate.
Result (Model 1, test set, seed 0, n_samples 200): KernelSHAP CMI = 0.612
(SampleMean), 0.612 (Zero), 0.624 (Laplace); DDS ≈ +0.44; PES = +1.00. Just below
FeatureAblation (0.65/0.65/0.63), well above random (0.06) / QRS (0.03), near the
oracle (0.65). Reference check CONFIRMED: KernelSHAP ranked in Šimić et al.'s
WEAKEST bracket (mean rank ≈ 9.9 of ~11–12 methods) — but on DEEP models. Model 1's
high score reflects its (near-)linear band-power design (Shapley is near-exact for
linear models); expect KernelSHAP's CMI to fall on the complex Models 2–5.