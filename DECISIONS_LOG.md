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

## Decision: generalise the harness to any #classes and #channels (1 Aug 2026)
Motivation: committed to Sleep-EDF (5-class, potentially multi-channel). Principle:
read dimensions from the data, assume nothing about their values; generalise by
removing hard-coded assumptions, not by adding config for hypothetical scenarios.
Scope = shared harness only (src/xai/). Dataset-specific modules (loader,
preprocessing/z-scoring, region-size choice, Model 1 bandpass) untouched.
Multi-class: was ALREADY generic — the harness reads n_classes from predict_proba's
output width and tracks the argmax predicted class throughout (deletion curves →
DDS/PES/CMI, concentration, FA, KS). Proven on a synthetic 3-class problem. No code
change needed. (DDS still normalises by max=100 on the predicted-class prob, same as
the reference on their multi-class datasets — magnitudes not comparable across
different class counts, by design, not a bug.)
Multi-channel — DESIGN DECISION (across-channel time-regions): a region is a TIME
segment hidden across ALL channels at once; the grid stays purely temporal;
relevance stays one value per time-region (n_regions,). Chosen over per-(channel,
region) cells because: (a) keeps a region a temporal segment consistent with the
autocorrelation-based region-size justification; (b) yields ONE comparable CMI per
model across datasets with different channel counts; (c) keeps the interface stable
so single-channel (C=1) is a bit-identical code path — ECG200 regression guaranteed;
(d) minimal surface, no speculative config. IMPORTANT finding: the reference's
multivariate perturber (subsequence_perturbation_multivariate.py) is UNUSED/not
imported in their active pipeline, so there is no validated multivariate path to
copy; their only hint (that unused module) does per-channel perturbation. This was
therefore a deliberate choice, not an alignment.
What changed (src/xai/): perturbation.py — SampleMean/Laplace backgrounds computed
PER CHANNEL for (C,T) (Zero unchanged); perturb_regions accepts (T,) or (C,T),
validates the last (time) axis, hides via [..., start:stop] (spans channels).
feature_ablation.py / kernel_shap.py — feature_mask broadcasts the region id across
channels (np.broadcast_to(grid.labels, signal.shape)), so a time-region groups all
channels; FA gather collapses the (equal) channel values; KS output stays
(1, n_regions). deletion_curves.py / concentration.py — unchanged (delegate shape
handling to perturbation + predict_proba). Signals are (T,) 1-D or (C,T); time axis
is last.
Regression (CRITICAL, all bit-identical): FA 0.648/0.648/0.630, KS 0.612/0.612/0.624,
oracle 0.648, random 0.060, QRS 0.034, concentration 0.242 — every number unchanged
to numerical precision (C=1 path is bit-identical). Multi-channel proof: full harness
runs end-to-end on a synthetic 3-channel, 4-class signal (region hidden across all
channels; FA/KS per-region; CMI/DDS/PES/concentration all valid & in range).

## Modularity map (revised — harness now supports arbitrary #classes and #channels)
Reuse UNTOUCHED for a new dataset (the whole harness): regions, perturbation,
deletion_curves, cmi, concentration, feature_ablation, kernel_shap. Handles any
class count and any channel count (1-D (T,) or multi-channel (C,T)); a region is a
time segment hidden across all channels.
Write NEW (dataset-specific glue — no harness surgery):
  1. A data loader (a load_<dataset> like load_ecg200) returning signals as (n, T)
     or (n, C, T) and labels.
  2. A predict_proba(batch) -> (n, n_classes) wrapper per model (models reshape to
     their expected input internally).
  3. A region-size analysis (autocorrelation, à la notebook 01b) → pass the chosen
     pct to build_region_grid. Region-size VALUES are per-dataset; the constants in
     regions.py are ECG200's.
  4. A new orchestration notebook (or parameterise 03).
  5. Optionally a bespoke baseline model; Models 2–5 on raw signal transfer.
Multi-class / multi-channel work for a new dataset: NONE — the shared harness
already handles both. The earlier "single-channel is the one real surgery" caveat is
resolved (across-channel design implemented). Only genuinely open item if a future
dataset wants PER-CHANNEL attribution (which channels matter, not just which time
regions): that would be an additive per-(channel,region) mode, deferred until a
real research question needs it.

## Decision: complexity operationalised as parameter count (1 Aug 2026)
Model complexity in this thesis = number of trainable parameters. It is the
INDEPENDENT VARIABLE / x-axis of the central result (CMI vs. complexity), a
standard, measurable, model-agnostic proxy. Every model reports its exact count.
Acknowledged limitation (for the writeup): parameter count does not capture HOW
parameters are used — architectural differences (conv vs. attention, depth vs.
width, inductive biases) can matter as much as raw count; two models with equal
parameters can differ in effective capacity. Parameter count is a defensible
first-order proxy, not a complete complexity ordering.
Ladder spacing (confirmed): ~6 (linear) → 3,858 (shallow CNN) → 43,682 (medium
CNN) → 403,522 (deep CNN) → ~1–2M (transformer, TBD). ~10x steps across ~2.5–3
orders of magnitude, chosen deliberately so the complexity axis has spread, not
clustered points. Transformer target kept conservative (~1–2M): a transformer that
overfits and fails the 80% gate is worse than a smaller top that trains properly —
five valid data points matter more than maximal spread; may revisit upward if
~1–2M trains comfortably above the bar.

## Decision: Model 2 — shallow 1D CNN + shared CNN recipe (1 Aug 2026)
Architecture (src/models/cnn.py): one parameterised CNN1D produces Models 2–4 by
changing the per-block channel-width list (CNN_VARIANTS); depth = list length,
width = values, so Models 3/4 are named configs, not bespoke builds. Per block:
Conv1d(k=7, 'same' pad, no bias) → BatchNorm1d → ReLU → MaxPool1d(2); then Global
Average Pooling → single Linear head. GAP keeps the head size independent of
depth/length and preserves a per-timestep story for later attribution. Kernel 7 ≈
ECG200's ~5–10-timestep autocorrelation length (notebook 01b). input_length /
in_channels / n_classes are read from the data (no hard-coded 96/1/2), consistent
with the generalised harness. Model 2 = variant 'shallow', channels [16, 32],
3,858 trainable parameters.
Shared training recipe (train_cnn — REUSED VERBATIM by Models 3–5 so models differ
in complexity, not confounded training): Adam, lr 1e-3, weight_decay 1e-4, batch 16,
up to 200 epochs, early stopping (patience 30) + best-validation-loss checkpoint
restore; class-weighted cross-entropy (inverse frequency) for the ~65/35 imbalance;
seed 42; same cached .npy split as Model 1. Reproducibility caveat: weight init
consumes RNG at build time, so set_seed(42) must be called BEFORE build_cnn (the
notebook does this) to make results order-independent; train_cnn re-seeds for the
training loop. Metrics consolidated in
src/evaluation/metrics.py (classification_metrics: accuracy, balanced accuracy, F1
[binary for 2-class, macro for multi-class → Sleep-EDF ready], ROC-AUC) so Models
1–5 report identically. Harness interface: torch_predict_proba(model) exposes the
same predict_proba(n,T)->(n,n_classes) contract Model 1 exposes and FeatureAblation/
KernelSHAP/deletion curves call — verified shape (3,2), rows sum to 1 — so Model 2
plugs into the harness unchanged (no XAI run yet, per scope).
Result (seed 42, deterministic, init-seeded-before-build): test accuracy 0.810,
balanced accuracy 0.785, F1 0.855, ROC-AUC 0.848 → CLEARS the >80% gate (modest
margin). Valid subject for faithfulness analysis.

## Decision: track overfitting as a reported confound (1 Aug 2026)
Overfitting is tracked for EVERY model (like concentration), reported alongside CMI.
Indicator = (train or val) accuracy − test accuracy (metrics.overfitting_gap).
Model 2: val 1.000 vs test 0.810 → val−test gap ≈ 0.19 (train−test also ≈ 0.19).
Rationale: overfitting is expected to GROW with parameter count up the ladder, so a
CMI change could partly reflect overfitting rather than complexity per se — it is a
confound for the central question and must be reported as an interpretive caveat.
Partly inherent to ECG200's small (80-sample) training set — a further reason the
second dataset (Sleep-EDF) matters. Captured from the start so it is available for
every model's row in the model × complexity × CMI table.

## Decision: Model 3 — medium 1D CNN (1 Aug 2026)
Architecture: same CNN1D module + shared train_cnn recipe as Model 2 (nothing
about training changes — only capacity), variant 'medium', conv_channels
[32, 64, 64] (3 blocks). Parameter count = 43,682 (~11x Model 2's 3,858), the
x-coordinate for this ladder point. Seed 42, init seeded BEFORE build (determinism
fix), same cached ECG200 split as Models 1–2. Reproducibility confirmed: two
independent runs (with RNG consumed in between) both give test accuracy 0.8800 —
build-order-independent.
Result: test accuracy 0.880, balanced accuracy 0.870, F1 0.906, ROC-AUC 0.941 →
CLEARS the >80% gate comfortably.
Overfitting watch (test HELD vs Model 2 — recorded explicitly): train/val pinned at
1.000 (expected with 44k params on 80 samples), but test IMPROVED from Model 2's
0.810 to 0.880 (+0.070), so the overfitting gap SHRANK to +0.12 (train−test and
val−test) from Model 2's +0.19. => the overfitting-hurts-test signal did NOT appear
at this rung; the ~11x capacity increase helped generalisation, i.e. ECG200 is not
yet capacity-saturated at ~44k params. Whether the trend turns (test dropping, gap
widening) is the thing to watch at Model 4 (~404k) and the transformer (~1–2M).
Ladder so far (params, test acc, val−test gap): Model 1 ~6 / 0.70 / ~0.20; Model 2
3,858 / 0.810 / 0.19; Model 3 43,682 / 0.880 / 0.12.

## Decision: Model 4 — deep 1D CNN (1 Aug 2026)
Architecture: same CNN1D + shared train_cnn recipe, variant 'deep',
conv_channels [64, 128, 128, 256] (4 blocks). Parameter count = 403,522 (~9x
Model 3). Architecture validity confirmed at depth 4 on the 96-length signal:
4 MaxPool blocks take 96 -> 48 -> 24 -> 12 -> 6 (stays >=1), GAP then collapses to
1; forward pass returns (2,2). NO config adjustment needed — the ladder's planned
parameter counts stand. Seed 42, init-before-build; reproducibility confirmed
(two independent runs both give test 0.8600, build-order-independent).
Result: test accuracy 0.860, balanced accuracy 0.854, F1 0.889, ROC-AUC 0.951 →
CLEARS the >80% gate. Train/val pinned at 1.000; overfitting gap +0.14 (train−test
and val−test), WIDER than Model 3's +0.12. Early stopping fired at epoch ~73 (vs
~170–200 for Models 2–3) — the larger model overfits faster.
OVERFITTING SCENARIO = SCENARIO 2 (passes gate but DROPPED below Model 3): test
fell 0.880 (M3) -> 0.860 (M4), −0.020. Generalisation PEAKED at the medium CNN and
has begun to decline at deep — the "bigger model generalises better" story from
M2->M3 REVERSES here. Modest drop (not a collapse); Model 4 remains a valid subject
(clears gate). Consequence for interpretation: Model 4 is now a WORSE-generalising
model than Model 3, so if its CMI is also lower it becomes "worse-generalising AND
harder to explain" — a different claim than the M2->M3 pattern. Flagged before any
scoring. Did NOT proceed to attributions (stopped per instruction).
Running ladder table (params / test acc / val−test gap):
  Model 1 (linear)     ~6       / 0.70  / ~0.20
  Model 2 (shallow CNN) 3,858   / 0.810 / 0.19
  Model 3 (medium CNN)  43,682  / 0.880 / 0.12   <- generalisation peak
  Model 4 (deep CNN)    403,522 / 0.860 / 0.14   <- Scenario 2, test declining
  (NB: single-seed. Multi-seed pass below OVERTURNS the M4 "declining test"
   reading — see the 1 Aug multi-seed entry.)

## Decision: multi-seed pass (5 seeds/arch) for Models 2–4 (1 Aug 2026)
Retrofit error bars: 5 seeds (0–4) per CNN architecture (Šimić et al.'s
5-runs-per-arch), full pipeline per seed (train → 3 attributions → CMI×3 PMs →
oracle → concentration → agreement → gap). No 80% gate on this pass (measure the
distribution). Retain if learned (balanced acc ≥ 0.55, not majority-collapsed);
flag+exclude collapsed. Result: 0/5 collapsed for every architecture (all learned).
Model 1 (LogisticRegression/lbfgs) is deterministic given the data (confirmed
identical across refits) — single point, no error bars, not faked.
Runner: run_multiseed_ladder.py (seeds 0–4, KernelSHAP seed 0); results saved to
results/metrics/multiseed_ladder.json; notebook 07 loads it. Runtime ~635s (~10.5m).
Aggregates (mean ± std, Zero PM):
  quantity        M2 shallow(3.9k)   M3 medium(44k)    M4 deep(404k)
  test accuracy   0.824±0.022        0.866±0.019       0.898±0.022
  oracle CMI      0.762±0.047        0.592±0.046       0.726±0.062
  concentration   0.450±0.040        0.429±0.019       0.629±0.046
  overfit gap     0.176±0.022        0.134±0.019       0.102±0.022
  agreement FA-IG 0.96±0.03          0.73±0.10         0.93±0.04
FOUR QUESTIONS — all answered YES:
  Q1 M3 dip REAL: medium oracle CMI (0.592) clearly below shallow (0.762) & deep
     (0.726); distributions barely overlap (medium max 0.67 < shallow min 0.70).
     Agreement dips too (FA-IG 0.73 vs 0.96/0.93). CONFOUND-FREE: medium test acc is
     mid-range and its concentration is flat, so the dip is NOT generalisation or
     diffuseness. The central question resolves: medium-capacity CNN is genuinely
     harder to explain faithfully on ECG200. Survives multi-seed.
  Q2 M4 concentration jump REPLICATES: deep 0.629±0.046, non-overlapping with
     shallow/medium (~0.43–0.45). Real overfitting-style fingerprint.
  Q3 Agreement non-monotonicity SURVIVES: high→dip-at-medium→high for all 3 pairs.
     NOT a monotonic decline → declining-agreement/explainability-ceiling hypothesis
     NOT supported; instead a robust medium-specific dip.
  Q4 IG convergence HOLDS: IG-FA gap narrows 0.27→0.21→0.06; IG PES 0.78→0.86→0.98.
HONEST CORRECTION (important): the single-seed M4 "Scenario 2 / worse-generalising /
overfitting confound" was SEED-42-SPECIFIC. Seed 42 gave deep test 0.86 = the WORST
of the 5 deep seeds (0.86/0.90/0.90/0.93/0.90). Multi-seed: test accuracy RISES
monotonically with capacity (0.824→0.866→0.898) and the overfitting gap SHRINKS
(0.176→0.134→0.102) — the deep CNN generalises BEST and overfits LEAST on average.
ECG200 is NOT capacity-saturated at 404k; the M4 generalisation confound does NOT
hold on average (concentration-ceiling confound on deep's CMI does remain).
NET corrected picture: generalisation improves monotonically with capacity, while
the faithfulness ceiling (oracle CMI) is NON-MONOTONIC — a real confound-free DIP at
the medium CNN, with deep recovery partly a concentration-ceiling effect. No
monotonic "harder to explain as models grow" trend across the three CNN rungs.
Central figure: results/figures/07_multiseed_ladder.png (4 panels w/ error bars).
Caveats: 5 seeds (small); one dataset/grid; Model 5 (transformer) not yet in ladder
(will be multi-seed from the start).

## Decision: Model 5 — Transformer (top of ladder), multi-seed from start (1 Aug 2026)
Architecture (src/models/transformer.py): per-timestep linear embed (Conv1d k=1) →
learned positional encoding → 6 TransformerEncoderLayers (d_model 128, 8 heads,
GELU, pre-norm, dropout 0.1) → mean-pool → linear head. Input (n,1,96), identical
contract to the CNNs (reuses train_cnn + torch_predict_proba). Parameter count =
1,202,690 (~1.2M) — top ladder step, chosen conservatively for trainability on 80
samples (not pushed to 4M).
Patching / grid choice: PER-TIMESTEP (patch_size=1) → 96 tokens, one per timestep,
so tokens map EXACTLY onto the 10-region grid (grid.labels) → grid-comparable to
the CNNs and, importantly, the later ATTENTION analysis pools per-token weights to
regions with NO boundary misalignment. Rejected larger patches: 96 doesn't split
into 10 equal patches, so fixed patches straddle region boundaries (flagged tension).
FA/KS/IG are unaffected either way (they perturb the raw input). patch_size is a
param for a patched variant later.
Recipe: NO deviation from the CNNs. A probe found the shared CNN recipe (Adam
lr 1e-3, wd 1e-4) trained the transformer BEST (lr 3e-4/1e-4 underperformed), so
reused unchanged (batch 16, 200 ep, patience 30, class-weighted CE, seed-before-
build). Only transformer-specific regularisation is the model's built-in dropout 0.1
(standard). Runner: run_transformer_seeds.py (seeds 0-4) → results/metrics/
transformer_seeds.json; notebook 08 loads it. Runtime ~349s.
Results (5 seeds, 0/5 collapsed): test acc 0.778±0.033 (spread [0.74,0.83]),
balanced 0.761, F1 0.826, ROC-AUC 0.849. Per-seed test: 0.83/0.80/0.75/0.77/0.74.
KEY HONEST FINDING — transformer UNDERPERFORMS every CNN AND is UNDERFITTING (not
overfitting): test 0.778 < shallow 0.824 < medium 0.866 < deep 0.898; and TRAIN
accuracy is only 0.890 (all CNNs reach 1.000), so it cannot even fit 80 training
samples well — it is under-fitting / struggling to optimise on tiny ECG200, not
overfitting.
Overfitting gap: val-test 0.112±0.061. Comparison to CNN trend (0.176→0.134→0.102)
is MUDDIED: the transformer's small gap is because BOTH train (0.89) and test (0.78)
are low (weak learning), NOT good generalisation — it does NOT continue the
decreasing-gap trend for the same reason.
Stability: no collapse/divergence, but MORE VARIABLE than CNNs — test std 0.033 (vs
~0.02), val acc 0.80–1.00 across seeds, best-epoch 71–126. Usable but less stable.
Implication: transformer is a POOR FIT for 96-sample univariate ECG200 (wrong
inductive bias); its eventual XAI/CMI results carry LESS weight (explaining a weakly-
fit model is less meaningful) — flag as the weakest-fit rung. Central figures:
results/figures/08_model5_training.png, 08_model5_ladder.png.
Full ladder test acc (params): M2 3.9k/0.824, M3 44k/0.866, M4 404k/0.898,
M5 1.2M/0.778 — accuracy rises across the CNNs then DROPS at the transformer.