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


================================================================================
PHASE 2 — SECOND DATASET: SLEEP-EDF (Sleep-Cassette, EEG sleep staging)
================================================================================
Purpose: a study of XAI faithfulness versus model complexity on the Sleep-EDF
sleep-staging task (single-channel EEG, 5-class AASM staging). The faithfulness of the
attribution methods is measured across a ladder of models of increasing complexity, on
this dataset on its own terms. This section records the DATA decisions only; models /
region sizes are chosen later. (ECG200 appears elsewhere in the repository as earlier
related work; no Sleep-EDF decision below depends on it.)

Channel & format
Single-channel EEG Fpz-Cz sampled at 100 Hz, cut into non-overlapping 30-s epochs =
3000 samples/epoch. Single-channel Fpz-Cz @100 Hz / 30-s epochs is the standard
Sleep-EDF setup (matches Khalili & Mohammadzadeh Asl 2021; Supratak et al.
DeepSleepNet 2017). The other PSG channels (Pz-Oz EEG, EOG @100 Hz; and the 1-Hz
EMG/resp/temp/event channels) are NOT used, keeping the input a single time series.

Label mapping (R&K → AASM 5-class)
Hypnogram stages are Rechtschaffen & Kales; mapped to the modern AASM 5-class scheme:
W→0, stage 1→N1(1), stage 2→N2(2), stages 3 AND 4→N3(3) [R&K splits deep sleep into
3 and 4; AASM merges them], stage R→REM(4). "Movement time" and "?"/Unknown epochs
are DROPPED (not a class). This is the standard sleep-staging target.

Wake-trimming (long-Wake removal)
Raw recordings are dominated by lights-on Wake before/after the sleep period. We keep
only 30 min (= 60 epochs, at 2 epochs/min) of Wake at EACH end of the sleep period
(first→last non-Wake epoch), discarding the rest. This is the DeepSleepNet convention
(w_edge_mins = 30). Effect: Wake share 68.8% → 33.7%; dataset 414,961 → 195,479 epochs
(153 recordings). Trimming is done PER RECORDING, before the split, so it cannot move
epochs across the train/test boundary.

Class balance (post-trim) & the deliberate NO-REBALANCING decision
Overall post-trim balance: W 33.7% / N1 11.0% / N2 35.4% / N3 6.7% / REM 13.2%.
N1 and N3 are minority classes; N3 is rarest at 6.7%. We deliberately DO NOT rebalance
or reweight (no oversampling, no class weights). Reason: rebalancing would substitute an
ARTIFICIAL class prior for the true one, recalibrating the model's decision function to a
distorted world — and since faithfulness probes the model's REAL decision function, that
recalibration would confound the very thing we measure. Class imbalance is therefore
handled by MEASUREMENT — per-class F1/recall, balanced accuracy, and the concentration
analysis — NOT by data manipulation. The same principle governs the training subsample
(see SUBSAMPLING below), which likewise preserves the natural class balance exactly and
for the same reason. FLAG carried forward: per-class faithfulness for the scarce classes
(N1, and especially N3) must be reported WITH caveats about sample scarcity.

Preprocessing (per-epoch z-norm; NO bandpass filter)
Each 30-s epoch is z-normalised by ITS OWN mean/std (per-epoch; guard std==0). We
apply NO bandpass filter. Rationale (faithfulness-motivated): this is a faithfulness
study, and the low-complexity baseline in particular serves as a directly-inspectable
interpretability anchor for validating the attribution/harness machinery. Keeping the
input unfiltered preserves the tightest correspondence between the raw signal, the
model's readable parameters, and the attribution — no preprocessing transform is
interposed between signal and model through which the attribution would then have to be
read. More generally the study prioritises an unmediated signal-to-attribution
correspondence over the accuracy-oriented preprocessing standard in the sleep-staging
literature (which bandpass-filters for accuracy — a different goal). This is therefore
a DELIBERATE DEPARTURE from the sleep-staging convention (which filters), made on
faithfulness grounds, not an oversight. Per-epoch (not global/train-fit) normalisation
also means NO dataset-level statistic crosses the split — leakage-free by construction
(see the NO-LEAKAGE NOTE in src/data/sleep_edf.py). CONDITIONAL ESCAPE HATCH: a bandpass
filter may be added later ONLY if the models demonstrably cannot learn on the raw
signal, documented as a data-driven necessity; any global normalisation would then be
fit on TRAIN only.

Split (leakage-free, subject-level)
Subject-level ~80/20 split, seed 42: 62 train / 16 test subjects → 121 train / 32 test
recordings (each subject has ~2 nights, always kept on the SAME side). Provably disjoint
subjects; both of a subject's nights land together. Per-split epochs: train 155,334 /
test 40,145 (total 195,479). Subject-level (not epoch-level) splitting prevents epochs
from the same night/subject leaking across train and test.

Loader hardening (Stage 3) & regression gate
src/data/sleep_edf.py now hardens the Stage 1–2 validated notebooks into a reusable
cached loader: build_processed_data() builds + disk-caches once
to data/sleep_edf/processed/sleep_edf_{train,test}_{X,y}.npy (~2.35 GB, gitignored),
load_sleep_edf(split) loads the cache (force_rebuild flag; cache-hit load ~0.4 s). The
per-recording loop CATCHES AND REPORTS (never silently drops) missing-channel /
missing-hypnogram / all-Wake / read-error recordings. Regression check vs the Stage 2
notebook: ALL quantities matched EXACTLY — 153 clean recordings, 62/16 subjects,
121/32 recordings, 155,334/40,145 epochs, overall per-class {W 65951, N1 21522,
N2 69132, N3 13039, REM 25835}, subjects disjoint, nights kept together. Build time
141 s over all 153 recordings; 0 issues.
NOTE — src/data/__init__.py was intentionally NOT edited: it exports ECG200's
build_processed_data, and Sleep-EDF's function shares that name (collision), so per the
"touch no ECG200 file" constraint the loader is imported by module path instead:
    from src.data.sleep_edf import load_sleep_edf, build_processed_data


Region-size
Notebook: notebooks/sleep_edf/03_region_size_analysis.ipynb. Config: src/config/sleep_edf.py.

DECISION. Region size = 60 samples (600 ms, 2% of a 3000-sample epoch → 50 regions,
which divide 3000 exactly). A single grid is used; there is no fine grid (see below).

Primary rationale — the physiological scale of sleep-staging events. Under the AASM
standard, sleep stages are identified by characteristic waveform events: sleep spindles
(~0.5–1.5 s), K-complexes (~0.5 s) and slow waves (~1 s) — i.e. ≈50–150 samples at 100 Hz.
A 60-sample (600 ms) region corresponds to approximately one such event, so the
perturbation grid is aligned with the physiologically meaningful units of sleep staging:
each region is about the size of the structure the label actually depends on, rather than
an arbitrary fraction of the epoch. This is a domain-motivated choice, fixed BEFORE any
faithfulness result was computed — not a tuned parameter.

Corroborating context (supporting, NOT validation). Khalili & Mohammadzadeh Asl (2021),
working on the same Sleep-EDF task, use multi-scale CNN kernels spanning ~25–200 samples on
the reasoning that stage-distinguishing structure lives at that scale; a 60-sample region
sits inside this range. This is a classification pipeline with no region grid, so it is
supporting evidence about the temporal scale of stage-relevant EEG structure — explicitly
NOT a validation of the region size itself (different construct).

Confirmatory signal-scale check (does NOT derive the choice). As an independent sanity
check, a per-epoch autocorrelation analysis (mean-subtract each 30-s epoch, linear ACF,
normalise to 1 at lag 0, average over a stage-balanced sample of 800 epochs × 5 classes)
measures Sleep-EDF's 1/e correlation length at ≈15 samples (150 ms; first zero crossing
≈45 samples), stable across stages (13–19 samples; N3 coarsest at 19, ~0.76 Hz — the
textbook slow-wave frequency). This ~15-sample correlated grain sits WELL BELOW the
60-sample event scale, confirming that a 60-sample region comfortably contains the signal's
correlated structure (it never slices inside one coherent grain). The correlation length is
thus a lower bound the region clears — it corroborates, but does NOT derive, the 60-sample
choice, which stands on the physiological event scale above.

No fine grid. An earlier draft carried a second, finer 30-sample grid. It has been DROPPED:
its only justification was a fixed primary:fine ratio, with no independent, physiologically-
or ML-grounded reason for a 30-sample region specifically, so it did not stand on its own.
The study uses the single 60-sample grid.

Wiring. regions.py::build_region_grid(length, region_size_pct) takes the region size as an
explicit PERCENTAGE (n_regions = round(100/pct)); Sleep-EDF passes REGION_SIZE_PRIMARY_PCT
= 2.0 per call. src/config/sleep_edf.py holds INPUT_LENGTH=3000, IN_CHANNELS=1, N_CLASSES=5,
the region size (60 samples / 2% / 50 regions), and an import-time assert proving the exact
division. NOT yet applied to any model/harness run — config + log only.


SUBSAMPLING (training set → fixed 20,000-epoch stratified subsample)
Notebook: sleep_edf/notebooks/02_dataset_construction.ipynb (Step D). Wired into
sleep_edf/loader.py as the DEFAULT for training loads.

Decision. Train on a fixed 20,000-epoch stratified random subsample of the training set.
The second marker required ≥10,000 training epochs; 20K sits comfortably above that floor
while keeping the minority classes viable. This is a COMPUTE-BUDGET decision — to make
five models × five seeds feasible — NOT a scientific one: it changes how much data we
train on, not what we measure.

Method (keep all subjects, thin epochs-per-subject). Stratified epoch-level sampling that
keeps ALL 62 training subjects, each contributing a proportional random slice of their
night — rather than dropping whole subjects. Rationale: the task classifies single 30-s
epochs INDEPENDENTLY, so whole contiguous nights give no modelling benefit; cross-subject
diversity (many individuals' EEG) matters more than per-subject depth for learning
generalisable staging, and keeping all 62 subjects preserves that diversity for the same
epoch budget.

No rebalancing (natural class balance preserved exactly). Subsample balance
W 34.1% / N1 10.9% / N2 35.1% / N3 6.6% / REM 13.3% — identical to the full training set.
Same reason as the NO-REBALANCING decision above: rebalancing would replace the true class
prior with an artificial one, recalibrating the decision function to a distorted world and
confounding the faithfulness measurement (which probes the model's REAL decision function).
Imbalance is handled by measurement (per-class F1/recall, balanced accuracy) and the
concentration analysis, not by data manipulation.

Identical across the ladder. Defined by a deterministic class-stratified function
(subsample_indices, seed 42) that is the SINGLE source of truth in sleep_edf/loader.py,
wired in as the DEFAULT for training loads with an explicit, logged opt-out for the full
set (load_sleep_edf("train", subsample=None) → 155,334). Every model on the complexity
ladder therefore trains on the SAME 20K epochs, so model complexity remains the only
variable across the ladder.

Minority classes survive. N1 = 2,185 and N3 = 1,329 (rarest) epochs in the subsample —
retained as a genuine property of sleep and reported with a scarcity caveat, NOT rebalanced
away.

Test set kept whole. The test split is NEVER subsampled: 40,145 epochs, 16 subjects —
evaluation stays on the full, natural-balance test set.

Effect on claims (acknowledged limitation). Subsampling may lower a model's ABSOLUTE
accuracy relative to full-set training. But the study concerns the COMPLEXITY→FAITHFULNESS
relationship — the SHAPE of that relationship across the ladder — not any single model's
absolute accuracy; identical training data across the ladder keeps that comparison clean.
Acknowledged as a limitation.


BASELINE REDEFINITION (bottom rung: raw logistic → band-power logistic)
Notebooks: sleep_edf/notebooks/04_model1_linear_baseline.ipynb (raw, superseded) →
04b_model1_bandpower_logistic.ipynb (new bottom rung). Features: sleep_edf/bandpower.py.

Decision. The bottom rung of the complexity ladder is BAND-POWER + logistic regression,
replacing raw logistic regression on the raw 3000-sample epoch.

Why raw was rejected. Raw multinomial logistic regression on the raw EEG epoch is
NEAR-CHANCE: balanced accuracy ~0.22 (vs 0.20 chance), N3 recall ~0.03. A linear model on
raw samples cannot represent the frequency/shape structure that distinguishes sleep stages,
so it neither learns the task NOR provides readable ground truth — it fails at both jobs a
bottom rung could do.

Why band-power (justified on Sleep-EDF / sleep-physiology terms). The rung's PURPOSE is not
to be a competitive classifier but to be a model whose reasoning is DIRECTLY READABLE from
physiology, so the XAI/CMI machinery can be validated against known ground truth ("do my
attribution methods recover the band I know the model uses?") BEFORE it is trusted on the
opaque CNNs — the supervisor's start-simple-and-interpretable-then-go-complex framing. Sleep
stages have well-known frequency-band signatures (above all N3 ↔ delta), so reducing each
30-s epoch to one power value per standard band (delta 0.5–4, theta 4–8, alpha 8–12,
sigma/spindle 12–16, beta 16–30 Hz; Welch PSD integrated per band, log-power) makes
logistic regression BOTH a legitimate simple baseline AND a readable ground-truth anchor:
each coefficient maps one-to-one onto a named band. Small, named feature set by design (5
features, not an opaque vector) — the legibility is the whole point. Parameter count 30
(5 classes × 5 bands + 5 intercepts) — the ladder's low point.

Consequence noted (input-story of the ladder). This rung's INPUT is band-power features,
whereas the CNN and transformer rungs see the RAW signal (the raw-signal, no-filter decision
still governs them). So this baseline's role is XAI-MECHANISM VALIDATION (attribution over
named bands vs known physiology), NOT a directly-comparable point on the raw-signal
faithfulness curve. The complexity→faithfulness comparison proper runs across the raw-input
CNNs and transformer; the band-power baseline sits alongside as the interpretable anchor.

Ground-truth expectations (recorded in ADVANCE, to check the XAI against later).
  PRIMARY  — N3 (deep sleep) ↔ delta (0.5–4 Hz): high-amplitude slow waves; the strongest,
             least ambiguous signature and the CLEAN primary validation target — a
             trustworthy attribution on N3 must recover delta.
  secondary — N2 ↔ sigma/spindle (12–16 Hz) + theta; Wake ↔ alpha (8–12) + beta (16–30);
             REM ↔ low-amplitude mixed / theta (weak); N1 ↔ theta (transitional, weakest).
Feature-level preview (no training) already shows the signature: per-class mean log-band-
power has N3 delta highest (0.48 vs ~0.31 elsewhere) with higher bands suppressed. After
training, the learned class×band coefficients are read back and compared to this table
directly — the readable-ground-truth property in action. Functional check expected: balanced
accuracy meaningfully above 0.20 chance and N3 recall far above the raw model's ~0.03.

## Decision: CNN first-layer kernel size = 15 for Sleep-EDF (12 Aug 2026)

Sleep-EDF's single-channel EEG has a measured autocorrelation length of ~15 samples (the
1/e correlation length; confirmed in the region-size analysis). A first-layer convolution of
15 samples = 150 ms at 100 Hz therefore spans roughly one coherence unit of the signal: each
filter sees one correlated segment of the waveform rather than a fraction of one or several
run together. The kernel is set from the dataset's own coherence scale, so the model's
smallest feature detector is matched to the smallest scale at which the EEG carries
structure. Value lives in `sleep_edf/config.py` (`CNN_KERNEL_SIZE = 15`) and is passed
explicitly to `build_cnn(..., kernel_size=15)`; the harness default (7) is unchanged and no
Sleep-EDF value is hardcoded in `harness/`. Kernel is odd, so `padding = kernel_size // 2 = 7`
preserves sequence length through every conv (only the per-block MaxPool(2) changes length).

## Decision: Transformer patch size = 60 for Sleep-EDF (12 Aug 2026)

Each token embeds a 60-sample (600 ms) patch, giving 3000 / 60 = 50 tokens per epoch. Two
reasons, both grounded in Sleep-EDF:
  (1) Grid alignment. The faithfulness harness attributes onto the 60-sample RegionGrid (50
      regions). A 60-sample patch places exactly one token per region, so attention weights
      map onto the CMI region grid with NO aggregation step — token-level and region-level
      analyses are the same partition, no boundary misalignment.
  (2) Event scale. 600 ms is the AASM stage-defining event scale (spindles/K-complexes/slow
      waves ≈ 50–150 samples) that also set the region size, so each token corresponds to
      approximately one physiologically meaningful unit.
A practical consequence, not the justification: 50 tokens keeps full O(N²) self-attention
tractable, whereas per-timestep tokens would mean attention over 3000 positions per epoch.
3000 is divisible by 60 exactly, so there is no padding or truncation. Value lives in
`sleep_edf/config.py` (`TRANSFORMER_PATCH_SIZE = 60`) and is passed to `build_transformer(...,
patch_size=60)`; the harness default (1) is unchanged. Instantiated ladder point:
1,204,741 params, 50 tokens, positional embedding (1, 50, 128) = 6,400 params.

## Decision: receptive-field progression as the ladder's mechanistic reading (12 Aug 2026)

With kernel 15, the CNN rungs have receptive fields of 46 / 106 / 226 samples =
460 / 1060 / 2260 ms (shallow / medium / deep). Read against the AASM stage-defining event
scale (~50–150 samples, 500–1500 ms): shallow (460 ms) resolves roughly a single minimal
event; medium (1060 ms) spans a full event; deep (2260 ms) spans multiple events plus
surrounding context. This gives the ladder a mechanistic reading — how much temporal context
each rung can integrate — that sits alongside parameter count and helps interpret any
CMI/faithfulness differences between rungs.

## Decision: depth and width are confounded across rungs; parameter count is the declared axis (12 Aug 2026)

The CNN rungs vary depth and channel width together (shallow [16,32] → medium [32,64,64] →
deep [64,128,128,256]), so those two factors are deliberately confounded. Parameter count
(~8.2K / ~93K / ~864K) is the DECLARED complexity axis of the thesis; receptive field
(460/1060/2260 ms) is reported alongside as the mechanistic interpretation. Consequence for
analysis: a CMI difference between rungs cannot be attributed to depth or to capacity
separately — it is a difference along the joint complexity axis, and claims must be phrased
that way rather than as "deeper" or "higher-capacity" in isolation.

## Decision: GAP head — CNN attributions read as "present in region", not "when it mattered" (12 Aug 2026)

The CNN head is Global Average Pooling over time followed by a single linear layer, so the
model's decision depends on WHETHER a channel's pattern is present anywhere in the epoch, not
on WHERE it occurs — GAP discards temporal position. Consequence for interpretation: a
region-level attribution on these CNNs indicates that the pattern the model relies on was
present in that region, not that its temporal position within the epoch carried the decision.
Attribution and CMI results on the CNN rungs must be read in those terms (presence, not
timing).

## Decision: expose per-epoch subject IDs in the loader (12 Aug 2026)

`sleep_edf/loader.py` now caches and can return a per-epoch subject-id array
(`load_sleep_edf(..., return_subjects=True)` → `(X, y, subj)`), needed to build a
subject-level validation split (next entry). The change is purely additive: the default
return signature stays `(X, y)`, no existing default or behaviour changed. Regression gate
(force-rebuild from the raw EDFs, compare to the pre-change cache): train/test `X` and `y`
are **bit-identical** (SHA-256 match and `array_equal`), the subject-level train/test split
is unchanged (62 train / 16 test subjects, zero subject overlap = leakage-free preserved),
and the fixed 20,000-epoch subsample is identical (same selection-index hash, same class
counts W=6822/N1=2185/N2=7011/N3=1329/REM=2653). Subject id = recording-name chars [3:5]
(the same field the train/test split groups on, so a subject's two nights share a code).

## Decision: subject-level early-stopping validation split, fixed across the ladder (12 Aug 2026)

**A validation split now exists** — Model 1 (band-power logistic regression) had none, as it
has no epochs and needs no early stopping. Every neural rung (Models 2–5) trains with the
shared `train_cnn` recipe, which early-stops on a validation set, so one is now required.

**It is subject-level, not a random epoch split.** Sleep-EDF has subject structure: epochs
from one subject share electrode placement, skull geometry and noise floor. A random
class-stratified split of *epochs* would place the same subject on both sides, letting the
model partially memorise subject identity — the early-stopping signal would be optimistic and
stopping would fire late. Crucially that effect grows with model capacity, so it would vary
along the very axis (parameter count) this thesis measures, confounding the comparison. A
random-epoch split is harmless only *without* subject structure (e.g. ECG200); that
assumption does not transfer, so ECG200's convention was explicitly not inherited. Holding
out whole subjects makes the early-stopping signal an honest generalisation-to-unseen-subject
signal, consistent with the leakage-free subject-level train/test split. (The reported test
metrics are unaffected regardless — the val set only governs when training stops.)

**Held out: 6 whole subjects — [2, 16, 37, 48, 60, 70]** (≈10% of the 62 training subjects;
both nights of a subject always on the same side). Fixed and frozen to
`sleep_edf/data/processed/sleep_edf_val_subjects.json`; every model on the ladder LOADS this
exact split via `sleep_edf/validation.py::train_val_split()` rather than regenerating it — a
per-model val split would move the early-stopping target between rungs and inject variance
along the measured axis, the same reason the 20K subsample is fixed. Resulting counts on the
20K subsample: **train 17,742 / val 2,258**; per class (train | val) —
W 5931|891, N1 1954|231, N2 6307|704, N3 1165|164, REM 2385|268. The test set is untouched
(the separate leakage-free subject-level holdout of 16 different subjects).

**Split seed = 8, not the project-wide 42.** Everything else in Sleep-EDF uses seed 42
(train/test split, 20K subsample); the validation split alone uses seed 8. Seed 42 was
REJECTED for this split because it held out subjects [1, 7, 61, 64, 73, 74], of which **three
had zero N3 epochs** (N3 = 102 total, 4.2% of val vs 6.6% natural) — the early-stopping signal
would have been effectively blind to N3, the rarest and hardest stage. The rejection
criterion — *no empty-N3 validation subject and an adequate absolute N3 count* — was fixed
BEFORE any candidate seed was examined, so this is selection for measurability of a known-hard
class, not selection on results. Candidates compared (all ~10%, 6 subjects unless noted;
N3/N1 = val counts):

  - seed 42 (default): N3 102 (4.2%, **3/6 subjects zero N3**), N1 286 — REJECTED
  - seed 8  (CHOSEN) : N3 164 (7.3%, 0 empty), N1 231
  - seed 3           : N3 155 (6.2%, 0 empty), N1 200
  - seed 23, k=7     : N3 165 (6.5%, 0 empty), N1 226 — closest to natural, but enlarges holdout to 11.3%

Seed 8 was chosen: it keeps the holdout at 6 subjects (9.7%, closest to the specified ~10%)
and has the healthiest minority counts among the k=6 options. A validation set for early
stopping needs signal stability on the minority classes (absolute counts) more than
proportional fidelity to population rates — the untouched test set carries the honest class
distribution. Seed 42 remains in use everywhere else; only the validation split uses seed 8.

## Decision: configurable early-stopping metric in harness train_cnn (12 Aug 2026)

`harness/models/cnn.py::train_cnn` gains three additive keyword arguments — `monitor`
({"val_loss", "val_balanced_accuracy"}), `mode` ({"min", "max"}) and `min_delta` — plus
restore-best-weights by the *monitored* metric and two new return keys, `stopped_epoch` and
`stop_reason` ({"patience", "max_epochs"}), so a caller can report where and why training
stopped. Both validation quantities were already computed every epoch, so monitoring balanced
accuracy needed no new machinery. The change is purely additive: the defaults
(`monitor="val_loss"`, `mode="min"`, `min_delta=1e-6`) reproduce the previous
`val_loss < best_val_loss - 1e-6` logic exactly, so ECG200 (Models 2–5) is untouched. The
optimizer, scheduler, batching, class-weighting and every existing default are unchanged.

Sanity check (light, not a full regression gate): re-running the existing ECG200 path with no
new arguments reproduces the committed `multiseed_ladder.json` metrics **exactly** for
shallow and deep across seeds 0–2 (balanced accuracy, F1, ROC-AUC to <1e-9; test accuracy
exact) — identical restored weights, hence identical stopping behaviour. Invalid `monitor`/
`mode` raise.

## Decision: ladder-wide early-stopping protocol — monitor validation BALANCED ACCURACY (12 Aug 2026)

Every neural rung (Models 2–5) early-stops on **validation balanced accuracy, not validation
loss**. Reason: the validation loss is dominated by the majority stages — W and N2 are ~69%
of the data — so a model can lower its loss while N1/N3 recall degrades, and loss-monitored
early stopping would select exactly the checkpoint that is worst on the minority stages this
thesis most needs to keep measurable. Balanced accuracy (mean per-class recall) weights all
five stages equally, so the restored checkpoint is the one that best serves the minority
classes. (Model 1, the band-power logistic baseline, had no early stopping — it has no
epochs — so this protocol begins at Model 2.)

Protocol settings, FIXED across Models 2–5 (passed from each notebook, not defaulted in the
harness, so the harness stays dataset-agnostic):

    monitor = "val_balanced_accuracy"   mode = "max"   patience = 10
    min_delta = 0.002                   max_epochs = 100

  - `min_delta = 0.002` (0.2 pp of balanced accuracy). The fixed val set has N3 = 164, so a
    single N3 recall flip moves balanced accuracy by ~1/164/5 ≈ 0.12 pp; a 0.2 pp threshold
    clears single-epoch minority-class noise without stalling on genuine gains.
  - `max_epochs = 100` is a generous ceiling, not the expected stop: with patience 10 on
    balanced accuracy over 17,742 training epochs, models should halt by patience well before
    100. Per-epoch cost on local CPU is unknown until seed 0 is timed, so this may be revised
    after seeing that timing. Each seed records `stopped_epoch` and `stop_reason`; if runs
    routinely hit the ceiling they are being truncated while still improving, and that will be
    visible per seed rather than hidden.

The stopping protocol is held identical across the ladder deliberately: models stopping at
different epoch counts is the models differing, not the protocol differing, so early stopping
is not a confound in the complexity→faithfulness comparison.

## Decision: Model 2 — shallow 1D CNN, second ladder rung (12 Aug 2026)

Training notebook `sleep_edf/notebooks/model2/06_model2_shallow_cnn.ipynb` (training only; the
XAI/CMI notebook is a later task, after all CNNs are trained). Mirrors Model 1's training
notebook section-for-section, minus the coefficient/known-answer sanity check (no CNN
analogue — faithfulness questions belong in the XAI notebook).

  - Variant: `build_cnn("shallow")` from `harness/models/cnn.py` — `CNN_VARIANTS["shallow"] =
    [16, 32]`. No architecture code is copied into sleep_edf/.
  - Kernel size: **15**, sourced from `sleep_edf/config.py::CNN_KERNEL_SIZE` (150 ms at 100 Hz,
    matched to Sleep-EDF's ~15-sample autocorrelation), passed to build_cnn — NOT the harness
    default of 7. The notebook hard-asserts the config kernel reached conv1 and refuses to
    train otherwise.
  - Parameters: **8,181 total = 8,016 conv trunk + 165 head** (5-class GAP head).
  - Receptive field: **46 samples = 460 ms** at 100 Hz (the minimal-event end of the ladder's
    mechanistic reading; see the receptive-field-progression entry).
  - Seeds: **5** (fixed up front, no single-seed preview), via the shared multi-seed driver
    `sleep_edf.training.run_all_seeds`.
  - Training data: **17,742 epochs** = the fixed 20,000-epoch subsample minus the 6 held-out
    validation subjects; validation = the fixed 2,258-epoch subject-level split; test = the
    full 40,145. No resampling or rebalancing.
  - Stopping rule: the ladder-wide protocol (monitor val balanced accuracy, mode max,
    patience 10, min_delta 0.002, max_epochs 100) — see that entry; not duplicated here. The
    notebook passes it to train_cnn and surfaces per-seed `stopped_epoch` / `stop_reason`.

Decisions made that were not specified:
  - **Majority-class floor is N2, not W.** In the test set N2 (14,679) outnumbers W (12,967),
    so the majority-class baseline is "always predict N2" = 0.3656, not the ~34% W figure
    assumed in the task. The notebook computes the floor dynamically and labels the class, so
    it reports the true reference rather than a hard-coded one.
  - **Per-seed checkpoints saved** to `sleep_edf/results/checkpoints/model2_shallow_cnn_seed{
    seed}.pt`. A CNN cannot be reconstructed from summary numbers the way the logistic model
    could, so the later XAI notebook loads these instead of retraining.
  - **Batched test inference** (`_predict_in_batches`, 512/chunk) in the notebook: the harness
    `torch_predict_proba` runs the whole array in one forward (fine for ECG200's ~100 samples)
    but would build a huge intermediate activation on the 40,145-epoch test set. Chunking is a
    notebook-side wrapper — it does not touch the harness and is not a training loop.
  - **No inner epoch progress bar.** `train_cnn` exposes no per-epoch callback, so the driver's
    optional inner bar isn't driven (outer per-seed bar + seed-0 time estimate still show). Not
    adding a hook — no harness changes in this task.

## Decision: MPS (Apple-Silicon GPU) as the ladder-wide training device for Models 2–5 (12 Aug 2026)

Neural training (Models 2–5) runs on **MPS** where available, falling back to CPU on non-Apple
machines. On this hardware MPS is ~100× faster than CPU for the shallow CNN, turning an
infeasible local-CPU run into a practical one (the full 5-seed Model 2 run drops from ~14 h to
a few minutes).

No harness change was needed: `train_cnn` and `torch_predict_proba` already accept a `device`
argument (default `"cpu"`, so ECG200 is untouched), and `train_cnn` already places model, data
and loss on-device and returns predictions to CPU before the sklearn/`classification_metrics`
step. The device is passed from each model notebook (`DEVICE = "mps" if
torch.backends.mps.is_available() else "cpu"`), not defaulted in the harness, keeping the
harness dataset- and hardware-agnostic. `run_all_seeds` is device-agnostic (the per-seed
function chooses the device).

Speedup, and how it was verified. Per-batch timing with `.item()` forcing completion each step
(so it measures real compute, not async queueing): **~235 ms/batch on CPU vs ~2 ms/batch on
MPS (~117×)**; both backends reach comparable loss (1.43 vs 1.42), i.e. MPS is genuinely
training, not no-op'ing. A 2,048-sample / 4-epoch A/B on the actual Sleep-EDF data reproduced
this (subset overhead makes the wall-clock ratio ~32× there, per-batch steady-state ~100×).

Determinism. `set_seed` (`torch.manual_seed`) covers MPS, and the model has no dropout, so the
only RNG — weight initialisation and DataLoader shuffling — runs CPU-side and is seeded.
Verified empirically: **two MPS runs of the same seed produced bit-identical loss curves**
(max|Δ| = 0.0). So MPS is reproducible for a given seed, and the 5-seed spread reflects genuine
seed variance, not backend noise — error bars are trustworthy.

Numerics across backends. float32 differs slightly between CPU and MPS: a same-seed CPU-vs-MPS
A/B tracked the same loss trajectory with max|Δ train_loss| ≈ 0.0002 (float32 rounding), so
they are clearly the same training but **not bit-comparable**. Consequence: results are
compared within-backend; a number produced on MPS is not expected to reproduce a CPU number to
full precision.

Model 1 note. The band-power logistic baseline (Model 1) was trained on CPU (sklearn, 30
parameters) — unaffected in any way that matters, but recorded here so the backend difference
across the ladder is on the record: Model 1 = CPU/sklearn, Models 2–5 = MPS/torch.

## Decision: Model 3 — medium 1D CNN, third ladder rung (12 Aug 2026)

Training notebook `sleep_edf/notebooks/model3/07_model3_medium_cnn.ipynb` (training only; XAI/CMI
is a later task). Mirrors the Model 2 notebook section-for-section.

  - Variant: `build_cnn("medium")` — `CNN_VARIANTS["medium"] = [32, 64, 64]` (one block deeper
    and wider than Model 2's `[16, 32]`). Built from the shared harness; no architecture code in
    sleep_edf/.
  - Parameters: **93,285 total = 92,960 conv trunk + 325 head** (vs Model 2's 8,181).
  - Receptive field: **106 samples = 1060 ms** at 100 Hz (vs Model 2's 46 / 460 ms); downsampling
    1/8, sequence length at head 375. This is the first rung whose RF spans a full AASM
    stage-defining event (~500–1500 ms) rather than sitting at its lower edge — the rung where a
    faithfulness dip is hypothesised.

Everything else is unchanged from Model 2 by design (controlled complexity ladder — only the
variant differs): same fixed subject-level split (17,742 train / 2,258 val) and untouched
40,145 test set; same `CNN_KERNEL_SIZE = 15`; same ladder-wide stopping rule
(`val_balanced_accuracy`, max, patience 10, min_delta 0.002, max_epochs 100); same 5 seeds
[0–4]; same MPS device, batch size, optimizer; same `run_all_seeds` helper; same learning-
verification presentation. See the ladder-wide entries (kernel size, val split, stopping rule,
MPS device) rather than a re-statement here.

Addition over Model 2: a ladder-comparison cell prints Model 2 vs Model 3 side by side (params,
RF, balanced accuracy, macro-F1, N1/N3 recall). Model 2's numbers are read from its saved
aggregate JSON (its RF recomputed from the recorded variant/kernel), not hardcoded. No new
judgement calls beyond those already logged for Model 2 (majority floor = N2, per-seed
checkpoints, batched test inference, no inner epoch bar), which carry over unchanged.

## Decision: Model 4 — deep 1D CNN, fourth ladder rung (12 Aug 2026)

Training notebook `sleep_edf/notebooks/model4/08_model4_deep_cnn.ipynb` (training only; XAI/CMI
is a later task). Mirrors the Model 3 notebook section-for-section. Largest, deepest CNN on the
ladder.

  - Variant: `build_cnn("deep")` — `CNN_VARIANTS["deep"] = [64, 128, 128, 256]` (four blocks, vs
    Model 3's three). Built from the shared harness; no architecture code in sleep_edf/.
  - Parameters: **863,557 total = 862,272 conv trunk + 1,285 head** (Model 2: 8,181; Model 3:
    93,285) — roughly a 100× span in complexity from the bottom CNN rung to the top.
  - Receptive field: **226 samples = 2260 ms** at 100 Hz (Model 2: 460 ms; Model 3: 1060 ms);
    downsampling 1/16, sequence length at head 187. Model 4's RF sits **above** the AASM
    stage-defining event scale (~500–1500 ms): across the CNN rungs the RF sweeps from the lower
    edge (Model 2, minimal event) through inside it (Model 3, one full event) to spanning
    multiple events with surrounding context (Model 4).

Everything else is unchanged from Model 3 by design (controlled complexity ladder — only the
variant differs): same fixed subject-level split (17,742 train / 2,258 val) and untouched
40,145 test set; same `CNN_KERNEL_SIZE = 15`; same ladder-wide stopping rule
(`val_balanced_accuracy`, max, patience 10, min_delta 0.002, max_epochs 100); same 5 seeds
[0–4]; same MPS device, batch size, optimizer; same `run_all_seeds` helper; same learning-
verification presentation. See the ladder-wide entries rather than a re-statement here.

The §5 ladder-comparison cell is extended to show all three CNN rungs (Models 2, 3, 4) side by
side with per-step deltas; Models 2 and 3 are read from their saved aggregate JSONs (RF
recomputed from the recorded variant/kernel), not hardcoded. No new judgement calls beyond
those already logged for Model 2, which carry over unchanged.

## Decision: Model 5 — Transformer, top ladder rung (14 Aug 2026)

Training notebook `sleep_edf/notebooks/model5/09_model5_transformer.ipynb` (training only; XAI/CMI
and the attention-weight analysis are a later task). Mirrors the Model 4 notebook
section-for-section. Only the architecture changes from Model 4; it is the only
non-convolutional rung.

  - Architecture: `build_transformer` from `harness/models/transformer.py` (Transformer encoder,
    6 layers, d_model 128, 8 heads, GELU, pre-norm). Built from the shared harness; no
    architecture code in sleep_edf/.
  - Patch size: **60**, sourced from `sleep_edf/config.py::TRANSFORMER_PATCH_SIZE` — one token per
    60-sample RegionGrid region, **not** the harness default of 1 (which would give 3000 tokens
    and O(3000²) attention). The notebook hard-asserts the config patch reached the embedding
    (Conv1d kernel *and* stride) and refuses to train otherwise.
  - Parameters: **1,204,741**. Tokens: **50** (3000 / 60, exact — no padding or truncation).
    Positional embedding **50 × d_model** (1×50×128 = 6,400 params), not 3000 × d_model.
  - Receptive field: **global** (self-attention — every token attends to all 50), so this rung
    sits *outside* the CNN receptive-field-vs-AASM-event framing rather than extending it.

Ladder framing recorded in the notebook: at 1,204,741 params vs Model 4's 863,557, Model 5 is
only **~1.4×** the previous rung — the tightest gap on the ladder (Models 2→3 and 3→4 are ~11×
and ~9×) **and** the only change of architecture family. A Model 4 vs Model 5 difference is
therefore better read as **convolution-vs-attention** than as another step along the parameter
axis. patch_size = 60 was chosen so attention weights map one-to-one onto the CMI region grid
with no aggregation step — this is what makes the Model 5 attention-weight analysis possible.

Trains on the UNMODIFIED ladder-wide protocol. Verified in the pre-checks that the transformer
learns with **no learning-rate change, no warmup, no optimizer change** — same fixed split
(17,742 / 2,258), untouched 40,145 test set, same stopping rule (`val_balanced_accuracy`, max,
patience 10, min_delta 0.002, max_epochs 100), same 5 seeds [0–4], same MPS device, batch size,
optimizer, and `run_all_seeds` helper. The top rung is not a special case, so the complexity
comparison stays fully controlled. See the ladder-wide entries rather than a re-statement here.

Pre-check findings (Step 1, before finalising the notebook):
  - **MPS op support:** all ops MPS-native — forward+backward runs with no
    `NotImplementedError`, so **no `PYTORCH_ENABLE_MPS_FALLBACK` needed** (and thus no per-op
    CPU-transfer penalty). Timing, batch-16 forward+backward: MPS **7.2 ms/step** vs CPU
    **54.4 ms/step** (~8×).
  - **Determinism, with the divergence control:** two MPS runs of seed 0 were **bit-identical**
    (max|Δ train_loss| = **0.0**), *and* seeds 0 vs 1 genuinely diverged (max|Δ| = **0.063**).
    The second number is the control: it rules out the bit-identity being an artefact of dropout
    being silently inactive. So the transformer — dropout and all — carries **no extra backend
    variance**; its 5-seed error bars rest on the same footing as the CNNs'. (Both numbers logged
    because the control is what makes the finding meaningful.)
  - **Memory:** batch-16 train step ~155 MB; a 512-row prediction chunk ~1.35 GB on MPS; the
    40,145-epoch test set runs in 79 chunks of 512 (the same chunked-predict path as Models 2–4).
    Batch 16 fits comfortably.
  - **Learnability smoke test:** 5 epochs on the full 17,742 set with the exact protocol —
    train_loss 1.48→0.88, val balanced accuracy 0.42→0.59 (well above 0.20 chance, still rising),
    ~14.4 s/epoch on MPS. Learning cleanly; no STOP, no protocol change.

No new judgement calls beyond those already logged for Model 2 (majority floor = N2, per-seed
checkpoints, batched test inference, no inner epoch bar), which carry over unchanged. The §5
comparison is extended to all four rungs; the transformer's RF shows as `global` and its RF
delta as `—` (no finite receptive field to difference).

## Decision: Model 5 side experiment — transformer at patch_size = 15 (resolution test) (14 Aug 2026)

A SIDE EXPERIMENT appended to the Model 5 notebook (§6), not a replacement: the patch-60 model
remains *the* Model 5 ladder rung, with its results, cells and artifacts intact. Prepared, not
run — the user trains it.

Motivation. Model 5 at patch 60 (50 tokens) reached balanced accuracy 0.6603 ± 0.0049 — the
WORST rung on the ladder, below even the 8,181-param shallow CNN (0.7133). The losses
concentrate in N1 (recall 0.573 → 0.350) and REM (0.739 → 0.630), which bleed heavily into each
other, while W and N2 hold up. Resolution hypothesis: a 600 ms patch is embedded by a single
linear projection, so fine within-patch structure is compressed before attention sees it, and
N1/REM are the stages that most depend on that structure. patch 15 (200 tokens, 4 per 60-sample
region) tests whether the underperformance is a consequence of the PATCHING choice or is
ARCHITECTURAL.

Origin / pre-registration — stated precisely. patch = 15 is Sleep-EDF's measured ~15-sample
autocorrelation coherence grain (the 1/e correlation length; see the region-size and CNN-kernel
entries), i.e. the same signal scale the CNN kernel = 15 was set to — so it is a signal-grounded,
pre-registered SCALE, not an arbitrary choice. Honesty note: the log did not previously contain
an explicit "if the transformer underfits, fall back to patch 15" clause; the pre-registration
is of the 15-sample scale, and this entry is where that scale is first applied as a transformer
patch size. (Recording this so the provenance is not overstated.)

Promotion criterion — fixed BEFORE the result. patch 15 is promoted to *the* Model 5 rung ONLY
if balanced accuracy recovers into the CNN range (~0.71+). A marginal gain does not justify the
cost: patch 60 gives exactly one token per CMI region (attention weights map onto the region
grid with no aggregation), whereas patch 15 gives four tokens per region, requiring a documented
within-region aggregation step for the attention-weight analysis. Either way the patch-60 result
stays in the thesis: if patch 15 recovers accuracy that is evidence the underperformance was a
resolution artifact rather than attention being unsuited to the task; if not, it strengthens the
architectural reading. The comparison belongs in the discussion regardless of outcome.

Protocol. Only patch_size changes. Identical fixed 17,742/2,258 split, untouched 40,145 test
set, stopping rule (`val_balanced_accuracy`, max, patience 10, min_delta 0.002, max_epochs 100),
5 seeds [0–4], MPS, batch size and `run_all_seeds` helper. Distinct `MODEL_NAME =
"model5_transformer_patch15"` → separate JSONs, checkpoints and confusion figure; no patch-60
artifact is overwritten, renamed or moved.

Figures. 1,218,181 params (+13,440 vs patch-60's 1,204,741, from the 200 × d_model positional
embedding vs 50 × d_model, partly offset by the smaller patch-embed conv); 200 tokens
(3000 / 15, exact, no padding); ~16× the self-attention cost of patch 60 (O(tokens²)), so
notably slower per epoch.

Outcome (run 14 Aug 2026) — a FALSIFIED HYPOTHESIS, not merely a rejected variant. Promotion
criterion NOT met and patch 15 is WORSE on both headline metrics: balanced accuracy 0.6395 vs
patch 60's 0.6603, macro-F1 0.5989 vs 0.6296. No promotion — patch 60 stays THE Model 5 ladder
rung.

The resolution hypothesis is NOT supported. It predicted N1 and REM would recover with the finer
patch; instead **N1 got WORSE (recall 0.350 → 0.316)**, **REM improved only marginally
(0.630 → 0.653)**, and **N2 dropped notably (0.674 → 0.601)**. So the transformer's
underperformance is not a within-patch resolution artifact.

The alternative the evidence points to is **sample efficiency, not resolution.** Patch 15 has 200
tokens vs 50 — more positional structure to learn — on the *same* 17,742 training epochs. Its
seeds stopped earlier (**18–21 epochs vs patch 60's 22–31**) with earlier best epochs (**8–11 vs
12–21**): it overfits sooner and settles worse. Making the problem finer-grained made the data
limitation worse, not better.

Consequence for the thesis (discussion chapter): the transformer's underperformance is better
attributed to **the architecture's data requirements at this scale** than to the patching choice.
That is a stronger claim than either "we chose the patch size badly" or "we didn't check" — and it
is available *only because the patch-15 fallback was pre-registered and then tested*. All patch-15
artifacts (JSONs, checkpoints, confusion figure) are KEPT as the evidence for this argument, not a
failed run to discard: the falsification is itself a result.

## Decision: harness change — perturbations_per_eval on kernel_shap (14 Aug 2026)

`harness/xai/kernel_shap.py::kernel_shap` gains an additive keyword `perturbations_per_eval`
(default 1). It is a pure COMPUTE-LAYOUT knob — it does not change which coalitions are sampled
(that is fixed by `n_samples` + `seed`), only how many are evaluated per forward call. Gate
evidence: `pe=1` vs `pe=200` attributions agree to **max|Δ| = 8.01e-08** (float64 rounding), so
results are unaffected. Motivation: batching many coalitions into one forward is a ~13× speed-up
on MPS (one batched forward vs thousands of tiny batch-1 forwards) but a ~13× *loss* on CPU — the
lever that makes a converged KernelSHAP `n_samples` affordable (see next entry and §6 of the
methodology notebook). Default 1 reproduces prior behaviour exactly.

## Decision: XAI/CMI phase parameters, justified empirically (14 Aug 2026)

The XAI/CMI faithfulness phase (Models 2–5, attribution over the 50-region temporal grid) has
parameters chosen from measured evidence rather than derived — the first such in the thesis. The
executable evidence lives in `sleep_edf/notebooks/methodology/00_methodology_checks.ipynb`; this
entry records the decisions and, where relevant, their limitations.

KernelSHAP n_samples = 8000 — a DECISION and a documented LIMITATION.
    KernelSHAP over 50 regions does **not** fully converge within tractable compute. On Model 2
    seed 0 (10 stratified samples, zero PM), sweeping n_samples ∈ {200, 500, 1000, 2000, 4000,
    8000}: CMI climbs monotonically 0.154 → 0.259 → 0.292 → 0.338 → 0.350 → 0.379 (n=200 is under
    HALF the n=8000 value), and consecutive-setting Spearman on the attribution vectors only
    reaches **mean 0.93 / min 0.75** by 4000→8000 — approaching but not at stability. 8000 is
    chosen as the best-converged setting affordable with MPS coalition-batching (~1.34 s/sample vs
    ~11.8 s unbatched). Evidential status, stated in the notebook: the **Spearman** curve is the
    primary evidence (computed on the attributions directly, not aggregated over samples, so free
    of small-N noise); the **CMI** curve is supporting only (aggregated over 10 samples, noisy).
    LIMITATION: absolute CMI is biased **low** (still rising at 8000, ≥~8% short of converged), so
    **absolute CMI values are comparable WITHIN this study only — not against published CMI
    figures.** The ladder COMPARISON stays valid because the same n_samples (hence the same
    systematic bias) is applied to every rung — an assumption tested in §2 by re-running the sweep
    on Model 5 and checking its trajectory tracks Model 2's (if Model 5 converges materially
    slower, that is a STOP-and-report, not a per-model n_samples change).

Perturbation method = zero (primary); laplace = pre-registered robustness check.
    On per-epoch z-scored data sample_mean ≈ zero, so the choice is zero vs laplace. zero is
    primary, decisively because of **stage-dependence**: laplace is a curvature/edge operator that
    preserves sharp edges (spindles, K-complexes) and flattens smooth stretches (slow waves), so
    its hiding strength varies by sleep stage and would bias per-class faithfulness on a 5-class
    problem; zero hides every region uniformly. Also: zero keeps the whole pipeline on one baseline
    (the oracle/FeatureAblation and concentration use zero), is the cleanest ablation on z-scored
    EEG ("remove this region's deviation from baseline"), and matches the validated Model 1 / ECG200
    machinery. laplace is retained as a single-model robustness check (one model, not across the
    grid), its differing artefact profile being exactly what such a check should probe.

Device — per-method, measured (see §6).
    FeatureAblation on CPU (44 ms/sample; 50 tiny batch-1 forwards, MPS transfer overhead
    dominates), Integrated Gradients on MPS (0.71 s vs 1.27 s CPU; one batched forward+backward),
    KernelSHAP MPS-**batched** (perturbations_per_eval=200: 0.37 s @ n=2000 / 1.34 s @ n=8000, vs
    ~11.8 s unbatched). The KS device choice is CONTINGENT on batching: unbatched it is faster on
    CPU, batched it must run MPS (batching is a loss on CPU). Choices verified not to affect
    results: cross-device attribution agreement max|Δ| ≈ 1e-7–2e-6; batching equivalence
    max|Δ| = 8e-8.

Target class = PREDICTED class, ladder-wide (CONFIRMED 14 Aug 2026).
    This is the thesis's central FAITHFULNESS-vs-PLAUSIBILITY distinction appearing as a concrete
    implementation choice, and should be written up as that distinction in the methods chapter —
    not as a technical detail. CMI measures whether an explanation reflects what the model ACTUALLY
    COMPUTED. On a misclassified sample the model made a decision — the wrong one — and the
    explanation must account for THAT decision. Attributing toward the true class would score an
    explanation of a decision the model NEVER MADE: that is a PLAUSIBILITY question ("did the model
    look at the right thing?"), not a FAITHFULNESS one ("does the explanation reflect the
    computation?"). The headline CMI is therefore predicted-class, identical across the ladder.
    Not hypothetical: 3 of the 10 stratified sweep samples were misclassified (W→N1, N1→N2, REM→N2),
    so predicted≠true affects a material fraction of the evaluation set — and disproportionately N1
    (Model 2 N1 recall ≈ 0.45), the hardest minority stage. True-class attribution remains available
    as a SEPARATE plausibility analysis if wanted, but never as the faithfulness metric.

Still pending (methodology notebook §3):
    - Evaluation sample count N: pending the bootstrap stability of CMI/PES vs N at n=8000, which
      also tests whether the proposed N = 500 is enough and whether the nonlinear CNN's PES drops
      below Model 1's linear-artifact PES = 1.0.

## Investigation: the KernelSHAP convergence-vs-CMI confound (Model 2 vs Model 5) (14 Aug 2026)

The §2 methodology sweep raised a possible confound that could have manufactured this thesis's
central claim by artifact: Model 5's attribution rank-stability is WORSE than Model 2's at every
n_samples (~0.05 lower), yet Model 5's CMI is HIGHER (0.62 vs 0.38 at n=8000). Naively "less
converged ⇒ noisier ranking ⇒ lower CMI", so the opposite looked suspicious — and an
under-converged transformer faking a faithfulness change is the most damaging confound available
here. Investigated directly (Model 2 & 5 seed 0, the 10 methodology samples, n_samples up to
16000). n_samples stays 8000 ladder-wide; the following is the evidence, not just the setting.

1. DIRECTION-OF-BIAS ARGUMENT (the resolution). "Convergence" (Spearman) and "CMI" are different
   axes: Spearman measures KernelSHAP ESTIMATOR NOISE (how stably the ranking is pinned down as
   coalitions are added); CMI measures the important-vs-unimportant SEPARATION (a property of the
   model + attribution). Estimator noise pushes any ranking toward random, and a random ranking
   gives DDS ≈ 0 — so **under-convergence deflates CMI toward 0 and can NEVER inflate it.**
   Measured, not asserted: CMI rises monotonically with n_samples for BOTH models —
       n_samples   2000    4000    8000    16000
       Model 2    0.338   0.350   0.379   0.401   (still rising)
       Model 5    0.529   0.593   0.623   0.665   (still rising)
   more samples (less noise) ⇒ higher CMI, both models. So Model 5 being LESS converged than
   Model 2 at n=8000 only DEFLATES its CMI; its true converged CMI advantage is even larger than
   the observed 0.62 vs 0.38. The asymmetry, stated explicitly: **a HIGH CMI on an under-converged
   model is SAFE (under-sampling can only have lowered it); a LOW CMI is NOT safe (under-sampling
   and genuine unfaithfulness both push down and are indistinguishable).** This is exactly why the
   dip check below exists.

2. THE TOP-k EXONERATION WAS TESTED AND FAILED (recorded as part of the evidence). The hoped-for
   resolution was that the instability lived only in the ranking TAIL (which the 25-step deletion
   curves never reach), leaving the CMI-relevant top intact. It does not: Model 5's top-10 ORDER
   stability lags Model 2's by ~0.05–0.15 at the high-n pairs (e.g. or10 0.67 vs 0.87 at
   8000→16000), and top-5/10 set overlap lags by ~0.05–0.08. Only the top-25 SET overlap matches
   (~0.86 both: the same regions are flagged important; their fine ordering is noisier for M5). So
   the instability DOES reach the top of the ranking the deletion curves use. The comparison is
   therefore defended by the direction-of-bias argument (1), NOT by the instability being confined
   to the tail. (Reporting a failed check is deliberate — it is part of the evidential record.)

3. MECHANISM — why Model 5's CMI is genuinely higher (n=8000, 10 samples). Two artifacts ruled out
   because the models are identical on them: **concentration** 0.091 (M2) vs 0.092 (M5) — Model 5
   is NOT more concentrated, so the higher CMI is not the mechanical "relies on fewer regions"
   effect the concentration control exists to catch; and **start confidence** (unperturbed
   pred-class prob ×100) 77.8 vs 77.7 — no headroom artifact. The driver is genuine separation in
   the curves: **MoRF** (delete most-relevant first) M2 77.8→54.9 by step 5 vs M5 77.7→**24.9**
   (far steeper); **LeRF** (delete least-relevant first) M5 HOLDS ~80 through step 10 (a merely
   fragile model's LeRF would drop — this is the clincher for genuine separation). The
   random-ranking control confirms it: **KS beats random deletion at step 5 by 35.5 points for M5
   vs 16.7 for M2** — Model 5's attributions front-load impact ~2× better than chance relative to
   Model 2's.

4. STATED CMI LIMITATION (present but minor, ratio NOT quantified). Model 5 is somewhat more
   perturbation-sensitive in general: under RANDOM deletion it drops faster than Model 2 (step-5
   random MoRF 60.4 vs 71.6). So CMI partly reflects a model's INTRINSIC perturbation sensitivity,
   not attribution quality alone. This component is minor — dominated by genuine separation (the
   flat LeRF and the 2× KS-vs-random margin) — but real, and should be stated as a CMI limitation
   in the methods. The exact split between "genuine faithfulness" and "intrinsic sensitivity" is
   NOT quantified from what was run; no number is invented for it.

DECISIONS from this investigation:
   - n_samples = 8000 kept ladder-wide; NOT raised to 16000. Measured directly (not extrapolated):
     the Model-5-minus-Model-2 Spearman gap is essentially CONSTANT (0.052 at 4000→8000, 0.051 at
     8000→16000), so it is a STRUCTURAL property (the transformer's Shapley values over 50 regions
     are inherently harder to estimate), not a sampling deficiency that closes; and CMI has not
     converged at 16000 either. ~10 h of ladder compute for a marginal (~0.01) Spearman gain and no
     gap closure is not worth it.
   - TARGETED dip check adopted (verification placed where the confound is live). Any rung whose CMI
     shows a DOWNWARD DIP relative to its neighbours gets its top-25 set overlap and top-10 order
     stability compared against those neighbours, so the dip cannot be an under-sampling artifact.
     A high CMI needs no such check (per the asymmetry in 1). This replaces a blanket n_samples
     increase with a check only where under-sampling and genuine unfaithfulness are confusable.
   Evidence: methodology notebook §1/§2; this investigation was report-only (no notebook/harness
   change).

## Decision: Model 2 XAI/CMI run — the faithfulness-phase pilot (14 Aug 2026)

Notebook `sleep_edf/notebooks/model2/10_model2_xai_cmi.ipynb` (prepared, not executed — the user runs
it). Written INLINE (mechanics visible, not behind an opaque helper) because it is also the reference
the methods chapter is written from. Settings are the phase-level logged decisions — see the "XAI/CMI
phase parameters", "target class = predicted", "perturbations_per_eval", and convergence-investigation
entries above; NOT restated here. In brief: n_samples 8000 with perturbations_per_eval 200 (KernelSHAP),
zero PM, predicted-class target (passed as the harness default target_class=None, so an attribution and
its deletion curves track the same class), 5 seeds aggregated mean ± std, per-method device split (FA
CPU, IG MPS, KS MPS-batched, deletion curves + concentration CPU).

Evaluation subset (fixed ladder-wide). N = 500 test epochs stratified by TRUE class (100/class), seed
42, built once and saved to `results/metrics/xai_eval_subset_idx.npy` (+ a `_meta.json` sidecar), loaded
thereafter — shared across all 5 seeds and all four models, for the same reason the training subsample
and validation split are fixed (evaluating models on different samples would confound the ladder
comparison). Adequacy confirmed by the methodology bootstrap (CMI std ≈ 0.006 at N = 500, mean stable
from N ≈ 100).

Cell order is deliberate so the cheap work is interpretable while the long one runs: §1 setup · §2
subset + per-seed misclassification breakdown · §3 FeatureAblation (~2 min, CPU) · §4 Integrated
Gradients (~1 min, MPS) · §5 SAVE FA+IG before KernelSHAP · §6 CMI for FA/IG (~3 min) · §7 concentration
(~1.5 min) · §8 FA-vs-IG presentation · §9 KernelSHAP (~56 min, MPS, saves per seed) · §10 KS CMI +
dip-check numbers · §11 all-three presentation + verdict. §1–§8 run in ~7 min; §9 saves each seed as it
finishes so an interruption loses at most one seed and a re-run skips seeds already on disk.

Decisions made that were not specified:
  - Notebook numbering `10_...`: opens the XAI series (10–13 for Models 2–5), continuing past the
    training notebooks (06–09); Model 1's XAI was 05.
  - No ground-truth recovery check (Models 2–5 have no readable coefficients). CMI read alongside
    concentration (the confound control) is the headline, per brief — no substitute invented.
  - Attribution heatmaps presented as mean attribution per (PREDICTED class × region) over the 30 s
    epoch, averaged over seeds — the temporal-region analogue of Model 1's stage × band heatmap, grouped
    by predicted class to match the predicted-class rule.
  - Dip-check (top-25 set overlap, top-10 order stability, KS n_samples 4000 vs 8000) computed on a
    25-sample subsample and RECORDED, not applied as pass/fail: Model 2 is the bottom rung with no
    neighbour to dip against; the numbers exist for the cross-ladder dip-check (any rung whose CMI dips
    below its neighbours), per the logged policy.
  - IG cost clarified: the earlier ~710 ms/sample device figure was MPS warm-up on the first calls;
    steady-state IG is ~5 ms/sample, so §4 is well under a minute (the notebook warms MPS before timing
    so its printed estimate is honest). No harness change needed.
  - Artifacts written: `xai_eval_subset_idx.npy` (+meta), `model2_xai_fa_ig_attr.npz`, per-seed
    `model2_xai_ks_attr_seed{seed}.npy`, `model2_xai_cmi_results.json`, and two heatmap figures.
## Decision: Model 3 XAI/CMI run — second faithfulness rung, first ladder comparison (14 Aug 2026)

Notebook `sleep_edf/notebooks/model3/11_model3_xai_cmi.ipynb` (prepared, not executed — the user runs
it). Mirrors the Model 2 XAI notebook (`10_...`) section-for-section, inline and readable; opens as the
second entry in the XAI series (10–13 for Models 2–5). Only the model changes: Model 3 (medium CNN
[32,64,64], 93,285 params, RF 1060 ms), its 5 seed checkpoints. All settings, the evaluation subset and
the code are identical to Model 2 — see the phase-level entries above (n_samples 8000 + pe 200, zero PM,
predicted-class target, per-method device split FA-CPU/IG-MPS/KS-MPS-batched, N=500, 5 seeds); NOT
restated. Nothing about Model 3 forced a deviation.

Evaluation subset: LOADS the shared `results/metrics/xai_eval_subset_idx.npy` (built by Model 2) — the
SAME 500 samples — and RAISES if the file is absent rather than rebuilding, so Model 3 can only ever run
on the identical subset. Model 3's misclassification profile differs from Model 2's (its N1 recall 0.553
vs 0.453) — surfaced per seed because the predicted-class rule means attribution follows the predicted,
not the true, class.

Cell order identical to Model 2 (cheap methods → save → CMI/concentration/presentation → KernelSHAP last,
saved per seed). Measured Model 3 costs (medium CNN): FA ~2 min, IG ~0.5 min, CMI(FA+IG) ~4 min,
concentration ~2 min, KernelSHAP ~37 min (its batched KS is ~0.9 s/sample, a touch faster than Model 2's
shallow at ~1.3 s). §1–§8 ≈ ~10 min, then §9 ~37 min.

New in §11 — the first LADDER COMPARISON (Model 2 vs Model 3): CMI, DDS, PES and concentration per method,
plus cross-method rank agreement, with deltas. Model 2's numbers are read from its saved
`model2_xai_cmi_results.json` (never hardcoded); if that file is absent (Model 2's §11 not yet run) the
cell says so and skips. Model 3 is the rung where the hypothesised faithfulness DIP would appear, so §10's
dip-check numbers (KS top-25 set overlap, top-10 order stability, n_samples 4000 vs 8000) are framed for
the live comparison: if Model 3's KernelSHAP CMI comes in below Model 2's, those numbers say whether it is
a genuine dip or an under-sampling artefact (under-sampling deflates CMI). Watch-threads surfaced
explicitly in §11: PES (Model 2 stayed ~1.0; if it stays pinned, CMI ≈ DDS in this setting — a finding
about the metric), concentration (confound control), and whether the GAP-head heatmaps stay near-uniform
within the epoch despite the 1060 ms RF.

Decisions made that were not specified:
  - §2 is load-ONLY (raises if the shared subset is missing) rather than build-or-load, to guarantee Model
    3 uses the identical 500 samples Model 2 used.
  - Model 3's results JSON additionally stores the cross-method `agreement` values (Model 2's JSON predates
    that key); the §11 comparison therefore reads Model 2's agreement by RECOMPUTING it from Model 2's
    saved attribution arrays (`model2_xai_fa_ig_attr.npz` + per-seed KS `.npy`) when the JSON lacks it, and
    skips that delta gracefully if those files are incomplete (e.g. Model 2's KernelSHAP not yet finished).
  - Notebook `11_...`, figures `sleep_edf_11_model3_xai_*`, artifacts `model3_xai_*` — parallel to Model 2's
    `10_/sleep_edf_10_/model2_xai_*`.

## Decision: Model 4 XAI/CMI run — deep CNN, the controlled-complexity rung (14 Aug 2026)

Notebook `sleep_edf/notebooks/model4/12_model4_xai_cmi.ipynb` (prepared, not executed — the user runs
it). Mirrors the Model 3 notebook (`11_...`) section-for-section, inline; third entry in the XAI series
(10–13). Only the model changes: Model 4 (deep CNN [64,128,128,256], 863,557 params, RF 2260 ms), its 5
seed checkpoints. All settings, the shared evaluation subset and the code are identical — see the
phase-level entries above (n_samples 8000 + pe 200, zero PM, predicted-class target, FA-CPU/IG-MPS/
KS-MPS-batched, N=500, 5 seeds); NOT restated. Nothing forced a deviation.

Model 4 is the ladder's CONTROLLED complexity comparison: ~9× Model 3's parameters (863,557 vs 93,285)
at essentially identical balanced accuracy (0.7439 vs 0.7453), so a CMI change 3→4 is attributable to
complexity rather than competence — not true of the 2→3 step (where accuracy also rose). Noted in the
notebook markdown.

Eval subset: LOADS the shared `xai_eval_subset_idx.npy` (same 500 as Models 2/3); raises if missing.

Costs (measured; the deep CNN's forward is much slower): FA ~6.5 min, IG ~2 min, CMI(FA+IG) ~12 min,
concentration ~6 min, KernelSHAP ~2 HOURS (~2.8 s/sample × 500 × 5, vs Model 3's ~0.9 s → ~37 min). So
§1–§8 ≈ ~27 min, then §9 ~2 h. Estimates set accordingly; §9 still times seed 0 and projects live, and
saves per seed (interruption loses ≤1 seed).

§11 extended to a THREE-rung ladder comparison (Models 2, 3, 4): CMI/DDS/PES per method + concentration +
cross-method agreement, with the last-step delta, reading Models 2 and 3 from their saved results JSONs
(missing JSON reported and that rung skipped — nothing hardcoded). Added a CMI-vs-concentration plot
across the rungs so the faithfulness trend and the concentration confound can be read directly (CMI rose
0.428→0.475 while concentration rose 0.087→0.115 through Model 3, so part of the rise may be mechanical).
Watch-threads surfaced explicitly per the brief: CMI DIRECTION (has risen twice; if Model 4 continues,
the ladder shows faithfulness INCREASING with complexity, against the monotonic-degradation hypothesis);
PES CEILING (Model 2 0.990/0.998, Model 3 pinned 1.000 — if Model 4 also pins, CMI ≈ DDS and the
consistency term is not discriminating, a finding about the metric); CONCENTRATION as confound (plotted
against CMI); DIP CHECK (computed though nothing has dipped); HEATMAP STRUCTURE (Model 2 near-uniform,
Model 3 class-level but not within-class localised — whether Model 4's 2260 ms RF localises).

Decisions not specified: notebook `12_...` / figures `sleep_edf_12_model4_xai_*` / artifacts
`model4_xai_*`; the ladder-comparison cell iterates over available previous-rung JSONs (extends cleanly
to Model 5); the CMI-vs-concentration plot is a new figure `sleep_edf_12_model4_xai_cmi_vs_concentration.png`.

## Decision: PES has not dropped for the nonlinear CNNs — a finding about CMI, not the models (14 Aug 2026)

The named watch-thread — whether PES (the consistency term in CMI) would fall below Model 1's PES = 1.0
once the models are nonlinear — resolves NEGATIVE. Across Models 2 and 3, PES sits at **0.976–1.000**
(most method/seed values exactly 1.000; Model 3 was exactly 1.000 on all 5 seeds for both FA and IG). The
original expectation, that Model 1's PES = 1.0 was a *linear-model artifact*, is **not supported** — PES
is **saturating at the ceiling**, not breaking away from it.

Likely reason (belongs in the discussion, as a property of the METRIC not the models): PES measures only
the SIGN-CONSISTENCY of the per-sample DDS (fraction of samples with DDS > 0 minus fraction < 0). Failing
it would require an attribution whose deletion curves are *anti-correlated* with the model's reliance
(DDS negative on a material fraction of samples) — i.e. an explanation that is worse than random. Any
attribution method carrying real signal clears that bar on essentially every sample, so PES pins at ~1.0.
Consequence: since CMI is the harmonic mean of |DDS| and |PES|, **with PES at the ceiling CMI carries no
information beyond DDS in this setting — the consistency component is not discriminating.** This is a
limitation of CMI as applied here, to be stated in the discussion, and it re-reads the whole faithfulness
ladder as effectively a DDS ladder.

Notebook fix (so the finding is not misread): the §11 PES watch-thread flag in the Model 2/3/4 XAI
notebooks was over-reading "any value < 1.0" as PES breaking away. Replaced with three bands — SATURATED
(min PES >= 0.97, most = 1.000; the observed case, reported with the CMI-carries-no-info-beyond-DDS
consequence), near-ceiling ([0.9, 0.97): mostly saturated with single-sample noise), and GENUINE DROP
(min PES < 0.9: the consistency component is discriminating). Applied to Models 2, 3 and 4 notebooks and
the Model 4 builder (which Model 5's XAI notebook will be templated from).

## Decision: Model 5 XAI/CMI run — transformer (patch 60), attention as a 4th method (14 Aug 2026)

Notebook `sleep_edf/notebooks/model5/13_model5_xai_cmi.ipynb` (prepared, not executed; new file alongside the
Model 5 training notebooks, which were not touched). Mirrors the Model 4 notebook, adds §5 (attention).
Fourth and final rung of the XAI series (10–13). Model 5 = transformer, PATCH 60 (the ladder rung, not the
falsified patch-15 variant): 1,204,741 params, 50 tokens. Settings identical to Models 2–4 (see phase-level
entries: n_samples 8000 + pe 200, zero PM, predicted-class target, FA-CPU/IG-MPS/KS-MPS-batched, N=500,
5 seeds; the shared 500-sample eval subset is LOADED). The 4→5 step is framed in the notebook as
architecture-family (conv→attention) + fit quality (bal. acc 0.6603 vs 0.7439), NOT a complexity step
(~1.4× params, the tightest gap on the ladder). Costs measured: FA ~7 min, IG ~1.5 min, attention ~1 min,
CMI(FA+IG+Att) ~21 min, concentration ~6 min, KernelSHAP ~1.5 h; §1–§9 ≈ ~36 min then §10.

ATTENTION — the harness wiring and the reduction decision.
  - The harness attention module (`harness/xai/attention.py`) is an UNIMPLEMENTED STUB
    (`extract_attention_weights` raises NotImplementedError; its "cls_token" option is inapplicable — the
    transformer has no CLS token, it mean-pools). `nn.TransformerEncoder` does not return attention weights.
    So attention is extracted INLINE in the notebook by reproducing the encoder forward layer-by-layer
    (respecting norm_first) with `self_attn(..., need_weights=True)` — verified to match `model()` logits
    EXACTLY (max|Δ|=0). No harness change (the stub was not touched).
  - Reduction to 50 per-region scores (a METHODS decision, taken by the supervisor, logged as CHOICES not
    defaults): LAST LAYER (not rollout, not mean-over-layers) → mean over the 8 heads → mean over the 50
    queries = each region's "attention received". One token ↔ one 60-sample region (patch 60), so no
    token→region aggregation. Rationale: the thesis question is whether RAW attention — what practitioners
    read off a model — tracks feature importance (Jain & Wallace 2019); rollout would test a *repaired*
    version and make a good score ambiguous between vindicating attention and validating the correction.
    Head=mean and query=attention-received fit the mean-pooled model; attention is class-AGNOSTIC (one
    ordering per sample) and non-negative. **Attention rollout (Abnar & Zuidema 2020) is a PRE-REGISTERED
    FOLLOW-UP:** if last-layer attention scores poorly on CMI, "does rollout recover it?" is planned, so
    running it later is a contingency, not an improvised response.

FINDING (surfaced in §5, expected to hold at N=500): this transformer's attention COLLAPSES TO EXACTLY
UNIFORM in its deeper layers — std across the 50 regions is 0.0 from layer 3 onward (only layer 0 has
structure, std ~0.013). So LAST-LAYER attention received = 1/50 for every region: **no per-region signal.**
Consequence for the chosen reduction: the attention attribution is DEGENERATE — undefined region ordering,
CMI ≈ 0, and rank-agreement with FA/KS/IG undefined (a constant vector has no Spearman). This is a genuine
result — the extreme "attention ≠ feature importance" case, consistent with deep/underfit transformer
rank-entropy collapse — NOT a pipeline bug. §5 reports the `std across regions` and flags uniformity
explicitly; the pre-registered rollout follow-up is now clearly motivated (earlier layers still carry
structure).

§12 four-method presentation + four-rung ladder comparison (Models 2–5): CMI/DDS/PES for the three shared
methods across all rungs with per-step deltas (read from saved JSONs, nothing hardcoded); Attention reported
Model-5-only; concentration + the CMI-vs-concentration plot across the four rungs; six pairwise cross-method
agreements with the three ATTENTION pairs surfaced (the Jain & Wallace watch-thread). The dip-check (§11) is
flagged as load-bearing at this rung: Model 5's KernelSHAP is the least-converged, so a LOW KS CMI vs Model 4
must be read against the §11 top-25/top-10 stability numbers (under-sampling deflates CMI); a HIGH one is
safe. PES flag uses the corrected saturated/near-ceiling/genuine-drop bands.

Judgement calls: attention extracted inline (harness stub unimplemented; verified faithful; no harness edit);
`pair_agree` made robust to undefined per-sample Spearman (skips constant/near-uniform-attention samples and
reports the fraction skipped — heavy skipping is itself the uniform-attention signal); attention treated as
Model-5-only in the ladder comparison; artifacts `model5_xai_*` / figures `sleep_edf_13_model5_xai_*`.

## Decision: Model 5 attention — last-layer is the primary result; rollout is a secondary follow-up (16 Aug 2026)

Diagnostic ruled out an extraction bug (post-softmax weights confirmed — rows sum to 1.0; last-layer output
bit-identical for a predicted-W vs predicted-N3 sample while layer 0 differs by 0.9995 on the same pair — so
the pipeline reacts to the input, the last layer does not). The precise finding:

  - Model 5's LAST-LAYER attention is EXACTLY UNIFORM — 1/50 = 0.02, std 0.000000 across all 8 heads — and
    input-INDEPENDENT (bit-identical for predicted-W vs predicted-N3). Layers 0 and 2 carry input-dependent
    structure (per-region std 0.0158 and 0.0084); layers 1, 3, 4, 5 are uniform.
  - This is attention RANK COLLAPSE / over-smoothing (representations collapse so last-layer attention logits
    are equal across keys -> uniform softmax), a known transformer failure mode, and it is consistent with
    Model 5's other symptoms: worst rung on the ladder (0.6603 balanced accuracy) and the patch-15 result
    pointing at sample efficiency rather than resolution.
  - The resulting attention CMI (0.058 ± 0.033, PES 0.112) is REAL but DEGENERATE, not informative: ranking
    50 identical values gives an arbitrary order, so MoRF and LeRF delete near-randomly and the curves cannot
    separate. ANY constant vector scores the same.
  - Framed precisely against the literature — and the two attention treatments land on DIFFERENT claims:
    Jain & Wallace (2019) found attention that HAS structure but fails to track feature importance. LAST-LAYER
    attention here has NO structure to assess (uniform) — the degenerate limiting case, NOT the Jain & Wallace
    case. ROLLOUT (below) is where the Jain & Wallace case is actually demonstrated: it has structure, and that
    structure is uncorrelated with feature importance. The write-up must not conflate the two.
  - Consequence: LAST-LAYER attention's CMI is NON-COMPARABLE to the other three methods on the ladder — FA, KS
    and IG measure attribution quality, whereas the last-layer score measures the ABSENCE of any attribution at
    all. It must not be read as a fourth point on the same faithfulness scale.

DECISION: last-layer attention STANDS as the primary reported attention result — it is what practitioners read
off a model, and switching the reduction after seeing it was degenerate would be selecting the recipe that
produces a usable answer. Attention ROLLOUT (Abnar & Zuidema 2020) is run as a clearly-labelled SECONDARY
analysis (§13, appended to the Model 5 notebook after the primary cells; it does not re-run or touch
KernelSHAP), saved to a SEPARATE file (`model5_xai_rollout_results.json`) so the primary results JSON is
untouched.

ROLLOUT RESULT (run 16 Aug 2026) — this is where the stronger claim comes from. Rollout is NON-degenerate: it
propagates attention through all layers via the residual path (0.5A+0.5I per layer, row-normalised, multiplied
across layers), so it incorporates layers 0/2 where input-dependent structure survives. But its faithfulness
is essentially nil: **CMI 0.078 ± 0.035, PES 0.279, and rank agreement 0.023 (FA) / 0.034 (IG) — i.e. ~ZERO
correlation with what the perturbation methods identify as important.** PES 0.279 means rollout does not even
get the deletion DIRECTION consistently right, against ~1.0 PES saturation for every other method on every
rung.

  - THIS is the Jain & Wallace case, and it is now properly DEMONSTRABLE: attention that HAS structure but
    whose structure is UNCORRELATED with feature importance. Their NLP finding reproduced in a TIME-SERIES
    setting — the open question named in the proposal. The correlation is well-defined here precisely because
    rollout is non-degenerate (unlike last-layer, where a constant vector makes Spearman undefined).
  - BOUND THE CLAIM. This transformer UNDERFIT (0.6603 balanced accuracy, the worst rung on the ladder), and
    attention collapse plausibly FOLLOWS from underfitting. The demonstrated claim is that attention failed as
    an explanation FOR THIS MODEL. Whether it would fail for a WELL-FIT transformer on this task is UNTESTED
    and is named as FUTURE WORK — not claimed here.
  - PRE-REGISTRATION (both treatments specified BEFORE results were seen): last-layer as PRIMARY with a stated
    rationale (what practitioners read off a model; testing raw attention, not a repaired version), and rollout
    PRE-REGISTERED as the contingency for a degenerate last-layer. Neither was chosen after seeing which gave a
    usable answer. This is what makes the negative result credible rather than a post-hoc search.

Submission-prep flag (NOT actioned — needs a decision): `harness/xai/attention.py` is an unimplemented STUB
(raises NotImplementedError) with an ECG200-era docstring ("ECG inputs", "(n_samples, n_timesteps)", a CLS-
token option that does not apply — the transformer mean-pools). It is UNUSED by Model 5 (attention is
extracted inline, verified to match model() logits exactly). A stub that looks like the method's home but
isn't will confuse a reader of the repo. Options: (a) implement it to match the notebook's inline extraction
(last-layer, mean-heads, attention-received + a rollout function), or (b) remove it. Recommend implementing to
match, so the repo has a single canonical attention wiring — but this is the supervisor's call; not changed
here.

## Investigation: Model 5's IG anomaly — attention-collapse mechanism REFUTED, reframed as a method-family split (16 Aug 2026)

Across the ladder IG barely moves (0.387 / 0.427 / 0.427 / 0.432) while at Model 5 FA jumps +0.170 to
0.654 and KS +0.131 to 0.610, IG only +0.005; FA-IG rank agreement falls from ~0.70 (CNNs) to 0.275.
Hypothesis tested: Model 5's attention rank collapse (last-layer exactly uniform; layers 1/3/4/5 uniform,
only 0/2 structured) smears the backward gradients the same way it flattened attention, so IG reads
flattened gradients while FA/KS (forward-only) are unaffected. Diagnostic only (existing checkpoints + saved
eval subset; no KernelSHAP; nothing changed).

REFUTED — the attention-collapse → smeared-gradients mechanism. Four independent lines of evidence:
  - Per-layer gradient magnitude (transformer, seed 0) is uniform ~4e-05 across ALL six encoder layers; the
    structured layers (0, 2) and the collapsed layers (1, 3, 4, 5) are indistinguishable. The direct test is
    negative — gradients do not collapse where attention does.
  - IG is NOT smeared: its per-region coefficient of variation is 1.374 vs FA's 0.885 — IG is MORE structured
    across regions than FA, not flatter. IG is ordered differently from FA, not flattened.
  - Sample-level: correlation between per-sample attention structure and per-sample IG-FA agreement is 0.078
    — essentially zero. The sample-level link the hypothesis predicts is absent.
  - The effect TRAVELS to patch-15 (FA CMI 0.559, IG CMI 0.402, FA-IG agreement 0.332; IG CoV 1.52), a
    transformer with 200 tokens and a different attention structure. A mechanism specific to patch-60's
    particular collapse cannot explain an effect that also appears at patch-15.
  - Also ruled out (alternative baseline explanation): IG completeness residual is 0.0003 (transformer) and
    0.0008 (CNN) — the path integral is exact, so the zero-baseline-on-patch-embedding concern does not hold.

REFRAMING (to write up instead of the refuted mechanism): IG's CMI is approximately MODEL-INVARIANT at ~0.4
across five models — 0.387 / 0.427 / 0.427 / 0.432 on the ladder and 0.402 on patch-15 — spanning ~100x in
parameters and two architecture families. FA and KS instead track the model, rising and then jumping at the
transformer. This is a PERTURBATION-vs-GRADIENT family split: FA and KS reward deletion impact — the same
quantity CMI's deletion curves measure — whereas IG measures path-integrated gradients, a different quantity
with no guarantee of aligning with discrete deletion impact. The split is already present on the CNNs
(cross-method agreement ~0.70, not 1.0) and WIDENS at the transformer.

HONEST BOUNDARY (preserve in the write-up): WHY the split widens specifically at the transformer is NOT
mechanistically established. The evidence supports the negative (collapse mechanism refuted) and the
reframing (IG model-invariant ~0.4; FA/KS rise; a method-family split); it does NOT support a causal story
for the widening. Do not claim one.

BOUND ON THE HEADLINE FINDING (belongs in the discussion): two of the three shared methods are aligned with
what CMI measures — FA is near-CIRCULAR with it by construction (single-region deletion impact vs cumulative
deletion in FA's own order), and KS rewards the same deletion impact. The one method that measures a
genuinely DIFFERENT quantity (IG, path-integrated gradients) does NOT rise across the ladder. So the CMI
trend Models 2->5 partly reflects WHAT CMI MEASURES (deletion impact), not only a property of the models.
This bounds the headline "faithfulness rises with complexity" claim: it is strongest for the deletion-based
methods and is not corroborated by the gradient-based one. State this explicitly.

## Decision: random-attribution baseline — the CMI floor (16 Aug 2026)

Notebook `sleep_edf/notebooks/baseline/00_random_baseline.ipynb` (prepared, not executed). PURPOSE: calibrate
the CMI scale. The ladder's CMI (0.387–0.654) had no floor reference, so a reader could not tell whether 0.39
is good — sharper given PES saturates at ~1.0 (its bar looks low but the study did not show how low). Random
attribution vectors over the 50 regions are pushed through the EXISTING deletion-curve + CMI machinery — no
attribution method is run; random vectors replace the attributions. Everything else matches the main runs so
the floor is directly comparable: same fixed N=500 eval subset (loaded, never regenerated), zero PM,
predicted-class target, same 5 MODEL seeds aggregated mean±std, same deletion settings (25 steps, MoRF/LeRF).

RANDOM-VECTOR SEED: **20260816** — explicit and INDEPENDENT of the model seeds. One `RandomState(20260816)` is
consumed in the fixed order rung(2,3,4,5) → model-seed(0–4) → sample(0–499), drawing a FRESH `rand(50)` per
(model, seed, sample) — fully reproducible, and per-sample variance not understated. Per-model runtime ~1.5 /
2 / 6 / 7 min (Models 2/3/4/5); ~16 min total. Saved to `random_floor_results.json`.

FLOOR VALUES: to be filled in once the user runs it. Expected: random CMI ≈ 0 (a random MoRF/LeRF ordering
gives ~symmetric DDS, so |DDS|≈0 and CMI≈0).

PES FRAMING CORRECTION (important, and it changes what the check reports): PES = fraction-positive minus
fraction-negative of the per-sample DDS, so its CHANCE value is ~0, NOT 0.5. A random ranking gets the
deletion direction right about half the time (fraction-positive ≈ 0.5), which corresponds to PES ≈ 0 (= 0.5 −
0.5). The "~0.5" intuition is the fraction-positive = (1 + PES)/2, reported alongside. So the check asks
whether random PES sits near 0: near 0 ⇒ PES DISCRIMINATES (the ladder's ~1.0 is real sign-consistency, not
saturation on garbage — reassuring); well away from 0 ⇒ the metric is positively biased even for meaningless
attributions (flag prominently). The notebook reports both random PES and the implied fraction-positive.

ORACLE VERIFICATION (done, zero new expensive compute): FeatureAblation(zero) was checked against the
single-region-deletion reliance (`region_reliance`, the quantity the concentration measure uses) on 6 samples
of Model 2 seed 0. **max|Δ| = 0.000e+00 — EXACT.** So "how close to the ORACLE" (the summary's framing for KS
and IG) is a DEMONSTRATED equivalence on Sleep-EDF, not merely asserted from the harness design. (This is the
single-region reliance, not the expensive cumulative/greedy oracle, which was out of scope.)

Summary notebook: a new §10 was APPENDED (floor table + margins above floor + attention-vs-floor, plus a
CMI-vs-rung plot with the random floor as a horizontal reference line). No other summary section changed; the
§4 plot was left untouched (§10 redraws its own). No model-specific notebook was touched.
