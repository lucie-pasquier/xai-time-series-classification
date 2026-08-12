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