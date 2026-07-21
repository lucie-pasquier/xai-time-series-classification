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