"""
sleep_edf/training.py
──────────────────────────────────────────────────────────────────────────────
Reusable "timed, progress-tracked, all-seeds" training driver — the single source
of truth used IDENTICALLY by every model on the Sleep-EDF complexity ladder
(linear baseline, three CNNs, transformer).

Why this exists
    Training is launched once and left to run unattended to completion. This driver
    provides one constant interface for that across all models:
      1. It TIMES the first freshly-trained seed and PRINTS a total-time estimate,
         then CONTINUES automatically (it is NOT a prompt/gate — launch and walk away).
      2. It trains all remaining seeds to completion with NO per-seed intervention.
      3. It shows a tqdm progress bar: an OUTER bar over seeds, and — for models with
         neural epochs — an optional INNER bar over epochs (via a `tick` callback the
         train function may call). Logistic regression has no epochs, so it simply
         never calls `tick` and only the outer seed bar shows.
      4. It SAVES each seed's results to disk AS THAT SEED FINISHES (not only at the
         end), so an interruption never loses completed seeds and a re-run can resume.
      5. After all seeds it aggregates, prints a per-seed + mean±std summary, and
         saves the aggregate.

The one contract every model implements
    train_one_seed(seed: int, tick=None) -> dict
        Train ONE seed and return a JSON-friendly dict of that seed's results
        (metrics, confusion, etc.). If the model trains over epochs, call
        `tick(done, total)` once per epoch to drive the inner progress bar; if it
        has no epochs (e.g. logistic regression), just ignore `tick`.

This module is model-agnostic: it imports no sklearn/torch, only tqdm/numpy/stdlib.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


# ── JSON helpers (make numpy types/arrays serialisable) ───────────────────────

def _jsonable(obj):
    """Recursively convert numpy scalars/arrays to plain Python for json.dump."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonable(data), f, indent=1)


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ── The inner (per-seed) epoch progress bar ───────────────────────────────────

def _make_tick(desc: str):
    """Return (tick, state). `tick(done, total)` lazily creates and drives an inner
    tqdm bar; if never called (no epochs), no inner bar is ever shown."""
    state = {"bar": None}

    def tick(done, total):
        if state["bar"] is None:
            state["bar"] = tqdm(total=int(total), desc=desc, leave=False)
        state["bar"].n = int(done)
        state["bar"].refresh()

    return tick, state


# ── Aggregation / summary ─────────────────────────────────────────────────────

def _aggregate(model_name, seeds, per_seed, summary_keys):
    agg = {"model": model_name, "seeds": list(seeds),
           "per_seed": {str(s): per_seed[s] for s in seeds}, "summary": {}}
    for key in (summary_keys or []):
        vals = [per_seed[s][key] for s in seeds if s in per_seed and key in per_seed[s]]
        if vals:
            agg["summary"][key] = {"mean": float(np.mean(vals)),
                                   "std": float(np.std(vals)),
                                   "values": [float(v) for v in vals]}
    total_secs = sum(per_seed[s].get("seconds", 0.0) for s in seeds if s in per_seed)
    agg["total_seconds"] = float(total_secs)
    return agg


def _print_summary(model_name, seeds, per_seed, summary_keys):
    keys = summary_keys or []
    print(f"\n================ {model_name}: all-seeds summary ================")
    header = f"{'seed':<6}" + "".join(f"{k:>18}" for k in keys) + f"{'secs':>9}"
    print(header); print("-" * len(header))
    for s in seeds:
        if s not in per_seed:
            continue
        row = f"{s:<6}" + "".join(f"{per_seed[s].get(k, float('nan')):>18.4f}" for k in keys)
        row += f"{per_seed[s].get('seconds', float('nan')):>9.1f}"
        print(row)
    print("-" * len(header))
    for stat in ("mean", "std"):
        cells = []
        for k in keys:
            vals = [per_seed[s][k] for s in seeds if s in per_seed and k in per_seed[s]]
            v = (np.mean(vals) if stat == "mean" else np.std(vals)) if vals else float("nan")
            cells.append(f"{v:>18.4f}")
        print(f"{stat:<6}" + "".join(cells))


# ── The public driver ─────────────────────────────────────────────────────────

def run_all_seeds(train_one_seed, seeds, out_dir, model_name, *,
                  summary_keys=None, resume=False, verbose=True):
    """Time seed 1, estimate total, then train all seeds to completion — unattended.

    Parameters
    ----------
    train_one_seed : callable(seed:int, tick=None) -> dict
        Trains ONE seed and returns a JSON-friendly results dict. For neural models,
        call `tick(done, total)` once per epoch to drive the inner epoch bar; models
        without epochs simply ignore `tick`.
    seeds : list[int]
        Seeds to train, in order. The first one is timed to form the estimate.
    out_dir : str | Path
        Directory for per-seed and aggregate result files (e.g.
        sleep_edf/results/metrics/). Created if missing.
    model_name : str
        Prefix for output files: ``{model_name}_seed{seed}.json`` per seed and
        ``{model_name}_aggregate.json`` for the aggregate.
    summary_keys : list[str], optional
        Scalar keys in each seed's dict to aggregate (mean±std) and tabulate.
    resume : bool, default False
        If True, seeds whose ``{model_name}_seed{seed}.json`` already exists are
        loaded from disk and NOT retrained (resume an interrupted run / skip done).
    verbose : bool, default True
        Print the per-seed + mean±std summary at the end.

    Returns
    -------
    dict : the aggregate (also written to ``{model_name}_aggregate.json``).

    Notes
    -----
    Each seed is saved to disk the moment it finishes, so an interruption keeps all
    completed seeds. The time estimate is printed after the first FRESHLY trained
    seed and execution then continues automatically — this is not a prompt.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)
    per_seed = {}
    first_trained_dt = None

    outer = tqdm(list(enumerate(seeds)), total=len(seeds), desc=f"{model_name}: seeds")
    for i, seed in outer:
        outer.set_description(f"{model_name}: seed {i + 1}/{len(seeds)} (seed={seed})")
        seed_path = out_dir / f"{model_name}_seed{seed}.json"

        if resume and seed_path.exists():
            per_seed[seed] = _load_json(seed_path)
            outer.set_postfix_str("resumed from disk")
            continue

        tick, state = _make_tick(f"  seed {seed}: epochs")
        t0 = time.perf_counter()
        result = train_one_seed(seed, tick=tick)
        dt = time.perf_counter() - t0
        if state["bar"] is not None:
            state["bar"].close()

        result = dict(result)
        result["seed"] = int(seed)
        result["seconds"] = float(dt)
        _save_json(seed_path, result)          # <-- saved AS THIS SEED FINISHES
        per_seed[seed] = result
        outer.set_postfix_str(f"{dt:.1f}s -> {seed_path.name}")

        if first_trained_dt is None:           # estimate after the FIRST trained seed
            first_trained_dt = dt
            est = dt * len(seeds)
            outer.write(
                f"[{model_name}] seed {seed} trained in {dt:.1f}s  ->  estimated "
                f"~{est/60:.1f} min for all {len(seeds)} seeds. Training continues "
                f"automatically (no input needed)."
            )
    outer.close()

    aggregate = _aggregate(model_name, seeds, per_seed, summary_keys)
    _save_json(out_dir / f"{model_name}_aggregate.json", aggregate)
    if verbose:
        _print_summary(model_name, seeds, per_seed, summary_keys)
        print(f"\nsaved: per-seed {model_name}_seed*.json  +  {model_name}_aggregate.json  "
              f"in {out_dir}")
    return aggregate
