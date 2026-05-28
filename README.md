# XAI Faithfulness Evaluation Across Model Complexity for Time Series Classification

**Author**: Lucie Pasquier  
**Programme**: MSc, Imperial College London  
**Dataset**: ECG200 (UCR Time Series Archive)

---

## Project overview

This project evaluates the faithfulness of three explainability methods —
**TimeSHAP**, **Integrated Gradients**, and **attention weights** — across five
models of increasing complexity trained on the ECG200 binary ECG classification
task. The primary faithfulness metric is **CMI** (Consistency-Magnitude-Index;
Šimić, Veas & Sabol, 2025).

### Models

| # | Model | Library |
|---|-------|---------|
| 1 | Bandpass features + Logistic Regression | `scipy` / `scikit-learn` |
| 2 | Shallow 1D CNN | PyTorch |
| 3 | Medium 1D CNN | PyTorch |
| 4 | Deep 1D CNN | PyTorch |
| 5 | Transformer | PyTorch |

### XAI methods

| Method | Applicable models | Module |
|--------|-------------------|--------|
| TimeSHAP | All | `src/xai/timeshap.py` |
| Integrated Gradients | Models 2–5 | `src/xai/integrated_gradients.py` |
| Attention weights | Model 5 | `src/xai/attention.py` |

---

## Code organisation rule

> **Reusable logic lives in `.py` modules under `src/`.  
> Notebooks import from `src/` — they never reimplement logic.**

| Location | What goes here |
|----------|----------------|
| `src/data/` | Data loading, normalisation, split management |
| `src/models/` | Model class definitions and factory functions |
| `src/xai/` | XAI wrappers, CMI, deletion curves |
| `src/evaluation/` | Predictive-performance metrics (accuracy, AUC, …) |
| `src/plotting/` | Shared figure helpers |
| `notebooks/` | Experiment runs, exploration, thesis figures |
| `results/` | Figures (`.png`/`.pdf`), metric CSVs, model checkpoints |
| `tests/` | Unit and integration tests for `src/` |

If you find yourself writing a function or class definition in a notebook cell,
**stop and put it in `src/` instead**, then import it in the notebook.

---

## Directory structure

```
Thesis-Repo/
├── data/
│   ├── raw/                    # ECG200 source files — committed to git
│   │   ├── ECG200_TRAIN.ts
│   │   ├── ECG200_TEST.ts
│   │   └── ...
│   └── processed/              # .npy arrays — git-ignored, regenerated automatically
│
├── src/
│   ├── data/
│   │   └── preprocessing.py    # load_ecg200(), build_processed_data()
│   ├── models/
│   │   ├── linear_baseline.py  # Model 1: bandpass + logistic regression
│   │   ├── cnn.py              # Models 2–4: Shallow / Medium / Deep 1D CNN
│   │   └── transformer.py      # Model 5: Transformer
│   ├── xai/
│   │   ├── timeshap.py
│   │   ├── integrated_gradients.py
│   │   ├── attention.py
│   │   ├── cmi.py              # CMI faithfulness metric (Šimić et al. 2025)
│   │   └── deletion_curves.py
│   ├── evaluation/
│   │   └── metrics.py          # Accuracy, AUC, confusion matrix helpers
│   └── plotting/
│       └── helpers.py          # Shared figure style and plot functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_linear_baseline.ipynb
│   ├── 03_model_shallow_cnn.ipynb
│   ├── 04_model_medium_cnn.ipynb
│   ├── 05_model_deep_cnn.ipynb
│   ├── 06_model_transformer.ipynb
│   └── 07_xai_analysis.ipynb
│
├── results/
│   ├── figures/                # Saved .png / .pdf plots
│   ├── metrics/                # CMI scores, deletion-curve CSVs
│   └── checkpoints/            # Saved model weights (.pt files)
│
├── tests/
│   └── test_preprocessing.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting started

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) CPU-only PyTorch (smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Run the data sanity-check notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

The first call to `load_ecg200()` automatically runs `build_processed_data()`,
which reads the raw `.ts` files and writes processed `.npy` arrays to
`data/processed/`. Subsequent calls load directly from those files.

---

## Testing

```bash
pytest tests/ -v
```

---

## References

- Šimić, A., Veas, E., & Sabol, V. (2025). *A comprehensive analysis of
  perturbation methods in explainable AI feature attribution validation for
  neural time series classifiers.*
  [CMI reference implementation](https://github.com/perturbationeffect/cmi-am-validation-for-dl-ts-classifiers)

- Dau, H. A., et al. (2019). The UCR time series archive.
  *IEEE/CAA Journal of Automatica Sinica*, 6(6), 1293–1305.

- Bento, J., et al. (2021). TimeSHAP: Explaining recurrent models through
  sequence perturbations. *KDD 2021*.

- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep
  networks. *ICML 2017*.
