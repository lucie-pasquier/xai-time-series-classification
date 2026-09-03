# Evaluating XAI Faithfulness Across Model Complexity for Time Series Classification

**Lucie Pasquier**
Supervised by Dr Felipe Tobar

A thesis submitted in fulfilment of the requirements for the degree of
**Master of Science in Artificial Intelligence Applications and Innovation**,
Imperial College London.

---

## About this repository

This repository holds the code written for the thesis above. It is shared for
**transparency and completeness**, so that the methods and experiments described
in the thesis can be inspected.

## What the work does

The thesis studies how the **faithfulness** of post-hoc explanations behaves as
model complexity increases. It evaluates attribution methods — Feature Ablation,
KernelSHAP, Integrated Gradients, and (for the transformer) attention weights —
against a **deletion-curve faithfulness metric**, CMI, the Consistency–Magnitude–
Index of Šimić, Veas & Sabol (2025). The evaluation runs across a ladder of five
sleep-staging models of increasing size, on EEG epochs from the Sleep-EDF
dataset, to ask whether more complex models yield more (or less) faithful
explanations.

## Layout

Two Python packages hold the work:

- **`harness/`** — the dataset-agnostic faithfulness apparatus: the attribution
  wrappers, the region grid and perturbation methods, the deletion curves, and
  the CMI metric.
- **`sleep_edf/`** — the Sleep-EDF experiments built on the harness: data
  loading, the five models, and the notebooks that produce the thesis's results
  and figures.

The notebooks under `sleep_edf/notebooks/` narrate the experiments; the reusable
logic lives in the two packages and is imported by the notebooks.

## Code attribution

This code adapts and uses third-party work. The notices below — together with the
per-file headers in the source — provide the attribution required by their
licences.

- **CMI metric and deletion-curve construction** — adapted from the reference
  implementation accompanying Šimić, A., Veas, E., & Sabol, V. (2025), *A
  comprehensive analysis of perturbation methods in explainable AI feature
  attribution validation for neural time series classifiers*
  ([repository](https://github.com/perturbationeffect/cmi-am-validation-for-dl-ts-classifiers)),
  used under the **Apache License 2.0**. The specific adaptations are documented
  in `harness/xai/cmi.py`, `harness/xai/deletion_curves.py`, and
  `harness/xai/perturbation.py`.
- **Attribution methods** — Feature Ablation, KernelSHAP and Integrated Gradients
  are computed with [Captum](https://captum.ai) (Kokhlikyan et al., 2020), used
  under the **BSD-3-Clause** licence.
