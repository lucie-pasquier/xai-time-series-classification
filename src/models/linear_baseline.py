"""
src/models/linear_baseline.py
──────────────────────────────────────────────────────────────────────────────
Model 1 (linear baseline): bandpass feature extraction + logistic regression.

Pipeline
    1. Bandpass-filter each ECG signal (scipy.signal.butter + filtfilt) into
       physiologically meaningful frequency bands.
    2. Extract per-band hand-crafted features: energy, mean absolute value, std,
       zero-crossing rate, peak-to-peak amplitude.
    3. Fit a logistic regression classifier (sklearn.linear_model.LogisticRegression)
       on the concatenated feature matrix.

This model is interpretable by design: the logistic regression coefficients
directly quantify the contribution of each feature. TimeSHAP and IG are not
well-suited to this pipeline; feature-weight attribution is used instead, giving
a per-timestep score via the inverse bandpass filter.

TODO — implement in notebook 02_model_linear_baseline.ipynb, then move
       the reusable classes/functions here.
"""
