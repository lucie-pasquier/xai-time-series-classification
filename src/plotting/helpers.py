"""
src/plotting/helpers.py
──────────────────────────────────────────────────────────────────────────────
Shared plotting utilities used across experiment notebooks.

Keeping plot code here (rather than inline in notebooks) ensures that all
figures use a consistent style and that the same plots can be regenerated
from any notebook without copy-pasting.

Planned functions
    set_thesis_style()
        Apply rcParams for publication-quality figures (font sizes, DPI,
        colour palette consistent with the thesis template).

    plot_attribution_heatmap(X, attributions, y, ax)
        Overlay per-timestep attribution scores as a heatmap on the raw signal.
        Used for TimeSHAP, IG, and attention visualisations.

    plot_deletion_curve(fractions, confidences_dict, ax)
        Plot multiple deletion curves (one per XAI method) on the same axes.

    plot_cmi_comparison(results_df, ax)
        Bar chart comparing CMI scores across models and XAI methods.

TODO — implement as plots are needed in experiment notebooks.
"""
