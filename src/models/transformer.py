"""
src/models/transformer.py
──────────────────────────────────────────────────────────────────────────────
Model 5: a Transformer encoder for ECG200 — the top of the complexity ladder.

How it differs from the CNNs (Models 2–4)
    A CNN detects LOCAL patterns with small filters sliding along the signal. A
    Transformer uses SELF-ATTENTION: every position can directly attend to every
    other position, so it can relate distant timesteps in one step rather than
    building locality up through depth. That flexibility is powerful but
    data-hungry — on 80 training samples it is the model most at risk of
    instability/overfitting, which is why Model 5 is multi-seed from the start.

Patching / region-grid compatibility (deliberate choice)
    We use PER-TIMESTEP embedding (patch_size = 1): a 1×1 conv maps each of the 96
    timesteps to a d_model token, so there are 96 tokens, one per timestep. Reason:
    the faithfulness harness attributes onto the layer-1 RegionGrid, which groups
    RAW timesteps into ~10 regions. Per-timestep tokens map EXACTLY onto that grid
    (each region = a contiguous block of tokens via grid.labels), so:
      - FeatureAblation / KernelSHAP / IG already operate on the raw input and pool
        to regions — internal tokenisation is invisible to them regardless — but
      - the eventual ATTENTION analysis reads per-token weights, and per-timestep
        tokens pool to regions with NO boundary misalignment.
    A larger patch_size (e.g. 8 or 12) would cut the token count and compute, but
    fixed-size patches do NOT align with the near-equal (array_split) region grid
    on 96 timesteps (96 doesn't divide into 10 equal parts), so patch boundaries
    would straddle region boundaries and muddy the attention→region pooling. We
    therefore chose patch_size=1 to keep the transformer grid-comparable to the
    CNNs; `patch_size` is a parameter if a patched variant is wanted later (with
    that misalignment flagged).

Input contract: forward takes (n, in_channels, T) — identical to the CNNs, so the
    shared train_cnn recipe and torch_predict_proba harness wrapper work unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    """Transformer encoder classifier for 1-D signals.

    Parameters
    ----------
    input_length : int
        Number of timesteps (read from the data; e.g. 96 for ECG200).
    in_channels : int
        Input channels (1 for ECG200; read from the data).
    n_classes : int
        Output classes (read from the data — no binary assumption).
    d_model, nhead, num_layers, dim_feedforward, dropout : transformer hyperparams.
    patch_size : int
        Timesteps per token (1 = per-timestep embedding; see module docstring).
    """

    def __init__(self, input_length, in_channels=1, n_classes=2,
                 d_model=128, nhead=8, num_layers=6, dim_feedforward=512,
                 dropout=0.1, patch_size=1):
        super().__init__()
        if input_length % patch_size != 0:
            raise ValueError(f"input_length {input_length} not divisible by patch_size {patch_size}")
        self.patch_size = patch_size
        n_tokens = input_length // patch_size
        # Patch/linear embedding: 1x1 conv when patch_size==1 (per-timestep linear map).
        self.embed = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, in_channels, T)
        z = self.embed(x)                 # (n, d_model, n_tokens)
        z = z.transpose(1, 2)             # (n, n_tokens, d_model)
        z = z + self.pos
        z = self.encoder(z)               # (n, n_tokens, d_model)
        z = self.norm(z).mean(dim=1)      # mean-pool tokens -> (n, d_model)
        return self.head(z)               # (n, n_classes) logits


def build_transformer(input_length=96, in_channels=1, n_classes=2,
                      d_model=128, nhead=8, num_layers=6, dim_feedforward=512,
                      dropout=0.1, patch_size=1) -> TimeSeriesTransformer:
    """Build the Model-5 transformer (defaults ~1.2M params; patch_size=1)."""
    return TimeSeriesTransformer(
        input_length, in_channels, n_classes, d_model, nhead, num_layers,
        dim_feedforward, dropout, patch_size,
    )
