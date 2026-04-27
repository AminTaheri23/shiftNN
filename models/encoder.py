"""Word-bitmap encoder: (B, 7, K*5) bitmap -> (B, d_model) embedding.

Pipeline:
  1. Reshape the word bitmap into K letter-cells of 7x5 = 35 pixels each.
  2. Per-cell linear embedding (every cell sees its own 35-bit pattern).
  3. Add learnable positional embedding per cell-position.
  4. Optional small intra-word transformer (cells attend to each other).
  5. Concat + project to a single d_model word embedding.

Cell-wise structure is preserved through step 4 so probes can recover
per-segment features (the LCD-font interpretability story).
"""

import torch
import torch.nn as nn

from pixenc.dotmatrix_font import DM_ROWS, DM_COLS, DM_BITS


class WordBitmapEncoder(nn.Module):
    def __init__(self, k=12, d_model=256, cell_dim=128, n_intra_layers=0, n_heads=4,
                 dropout=0.0):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.cell_dim = cell_dim

        # Per-cell linear projection from 35 raw pixels.
        self.cell_embed = nn.Linear(DM_BITS, cell_dim)
        self.cell_pos = nn.Parameter(torch.zeros(1, k, cell_dim))
        nn.init.trunc_normal_(self.cell_pos, std=0.02)

        if n_intra_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=cell_dim, nhead=n_heads,
                dim_feedforward=cell_dim * 4,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.intra = nn.TransformerEncoder(layer, num_layers=n_intra_layers)
        else:
            self.intra = None

        self.word_proj = nn.Linear(cell_dim * k, d_model)

    def forward(self, bitmap):
        """
        bitmap: (B, 7, K*5) float in [0, 1]
        returns: (B, d_model)
        """
        B, H, W = bitmap.shape
        assert H == DM_ROWS, f"expected {DM_ROWS} rows, got {H}"
        assert W == self.k * DM_COLS, f"expected {self.k * DM_COLS} cols, got {W}"

        # Reshape into cells: (B, 7, K, 5) -> (B, K, 7, 5) -> (B, K, 35)
        cells = bitmap.reshape(B, DM_ROWS, self.k, DM_COLS)
        cells = cells.permute(0, 2, 1, 3).reshape(B, self.k, DM_BITS)

        x = self.cell_embed(cells)  # (B, K, cell_dim)
        x = x + self.cell_pos
        if self.intra is not None:
            x = self.intra(x)
        return self.word_proj(x.flatten(1))  # (B, d_model)
