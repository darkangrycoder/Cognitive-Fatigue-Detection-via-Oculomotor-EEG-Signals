import torch
import torch.nn as nn

D_SHARED = 128
D_LATENT = 64


class SharedEncoder(nn.Module):
    """
    Domain-general fatigue encoder.
    Architecture: Linear -> LayerNorm -> GELU -> Dropout (x2) -> 64-dim latent
    This backbone is shared between the GazeBase regression model and the
    SEED-VIG vigilance classifier, enabling cross-modal weight transfer.
    """
    def __init__(self, in_dim, d_shared=D_SHARED, d_latent=D_LATENT, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_shared),
            nn.LayerNorm(d_shared), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_shared, d_shared),
            nn.LayerNorm(d_shared), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_shared, d_latent),
            nn.LayerNorm(d_latent),
        )

    def forward(self, x):
        return self.net(x)


class GazeModel(nn.Module):
    """
    Fatigue regression model pretrained on GazeBase (881 subjects, 1000 Hz).
    Input: 16 oculomotor features
    Output: continuous fatigue z-score
    """
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = SharedEncoder(in_dim)
        self.head = nn.Sequential(
            nn.Linear(D_LATENT, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(-1)