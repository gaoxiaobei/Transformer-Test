import torch
from torch import nn


class SequenceTransformerClassifier(nn.Module):
    """Serialize 28x28 image into 784 tokens and classify with Transformer encoder."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.seq_len = 28 * 28

        self.token_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.token_proj.weight)
        nn.init.zeros_(self.token_proj.bias)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        bsz = x.shape[0]
        x = x.view(bsz, self.seq_len, 1)
        x = self.token_proj(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.head(x)
        return logits
