import torch
import torch.nn as nn
import math
from typing import Optional

from .config import ModelConfig


class PatchEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_patches = config.num_patches

        self.projection = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)

        if self.config.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_len, config.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        return self.dropout(x)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        pe = self._create_sinusoidal_embeddings(config.seq_len, config.embed_dim)
        self.register_buffer("pos_embed", pe)

        self.dropout = nn.Dropout(config.dropout)

    def _create_sinusoidal_embeddings(
        self, seq_len: int, embed_dim: int
    ) -> torch.Tensor:
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(1, seq_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        return self.dropout(x)
