import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Union

from .config import ModelConfig
from .embeddings import PatchEmbedding, PositionalEmbedding
from .normalization import RMSNorm, LayerNorm
from .attention import MultiHeadLatentAttention, MultiHeadSelfAttention
from .moe import MixtureOfExperts, FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        norm_class = RMSNorm if config.use_rms_norm else LayerNorm
        self.norm1 = norm_class(config.embed_dim, config.norm_eps)
        self.norm2 = norm_class(config.embed_dim, config.norm_eps)

        if config.use_mla:
            self.attention = MultiHeadLatentAttention(config)
        else:
            self.attention = MultiHeadSelfAttention(config)

        if config.use_moe:
            self.ffn = MixtureOfExperts(config)
        else:
            self.ffn = FeedForward(config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm2(x)

        aux_loss = None
        if self.config.use_moe:
            x, aux_loss = self.ffn(x)
        else:
            x = self.ffn(x)

        x = residual + x

        return x, aux_loss


class VisionTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbedding(config)

        if config.use_pos_embed:
            self.pos_embed = PositionalEmbedding(config)
        else:
            self.pos_embed = None

        self.blocks = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )

        norm_class = RMSNorm if config.use_rms_norm else LayerNorm
        self.norm = norm_class(config.embed_dim, config.norm_eps)

        self.head = nn.Linear(config.embed_dim, config.num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = self.pos_embed(x)

        total_aux_loss: Optional[torch.Tensor] = None
        aux_count = 0

        for block in self.blocks:
            x, aux_loss = block(x)
            if aux_loss is not None:
                if total_aux_loss is None:
                    total_aux_loss = aux_loss
                else:
                    total_aux_loss = total_aux_loss + aux_loss
                aux_count += 1

        x = self.norm(x)

        if self.config.use_cls_token:
            features = x[:, 0]
        else:
            features = x.mean(dim=1)

        if return_features:
            return features, total_aux_loss

        logits = self.head(features)

        if total_aux_loss is not None and aux_count > 0:
            avg_aux_loss = total_aux_loss / aux_count
        else:
            avg_aux_loss = None

        return logits, avg_aux_loss

    def get_num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.patch_embed, "projection"):
            n_params -= self.patch_embed.projection.weight.numel()
        return n_params
