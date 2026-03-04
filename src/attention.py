import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import ModelConfig


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_dim = config.qk_rope_dim
        self.qk_nope_dim = config.qk_rope_dim
        self.v_head_dim = config.head_dim
        self.q_head_dim = self.qk_rope_dim + self.qk_nope_dim

        self.kv_down = nn.Linear(
            config.embed_dim, config.kv_lora_rank + config.qk_rope_dim, bias=False
        )

        self.kv_up = nn.Linear(
            config.kv_lora_rank,
            config.num_heads * (self.v_head_dim + self.qk_nope_dim),
            bias=False,
        )

        self.q_proj = nn.Linear(
            config.embed_dim, config.num_heads * self.q_head_dim, bias=False
        )

        self.out_proj = nn.Linear(
            config.num_heads * self.v_head_dim, config.embed_dim, bias=False
        )

        self.scale = self.q_head_dim**-0.5
        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        compressed = self.kv_down(x)
        kv_compressed, k_rope = compressed.split(
            [self.kv_lora_rank, self.qk_rope_dim], dim=-1
        )

        kv_expanded = self.kv_up(kv_compressed)
        kv_expanded = kv_expanded.view(
            batch_size, seq_len, self.num_heads, self.v_head_dim + self.qk_nope_dim
        ).transpose(1, 2)

        v, k_nope = kv_expanded.split([self.v_head_dim, self.qk_nope_dim], dim=-1)

        k_rope = k_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        k = torch.cat([k_rope, k_nope], dim=-1)

        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        q = q.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.num_heads * self.v_head_dim)

        return self.out_proj(out)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)

        return self.out_proj(out)
