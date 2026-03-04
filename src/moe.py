import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .config import ModelConfig


class Expert(nn.Module):
    def __init__(self, embed_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token

        self.gate = nn.Linear(config.embed_dim, config.num_experts, bias=False)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)

        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_token, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)

        aux_loss = self._compute_aux_loss(router_logits, selected_experts)

        return routing_weights, selected_experts, aux_loss

    def _compute_aux_loss(
        self, router_logits: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = router_logits.shape[0]

        expert_mask = F.one_hot(selected_experts, self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=(0, 1))
        f = tokens_per_expert / (num_tokens * self.num_experts_per_token)

        routing_probs = F.softmax(router_logits, dim=-1)
        P = routing_probs.mean(dim=0)

        return self.num_experts * torch.sum(f * P)


class MixtureOfExperts(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.router = Router(config)

        if config.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [
                    Expert(config.embed_dim, config.intermediate_dim)
                    for _ in range(config.num_shared_experts)
                ]
            )
        else:
            self.shared_experts = None

        self.experts = nn.ModuleList(
            [
                Expert(config.embed_dim, config.intermediate_dim)
                for _ in range(config.num_experts)
            ]
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, embed_dim = hidden_states.shape

        hidden_flat = hidden_states.view(-1, embed_dim)

        if self.shared_experts is not None:
            shared_output = sum(expert(hidden_flat) for expert in self.shared_experts)
        else:
            shared_output = torch.zeros_like(hidden_flat)

        routing_weights, selected_experts, aux_loss = self.router(hidden_flat)

        routed_output = self._route_tokens(
            hidden_flat, routing_weights, selected_experts
        )

        final_output = shared_output + routed_output
        final_output = self.dropout(final_output)
        final_output = final_output.view(batch_size, seq_len, embed_dim)

        return final_output, aux_loss

    def _route_tokens(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, embed_dim = hidden_states.shape
        k = self.config.num_experts_per_token
        device = hidden_states.device

        output = torch.zeros_like(hidden_states)

        flat_experts = selected_experts.view(-1)
        flat_weights = routing_weights.view(-1)

        token_indices = torch.arange(num_tokens, device=device)
        token_indices = token_indices.unsqueeze(1).expand(-1, k).reshape(-1)

        for expert_idx in range(self.config.num_experts):
            expert = self.experts[expert_idx]

            mask = flat_experts == expert_idx
            if not mask.any():
                continue

            selected_positions = mask.nonzero(as_tuple=True)[0]
            expert_token_indices = token_indices[selected_positions]
            expert_weights = flat_weights[selected_positions]

            expert_input = hidden_states[expert_token_indices]
            expert_output = expert(expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)

            output.index_add_(0, expert_token_indices, weighted_output)

        return output


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.embed_dim, config.intermediate_dim, bias=False
        )
        self.up_proj = nn.Linear(config.embed_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(
            config.intermediate_dim, config.embed_dim, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.dropout(out)
