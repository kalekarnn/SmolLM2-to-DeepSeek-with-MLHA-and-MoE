import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Get batch size, sequence length, num heads, and head dimension
    batch_size, seq_len, num_heads, head_dim = xq.shape

    # Reshape the queries and keys to match the complex number format
    xq = xq.reshape(batch_size, seq_len, num_heads, -1, 2)
    xk = xk.reshape(batch_size, seq_len, xk.size(2), -1, 2)

    # Split real and imaginary parts
    xq_r, xq_i = xq.unbind(-1)
    xk_r, xk_i = xk.unbind(-1)

    # Expand freqs_cis for broadcasting
    # Original shape: [seq_len, dim/2, 2] -> [1, seq_len, 1, dim/2, 2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    cos, sin = freqs_cis.unbind(-1)

    # Apply rotary embeddings
    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos

    # Combine real and imaginary parts and reshape back
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1)

    xq_out = xq_out.reshape(batch_size, seq_len, num_heads, head_dim)
    xk_out = xk_out.reshape(batch_size, seq_len, xk.size(2), head_dim)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int = 576,
        max_position_embeddings: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        # Create the embedding matrix directly
        pos = torch.arange(self.max_position_embeddings).float()
        freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / dim))
        emb = pos[:, None] * freqs[None, :]  # [max_pos, dim/2]
        # Shape: [max_pos, dim/2, 2] where last dim is (cos, sin)
        self.register_buffer(
            "freqs_cis", torch.stack([torch.cos(emb), torch.sin(emb)], dim=-1)
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs_cis = self.freqs_cis[:seq_len].to(q.device)  # [seq_len, dim/2, 2]
        return apply_rotary_emb(q, k, freqs_cis)


class MoE(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert networks
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )

        # Router network
        self.router = nn.Linear(hidden_size, num_experts)

        # Load balancing loss
        self.aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Calculate routing logits
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]

        # Get top-k experts and their probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Process through selected experts
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]  # [batch_size, seq_len]
            expert_probs = top_k_probs[..., i]  # [batch_size, seq_len]

            # Process through each expert
            for j in range(self.num_experts):
                mask = expert_idx == j
                if mask.any():
                    expert_output = self.experts[j](x[mask])
                    output[mask] += expert_output * expert_probs[mask].unsqueeze(-1)

        # Calculate auxiliary loss for load balancing
        router_probs = router_probs.mean(dim=1)  # [batch_size, num_experts]
        expert_usage = router_probs.sum(dim=0)  # [num_experts]
        expert_usage = expert_usage / expert_usage.sum()

        # Calculate load balancing loss - use absolute value to ensure non-negative
        entropy = (expert_usage * torch.log(expert_usage + 1e-10)).sum()
        # Use absolute value to ensure non-negative loss
        self.aux_loss = torch.abs(entropy)

        return output


class DeepSeekAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        key_states = key_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        value_states = value_states.view(
            batch_size, seq_length, self.num_heads, self.head_dim
        )

        # Apply rotary embeddings
        query_states, key_states = self.rotary_emb(query_states, key_states, seq_length)

        # Reshape for attention computation
        query_states = query_states.transpose(
            1, 2
        )  # [batch, num_heads, seq_len, head_dim]
        key_states = key_states.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        value_states = value_states.transpose(
            1, 2
        )  # [batch, num_heads, seq_len, head_dim]

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=attn_weights.dtype)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class DeepSeekDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DeepSeekAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=config.num_attention_heads,
        )
        self.moe = MoE(
            hidden_size=self.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
        )
        self.input_layernorm = LlamaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DeepSeekModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepSeekDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class DeepSeekForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = DeepSeekModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights if configured
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module):
        std = self.config.initializer_range if hasattr(self, "config") else 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            # Add MoE auxiliary loss
            aux_loss = sum(layer.moe.aux_loss for layer in self.model.layers)

            # Total loss
            loss = ce_loss + 0.01 * aux_loss

            # For debugging, return a dictionary with both loss components
            return loss

        return logits
