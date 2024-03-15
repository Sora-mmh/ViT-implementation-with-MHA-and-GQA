import logging

import torch
import torch.nn as nn
import math

logging.basicConfig(level=logging.INFO)


class MHAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 3072,
        num_heads: int = 12,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.pre_norm(x)
        x = x + self.attention(x_norm, x_norm, x_norm)[0]
        return x + self.MLP(self.norm(x))


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        attention_outputs: bool,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, logging.info(
            "The embedding dimension must be divisible by the number of attention heads"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_output = attention_outputs
        self.head_dim = self.embed_dim // self.num_heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_heads):
            self.heads.append(
                HeadAttention(self.embed_dim, self.head_dim, bias, dropout)
            )
        self.project_outputs = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        concatenated_attention_outputs = torch.cat(
            [attention_output_per_head for attention_output_per_head, _ in outputs],
            dim=-1,
        )
        projected_outputs = self.project_outputs(concatenated_attention_outputs)
        projected_outputs = self.dropout(projected_outputs)
        if not self.attention_output:
            return projected_outputs, None
        else:
            attention_weights = torch.cat(
                [
                    attention_weights_per_head
                    for _, attention_weights_per_head in outputs
                ],
                dim=1,
            )
            return projected_outputs, attention_weights


class HeadAttention(nn.Module):
    def __init__(
        self, embed_dim: int, head_dim: int, dropout: float, bias=True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.Q = nn.Linear(self.embed_dim, self.head_dim, bias=bias)
        self.K = nn.Linear(self.embed_dim, self.head_dim, bias=bias)
        self.V = nn.Linear(self.embed_dim, self.head_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.Q(x)
        key = self.K(x)
        value = self.V(x)
        # Scaled dot product
        q_kt = torch.matmul(query, key.transpose(-1, -2))
        scaled_q_kt = q_kt / math.sqrt(self.head_dim)
        scaled_q_kt_probs = nn.functional.softmax(scaled_q_kt, dim=-1)
        attention_weights = torch.matmul(scaled_q_kt_probs, value)
        return attention_weights, scaled_q_kt_probs
