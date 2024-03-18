import logging

import torch
import torch.nn as nn
import math
import einops


class GQAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 3072,
        num_query_heads: int = 8,
        num_queries_per_group: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
        attention_outputs: bool = True,
    ) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.attention = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_query_heads=num_query_heads,
            num_queries_per_group=num_queries_per_group,
            dropout=dropout,
            bias=bias,
            attention_outputs=attention_outputs,
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
        x = x + self.attention(x_norm)[0]
        return x + self.MLP(self.norm(x))


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int,
        num_queries_per_group: int,
        dropout: float,
        bias: bool,
        attention_outputs: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_query_heads == 0, logging.info(
            "The embedding dimension must be divisible by the number of query heads"
        )
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.query_head_dim = self.embed_dim // self.num_query_heads
        self.attention_outputs = attention_outputs
        assert self.num_query_heads % num_queries_per_group == 0, logging.info(
            "The number of query heads must be divisible by the number of grouped queries"
        )
        self.num_kv_heads = self.num_query_heads // num_queries_per_group
        self.branchs = nn.ModuleList([])
        for _ in range(self.num_kv_heads):
            self.branchs.append(
                GroupedQueryAttentionBranch(
                    self.embed_dim,
                    self.query_head_dim,
                    num_queries_per_group,
                    dropout,
                    bias,
                )
            )
        self.project_outputs = nn.Linear(
            self.query_head_dim * self.num_kv_heads, self.embed_dim
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs = [branch(x) for branch in self.branchs]
        concatenated_attention_outputs = torch.cat(
            [attention_output_per_head for attention_output_per_head, _ in outputs],
            dim=-1,
        )
        projected_outputs = self.project_outputs(concatenated_attention_outputs)
        projected_outputs = self.dropout(projected_outputs)
        if not self.attention_outputs:
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


class GroupedQueryAttentionBranch(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        num_queries_per_group: int,
        dropout: float,
        bias=True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_queries_per_group = num_queries_per_group
        self.bias = bias
        self.Qs = nn.ModuleList(
            [
                nn.Linear(self.embed_dim, self.head_dim, bias=bias)
                for _ in range(self.num_queries_per_group)
            ]
        )
        self.K = nn.Linear(self.embed_dim, self.head_dim, bias=bias)
        self.V = nn.Linear(self.embed_dim, self.head_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        queries = torch.cat([query(x) for query in self.Qs], dim=0)
        key = self.K(x)
        value = self.V(x)
        # Scaled dot product (SDP)
        q_kts = torch.cat(
            [torch.matmul(query, key.transpose(-1, -2)) for query in queries], dim=0
        )
        scaled_q_kts = torch.cat(
            [(q_kt / math.sqrt(self.head_dim)).unsqueeze(0) for q_kt in q_kts], dim=0
        )
        scaled_q_kts_probs = torch.cat(
            [
                nn.functional.softmax(scaled_q_kt, dim=-1).unsqueeze(0)
                for scaled_q_kt in scaled_q_kts
            ],
            dim=0,
        )
        attention_weights = torch.cat(
            [
                torch.matmul(scaled_q_kt_probs, value)
                for scaled_q_kt_probs in scaled_q_kts_probs
            ],
            dim=0,
        )
        # convert attention_weights shape : num_queries_per_head, patches, head_dim --> num_queries_per_head * patches, head_dim
        attention_weights = einops.rearrange(
            attention_weights, "q p h -> p (q h)"
        )  # num_queries, num_patches, head_dim
        _, merged_dim = attention_weights.shape
        attention_weights = nn.Linear(merged_dim, self.head_dim, bias=self.bias)(
            attention_weights
        ).unsqueeze(0)
        return attention_weights, scaled_q_kts_probs
