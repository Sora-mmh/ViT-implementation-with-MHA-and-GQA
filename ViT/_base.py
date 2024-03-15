import torch
import torch.nn as nn
import math
import einops

from ..attention import MHAttentionBlock, GCAttentionBlock
from ..patches_embedder import PatchesEmbedder


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        hidden_dim: int = 3072,
        num_layers: int = 12,
        num_query_heads: int = 8,
        num_queries_per_group: int = 2,
        dropout: float = 0.1,
        num_classes: int = 8,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // self.patch_size) ** 2
        self.patches = PatchesEmbedder(
            channels=channels, embed_dim=embed_dim, patch_size=patch_size
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attention_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.attention_layers.append(
                GCAttentionBlock(
                    embed_dim,
                    hidden_dim,
                    num_query_heads,
                    num_queries_per_group,
                    dropout=dropout,
                    bias=bias,
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        x = self.patches(x)
        b, _, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
        x = self.norm(x)
        return self.head(x[:, 0])
