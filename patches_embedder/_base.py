import torch.nn as nn


class PatchesEmbedder(nn.Module):
    def __init__(
        self, channels: int = 3, embed_dim: int = 768, patch_size: int = 16
    ) -> None:
        super().__init__()
        self.patch = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        return self.patch(x).flatten(2).transpose(1, 2)
