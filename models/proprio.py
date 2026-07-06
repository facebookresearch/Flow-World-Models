import torch.nn as nn


class ProprioceptiveEmbedding(nn.Module):
    """Shared encoder for actions and proprioceptive state.

    Ported from dino_wm/models/proprio.py. Uses a 1D convolution with kernel_size=1,
    which is equivalent to a per-timestep linear projection.
    """

    def __init__(self, in_chans, emb_dim, tubelet_size=1):
        super().__init__()
        self.in_chans = in_chans
        self.emb_dim = emb_dim
        self.patch_embed = nn.Conv1d(
            in_chans, emb_dim, kernel_size=tubelet_size, stride=tubelet_size
        )

    def forward(self, x):
        # x: [B, T, in_chans]
        x = x.permute(0, 2, 1)  # [B, in_chans, T]
        x = self.patch_embed(x)  # [B, emb_dim, T]
        x = x.permute(0, 2, 1)  # [B, T, emb_dim]
        return x
