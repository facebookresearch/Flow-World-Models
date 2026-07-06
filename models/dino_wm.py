# Ported from https://github.com/gaoyuezhou/dino_wm/blob/main/models/vit.py
# with action + proprio conditioning added (DINO-WM concat_dim=1 style)

import torch
from torch import nn
from einops import rearrange

from .proprio import ProprioceptiveEmbedding


def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_frames, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        # Block-causal mask: each frame's patches attend to all previous frames
        self.register_buffer(
            "causal_mask", generate_mask_matrix(num_patches, num_frames)
        )

    def forward(self, x):
        B, T, C = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, mlp_dim, num_patches, num_frames, dropout=0.0
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            num_patches,
                            num_frames,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_context_latents,
        num_target_latents,
        dim,
        depth,
        heads,
        mlp_dim,
        input_dim=None,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        # Action + proprio conditioning
        action_dim=0,
        proprio_dim=0,
        action_emb_dim=10,
        proprio_emb_dim=10,
        num_action_repeat=1,
        num_proprio_repeat=1,
    ):
        super().__init__()
        assert pool in {"cls", "mean"}

        self.num_context_latents = num_context_latents
        self.num_target_latents = num_target_latents
        self.num_patches = num_patches
        self.input_dim = input_dim if input_dim is not None else dim
        self.dim = dim
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.action_emb_dim = action_emb_dim * num_action_repeat
        self.proprio_emb_dim = proprio_emb_dim * num_proprio_repeat
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat

        # Total dimension per patch after concatenating visual + proprio + action
        self.total_emb_dim = self.input_dim + self.proprio_emb_dim + self.action_emb_dim

        # Action and proprio encoders (shared Conv1d architecture)
        if action_dim > 0:
            self.action_encoder = ProprioceptiveEmbedding(
                in_chans=action_dim, emb_dim=action_emb_dim
            )
        if proprio_dim > 0:
            self.proprio_encoder = ProprioceptiveEmbedding(
                in_chans=proprio_dim, emb_dim=proprio_emb_dim
            )

        # Projections: from concatenated space to transformer dim and back
        if self.total_emb_dim != dim:
            self.input_proj = nn.Linear(self.total_emb_dim, dim)
            self.output_proj = nn.Linear(dim, self.input_dim)
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_context_latents * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            num_patches,
            num_context_latents,
            dropout,
        )
        self.pool = pool

    def _encode_and_concat(self, visual, actions=None, states=None):
        """Encode actions/proprio and concat with visual features (DINO-WM concat_dim=1).

        Args:
            visual: [B, T, num_patches, input_dim]
            actions: [B, T_total, action_dim] or None
            states: [B, T_total, proprio_dim] or None

        Returns:
            z: [B, T, num_patches, total_emb_dim]
        """
        B, T, P, D = visual.shape
        parts = [visual]

        if self.action_dim > 0:
            if actions is not None:
                act = actions[:, :T]
                act_emb = self.action_encoder(act)
                act_emb = act_emb.unsqueeze(2).expand(-1, -1, P, -1)
                act_emb = act_emb.repeat(1, 1, 1, self.num_action_repeat)
            else:
                act_emb = torch.zeros(
                    B, T, P, self.action_emb_dim, device=visual.device
                )
            parts.append(act_emb)

        if self.proprio_dim > 0:
            if states is not None:
                proprio = states[:, :T]
                proprio_emb = self.proprio_encoder(proprio)
                proprio_emb = proprio_emb.unsqueeze(2).expand(-1, -1, P, -1)
                proprio_emb = proprio_emb.repeat(1, 1, 1, self.num_proprio_repeat)
            else:
                proprio_emb = torch.zeros(
                    B, T, P, self.proprio_emb_dim, device=visual.device
                )
            parts.append(proprio_emb)

        return torch.cat(parts, dim=-1)  # [B, T, P, total_emb_dim]

    def forward(self, context_latents, actions=None, states=None):
        """
        Args:
            context_latents: [B, D, T, H, W] — visual latents in channel-first format
            actions: [B, T_total, action_dim] or None
            states: [B, T_total, proprio_dim] or None

        Returns:
            dict with "predicted_latents": [B, D, T, H, W]
        """
        b, d, t, h, w = context_latents.shape
        # Reshape to [B, T, num_patches, D]
        visual = rearrange(context_latents, "b d t h w -> b t (h w) d")

        # Encode and concatenate action/proprio
        z = self._encode_and_concat(visual, actions, states)

        # Flatten time and patches, project, add positional encoding
        z = rearrange(z, "b t p d -> b (t p) d")
        z = self.input_proj(z)
        z = z + self.pos_embedding[:, : z.shape[1]]
        z = self.dropout(z)

        # Causal transformer
        z = self.transformer(z)

        # Project back to visual dim only (strip action/proprio)
        z = self.output_proj(z)
        z = rearrange(z, "b (t h w) d -> b d t h w", t=t, h=h, w=w)
        return {"predicted_latents": z}

    def rollout(self, context_latents, actions=None, states=None):
        """Autoregressive rollout with GT actions (no teacher forcing on visual latents).

        Args:
            context_latents: [B, D, T_ctx, H, W]
            actions: [B, T_total, action_dim] — GT actions for all frames (context + target)
            states: [B, T_total, proprio_dim] — GT states (only context frames are reliable;
                    predicted frames get zero-padded)

        Returns:
            dict with "predicted_latents": [B, D, T_tgt, H, W]
        """
        num_steps = self.num_target_latents
        ctx = context_latents  # grows as we append predictions
        predicted = []

        for step in range(num_steps):
            # Sliding window: take last num_context_latents frames
            window = ctx[:, :, -self.num_context_latents :]

            # Actions for the window frames
            # Frame indices in the growing sequence: [step, ..., step + num_ctx - 1]
            act_window = None
            if actions is not None and self.action_dim > 0:
                act_window = actions[:, step : step + self.num_context_latents]

            # States for the window frames (zero-pad for predicted frames)
            state_window = None
            if states is not None and self.proprio_dim > 0:
                num_ctx = self.num_context_latents
                t_total = states.shape[1]
                state_window = torch.zeros(
                    ctx.shape[0],
                    num_ctx,
                    states.shape[2],
                    device=ctx.device,
                    dtype=states.dtype,
                )
                for i in range(num_ctx):
                    frame_idx = step + i
                    if frame_idx < t_total:
                        state_window[:, i] = states[:, frame_idx]

            out = self.forward(window, actions=act_window, states=state_window)
            next_latent = out["predicted_latents"][:, :, -1:]
            predicted.append(next_latent)
            ctx = torch.cat([ctx, next_latent], dim=2)

        predicted = torch.cat(predicted, dim=2)
        return {"predicted_latents": predicted}
