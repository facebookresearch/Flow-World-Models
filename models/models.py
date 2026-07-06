import logging
import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoImageProcessor, AutoModel
from timm.models.vision_transformer import Mlp

from .transformers import MultiHeadAttention, TransformerDecoderBlock

# from .model_utils import VideoRotaryEmbedding3D


logger = logging.getLogger(__name__)


def get_timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """Sinusoidal timestep embeddings following DiT-style parameterization."""

    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    timesteps = timesteps.float()
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half, device=timesteps.device, dtype=torch.float32
    )
    exponent = exponent / max(half - 1, 1)
    sinusoid = timesteps[:, None] * torch.exp(exponent)[None, :]
    embedding = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding.to(timesteps.dtype)


def build_backbone(cfg):
    name = cfg.backbone.backbone_name
    if name == "dinov3":
        processor, model = build_backbone_dinov3(cfg)
    else:
        raise ValueError(f"Unsupported backbone '{name}'. Expected: dinov3")
    return processor, model, name


def build_backbone_dinov3(cfg):
    pretrained_model_name = cfg.backbone.pretrained_model_name
    processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name, local_files_only=True
    )
    processor.size = {"height": cfg.image_size, "width": cfg.image_size}
    model = AutoModel.from_pretrained(pretrained_model_name, local_files_only=True)
    return processor, model


################ BASE MODEL ################
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        pass


################ HEATMAP PREDICTORS TO BE ADDED ON TOP OF THE WORLD MODEL ################


class HeatmapLatentPredictor(BaseModel):
    """This is just the 4 predictors, pixels and heatmaps, without backbone"""

    def __init__(self, cfg_model):
        super().__init__()
        self.square_heatmap_patch_predictor = HeatmapPatchPredictor(
            **cfg_model.heatmap_patch_predictor
        )
        self.circle_heatmap_patch_predictor = HeatmapPatchPredictor(
            **cfg_model.heatmap_patch_predictor
        )

        self.square_heatmap_pixel_predictor = HeatmapPixelPredictor(
            **cfg_model.heatmap_pixel_predictor
        )
        self.circle_heatmap_pixel_predictor = HeatmapPixelPredictor(
            **cfg_model.heatmap_pixel_predictor
        )

    def forward(self, batch):
        latents = batch["video_latents"]

        square_patch_logits = self.square_heatmap_patch_predictor(latents)
        circle_patch_logits = self.circle_heatmap_patch_predictor(latents)
        square_pixel_logits = self.square_heatmap_pixel_predictor(latents)
        circle_pixel_logits = self.circle_heatmap_pixel_predictor(latents)

        return {
            "square_patch_heatmap": square_patch_logits,  # [B, 1, H/P, W/P]
            "circle_patch_heatmap": circle_patch_logits,  # [B, 1, H/P, W/P]
            "square_heatmap_pixel": square_pixel_logits,  # [B, 1, H, W]
            "circle_heatmap_pixel": circle_pixel_logits,  # [B, 1, H, W]
        }


class HeatmapPatchPredictor(BaseModel):
    """Lightweight per-patch predictor head used for heatmap logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # input [B, D, H, W] -> output [B, 1, H, W]
        self.network = nn.Sequential(
            nn.Conv2d(
                input_dim, hidden_dim, kernel_size=1, stride=1
            ),  # [B, D, H, W] -> [B, hidden_dim, H, W]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(
                hidden_dim, 1, kernel_size=1, stride=1
            ),  # [B, hidden_dim, H, W] -> [B, 1, H, W]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.network(features)
        return logits


class HeatmapPixelPredictor(BaseModel):
    """Predicts a heatmap from latent inputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        patch_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                input_dim, hidden_dim, kernel_size=1
            ),  # [D, H/16, W/16] -> [hidden_dim, H/16, W/16]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(
                hidden_dim, 1, kernel_size=patch_size, stride=patch_size
            ),  # [hidden_dim, H/16, W/16] -> [1, H, W]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.network(features)
        return logits


################ WORLD MODELING PART ################
class DeterministicLatentPredictor(BaseModel):
    """Predicts future latents deterministically from observed context."""

    def __init__(
        self,
        cfg_model,
        *,
        number_context_latents,
        number_target_latents,
    ):
        super().__init__()
        self.cfg = cfg_model
        # infer H, W, P, H/P, and number of patches from config.
        self.patch_size = cfg_model.patch_size
        self.image_size = cfg_model.image_size
        self.w_patches = self.h_patches = self.image_size // self.patch_size
        self.number_context_latents = number_context_latents
        self.number_target_latents = number_target_latents

        predictor_cfg = cfg_model.deterministic_predictor
        self.predictor_cfg = predictor_cfg
        self.hidden_dim = predictor_cfg.hidden_dim
        num_heads = predictor_cfg.num_heads
        num_layers = predictor_cfg.num_layers
        mlp_ratio = predictor_cfg.mlp_ratio
        dropout = predictor_cfg.dropout
        self.input_dim = predictor_cfg.input_dim

        self.number_input_queries = (
            self.h_patches * self.w_patches * self.number_target_latents
        )
        self.number_context_tokens = (
            self.h_patches * self.w_patches * self.number_context_latents
        )
        self.copy_last_frame = predictor_cfg.copy_last_frame
        self.use_pos_encoding_at_every_transformer_layer = (
            predictor_cfg.use_pos_encoding_at_every_transformer_layer
        )

        # Input projections
        self.context_proj = nn.Linear(self.input_dim, self.hidden_dim)

        self.query_positional_encoding = nn.Parameter(
            torch.zeros(1, self.number_input_queries, self.hidden_dim)
        )
        self.context_positional_encoding = nn.Parameter(
            torch.zeros(1, self.number_context_tokens, self.hidden_dim)
        )
        nn.init.normal_(self.query_positional_encoding, std=0.02)
        nn.init.normal_(self.context_positional_encoding, std=0.02)

        self.blocks = nn.ModuleList(
            TransformerDecoderBlock(
                dim=self.hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.norm_out = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, context_latents):
        batch_size = context_latents.shape[0]
        query_positional_encoding = self.query_positional_encoding.expand(
            batch_size, -1, -1
        )
        context_positional_encoding = self.context_positional_encoding.expand(
            batch_size, -1, -1
        )
        if self.copy_last_frame:
            target_tokens = context_latents[:, -1:, :]  # [B, 1, D, num_h, num_w]
            target_tokens = target_tokens.expand(
                -1, self.number_target_latents, -1, -1, -1
            )
            target_tokens = rearrange(target_tokens, "b t d h w -> b (t h w) d")
        else:
            target_tokens = torch.zeros_like(query_positional_encoding)

        context_tokens = rearrange(context_latents, "b d t h w -> b (t h w) d")
        context_tokens = self.context_proj(context_tokens)

        if context_tokens.shape[1] != self.number_context_tokens:
            raise ValueError(
                f"Expected {self.number_context_tokens} context tokens, got {context_tokens.shape[1]}"
            )

        if self.use_pos_encoding_at_every_transformer_layer:
            block_query_positional_encoding = query_positional_encoding
            block_context_positional_encoding = context_positional_encoding
        else:
            target_tokens = target_tokens + query_positional_encoding
            context_tokens = context_tokens + context_positional_encoding
            block_query_positional_encoding = None
            block_context_positional_encoding = None

        for block in self.blocks:
            target_tokens = block(
                target_tokens=target_tokens,
                query_positional_encoding=block_query_positional_encoding,
                context_tokens=context_tokens,
                context_positional_encoding=block_context_positional_encoding,
            )

        predicted_tokens = self.output_proj(self.norm_out(target_tokens))
        predicted_latents = rearrange(
            predicted_tokens,
            "b (t h w) d -> b d t h w",
            t=self.number_target_latents,
            h=self.h_patches,
            w=self.w_patches,
        )
        return {
            "predicted_latents": predicted_latents  # [B, D, number_target_latents, n_h_patches, n_w_patches]
        }


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    shift = shift.unsqueeze(1)
    scale = scale.unsqueeze(1)
    return x * (1.0 + scale) + shift


# TODO: Check the Flow Matching Block
class FlowMatchingTransformerBlock(nn.Module):
    """Transformer block with AdaLN modulation conditioned on time embeddings."""

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout=dropout)
        self.ff = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), drop=dropout
        )
        self.norm_self = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_ff = nn.LayerNorm(dim, elementwise_affine=False)
        self.context_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 9, bias=True),
        )
        self._init_adaln_zero()

    def _init_adaln_zero(self) -> None:
        linear = self.modulation[-1]
        nn.init.constant_(linear.weight, 0.0)
        nn.init.constant_(linear.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        temb: torch.Tensor,
        target_positional_encoding: Optional[torch.Tensor] = None,
        context_positional_encoding: Optional[torch.Tensor] = None,
        target_rope: Optional[nn.Module] = None,
        context_rope: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        (
            shift_self,
            scale_self,
            gate_self,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_ff,
            scale_ff,
            gate_ff,
        ) = self.modulation(temb).chunk(9, dim=-1)
        # self attention, for target tokens
        target_for_self = (
            x + target_positional_encoding
            if target_positional_encoding is not None
            else x
        )
        attn_in = modulate(self.norm_self(target_for_self), shift_self, scale_self)
        attn_out = self.self_attn(attn_in, key=attn_in, value=attn_in, rope=target_rope)
        gate_self = gate_self.unsqueeze(1)
        x = x + self.dropout(gate_self * attn_out)
        # cross attention between target tokens and context tokens
        cross_query = (
            x + target_positional_encoding
            if target_positional_encoding is not None
            else x
        )
        cross_in = modulate(self.norm_cross(cross_query), shift_cross, scale_cross)
        gate_cross = gate_cross.unsqueeze(1)
        context_input = (
            context + context_positional_encoding
            if context_positional_encoding is not None
            else context
        )
        context_norm = self.context_norm(context_input)
        cross_out = self.cross_attn(
            cross_in,
            key=context_norm,
            value=context_norm,
            rope=(target_rope, context_rope),
        )
        x = x + self.dropout(gate_cross * cross_out)
        # Final FFN
        ff_in = modulate(self.norm_ff(x), shift_ff, scale_ff)
        gate_ff = gate_ff.unsqueeze(1)
        x = x + self.dropout(gate_ff * self.ff(ff_in))
        return x


class FlowMatchingVideoLatentPredictor(BaseModel):
    """Flow-matching latent predictor using AdaLN-modulated transformer blocks.

    Supports 3 action conditioning modes:
      - "none": no action conditioning (default, backward compatible)
      - "concat": project actions to embedding, tile across patches, concat with latent features
      - "cross_attn": project actions to hidden_dim tokens, append to cross-attention KV
      - "adaln": sum action embeddings with timestep embedding for AdaLN modulation
    """

    def __init__(
        self,
        cfg_model,
        *,
        number_context_latents: int,
        number_target_latents: int,
    ):
        super().__init__()
        self.cfg = cfg_model
        cfg_predictor = cfg_model.flow_matching_video_latent_predictor

        self.input_dim = cfg_predictor.input_dim
        self.hidden_dim = cfg_predictor.hidden_dim
        self.num_heads = cfg_predictor.num_heads
        self.num_layers = cfg_predictor.num_layers
        self.mlp_ratio = cfg_predictor.mlp_ratio
        self.dropout = cfg_predictor.dropout
        self.use_pos_encoding_at_every_transformer_layer = (
            cfg_predictor.use_pos_encoding_at_every_transformer_layer
        )
        self.use_rope = cfg_predictor.use_rope

        # compute spatial/token bookkeeping
        self.patch_size = cfg_model.patch_size
        self.image_size = cfg_model.image_size
        self.num_h_patches = self.image_size // self.patch_size
        self.num_w_patches = self.image_size // self.patch_size
        self.num_patches_per_frame = self.num_h_patches * self.num_w_patches
        self.number_context_latents = number_context_latents
        self.number_target_latents = number_target_latents
        self.number_context_tokens = (
            self.num_patches_per_frame * self.number_context_latents
        )
        self.number_target_tokens = (
            self.num_patches_per_frame * self.number_target_latents
        )

        # Action conditioning
        self.action_conditioning = getattr(cfg_predictor, "action_conditioning", "none")
        self.action_dim = getattr(cfg_predictor, "action_dim", 0)

        if self.action_conditioning != "none" and self.action_dim > 0:
            self._init_action_modules()

        # Input projections
        self.context_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.target_proj = nn.Linear(self.input_dim, self.hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )

        self.transformer_blocks = nn.ModuleList(
            FlowMatchingTransformerBlock(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        )
        self.norm_out = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.input_dim)

        self.context_positional_encoding = nn.Parameter(
            torch.zeros(1, self.number_context_tokens, self.hidden_dim)
        )
        self.target_positional_encoding = nn.Parameter(
            torch.zeros(1, self.number_target_tokens, self.hidden_dim)
        )

        nn.init.normal_(self.context_positional_encoding, std=0.02)
        nn.init.normal_(self.target_positional_encoding, std=0.02)

        head_dim = self.hidden_dim // self.num_heads

        if self.use_rope:
            self.target_rope = VideoRotaryEmbedding3D(
                head_dim=head_dim,
                grid_size=(
                    self.number_target_latents,
                    self.num_h_patches,
                    self.num_w_patches,
                ),
            )
            self.context_rope = VideoRotaryEmbedding3D(
                head_dim=head_dim,
                grid_size=(
                    self.number_context_latents,
                    self.num_h_patches,
                    self.num_w_patches,
                ),
            )
        else:
            self.target_rope = None
            self.context_rope = None

    def _init_action_modules(self):
        """Initialize action conditioning modules based on conditioning mode."""
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _time_embed(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(get_timestep_embedding(timesteps, self.hidden_dim))

    def _embed_actions_concat(self, actions, context_tokens, target_tokens):
        """Concat mode: embed actions per-frame, tile across patches, add to tokens."""
        # actions: [B, T_ctx + T_tgt, action_dim]
        action_emb = self.action_encoder(actions)  # [B, T, hidden_dim]
        ctx_act = action_emb[:, : self.number_context_latents]  # [B, T_ctx, hidden_dim]
        tgt_act = action_emb[:, self.number_context_latents :]  # [B, T_tgt, hidden_dim]

        # Tile across patches: [B, T, hidden_dim] -> [B, T*P, hidden_dim]
        ctx_act = ctx_act.unsqueeze(2).expand(-1, -1, self.num_patches_per_frame, -1)
        ctx_act = ctx_act.reshape(ctx_act.shape[0], -1, self.hidden_dim)
        tgt_act = tgt_act.unsqueeze(2).expand(-1, -1, self.num_patches_per_frame, -1)
        tgt_act = tgt_act.reshape(tgt_act.shape[0], -1, self.hidden_dim)

        context_tokens = context_tokens + ctx_act
        target_tokens = target_tokens + tgt_act
        return context_tokens, target_tokens

    def _embed_actions_cross_attn(self, actions, context_tokens):
        """Cross-attn mode: embed actions as extra tokens appended to context."""
        action_emb = self.action_encoder(actions)  # [B, T, hidden_dim]
        context_tokens = torch.cat([context_tokens, action_emb], dim=1)
        return context_tokens

    def _embed_actions_adaln(self, actions, temb):
        """AdaLN mode: sum per-sample action embedding with timestep embedding."""
        # Average actions across frames to get a single per-sample embedding
        action_emb = self.action_encoder(actions)  # [B, T, hidden_dim]
        action_emb = action_emb.mean(dim=1)  # [B, hidden_dim]
        return temb + action_emb

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context_latents: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: noisy target latents [B, D, T_tgt, H, W]
        t: timesteps [B]
        context_latents: [B, D, T_ctx, H, W]
        actions: optional [B, T_ctx + T_tgt, action_dim]
        """
        context_tokens = rearrange(context_latents, "b d t h w -> b (t h w) d")
        target_tokens = rearrange(x, "b d t h w -> b (t h w) d")

        context_tokens = self.context_proj(context_tokens)
        target_tokens = self.target_proj(target_tokens)

        # Apply action conditioning
        has_actions = actions is not None and self.action_conditioning != "none"
        if has_actions and self.action_conditioning == "concat":
            context_tokens, target_tokens = self._embed_actions_concat(
                actions, context_tokens, target_tokens
            )

        batch_size = context_tokens.shape[0]
        context_pe = self.context_positional_encoding.expand(batch_size, -1, -1)
        target_pe = self.target_positional_encoding.expand(batch_size, -1, -1)

        if has_actions and self.action_conditioning == "cross_attn":
            context_tokens = self._embed_actions_cross_attn(actions, context_tokens)
            # Pad context PE with zeros for action tokens
            action_pe = torch.zeros(
                batch_size,
                actions.shape[1],
                self.hidden_dim,
                device=context_pe.device,
                dtype=context_pe.dtype,
            )
            context_pe = torch.cat([context_pe, action_pe], dim=1)

        if self.use_pos_encoding_at_every_transformer_layer:
            block_context_pe = context_pe
            block_target_pe = target_pe
        else:
            context_tokens = context_tokens + context_pe
            target_tokens = target_tokens + target_pe
            block_context_pe = None
            block_target_pe = None

        temb = self._time_embed(t)

        if has_actions and self.action_conditioning == "adaln":
            temb = self._embed_actions_adaln(actions, temb)

        for block in self.transformer_blocks:
            target_tokens = block(
                target_tokens,
                context_tokens,
                temb,
                target_positional_encoding=block_target_pe,
                context_positional_encoding=block_context_pe,
                target_rope=self.target_rope,
                context_rope=self.context_rope,
            )

        target_tokens = self.norm_out(target_tokens)
        predictions = self.output_proj(target_tokens)

        predictions = rearrange(
            predictions,
            "b (t h w) d -> b d t h w",
            t=self.number_target_latents,
            h=self.num_h_patches,
            w=self.num_w_patches,
        )

        return predictions


#  Models from  RAE
"""Encoder registry for stage 2 models."""
# Import modules that perform registration on import.
from typing import Optional
from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class Stage2ModelProtocol(Protocol):
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def forward_with_cfg(self, *args: Any, **kwargs: Any) -> Any: ...
    def forward_with_autoguidance(self, *args: Any, **kwargs: Any) -> Any: ...
    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...
