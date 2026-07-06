import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lightningDiT import Mlp, NormAttention
from timm.models.vision_transformer import Mlp
from .model_utils import (
    RMSNorm,
    SwiGLUFFN,
    GaussianFourierEmbedding,
    NormAttention,
    NormCrossAttention,
    VideoRotaryEmbedding3D,
)
from typing import *
from typing import Callable, Optional, Tuple, Union  # explicit: annotations on PatchEmbed3D
from einops import rearrange


# ---------------------------------------------------------------------------
# PatchEmbed3D + to_3tuple: lifted verbatim from a local timm fork (the
# "FRANCOIS MODIFICATIONS" block of timm/layers/patch_embed.py) so this repo can
# depend on stock timm==1.0.20 from PyPI instead of a vendored copy. The
# original timm code is Apache-2.0. This is the 3D (video) analogue of timm's
# 2D PatchEmbed and is used by DiTwDDTHead below.
# ---------------------------------------------------------------------------
def to_3tuple(x) -> Tuple[int, int, int]:
    if isinstance(x, (tuple, list)):
        assert len(x) == 3, "Expected a 3-tuple"
        return tuple(x)
    return (x, x, x)


class PatchEmbed3D(nn.Module):
    """
    3D Video to Patch Embedding (Conv3d-based)

    Input:  x of shape [B, C, T, H, W]
    Output:
      - if flatten=True:  [B, N, embed_dim], where N = (T//pt)*(H//ph)*(W//pw) (or ceil if dynamic pad)
      - else:             [B, embed_dim, T', H', W']
    """

    def __init__(
        self,
        vid_size: Optional[
            Union[int, Tuple[int, int, int]]
        ] = None,  # (T, H, W) if you want static checks
        patch_size: Union[int, Tuple[int, int, int]] = 2,  # (pt, ph, pw)
        in_chans: int = 384,
        embed_dim: int = 768,
        norm_layer: Optional[
            Callable[[int], nn.Module]
        ] = None,  # e.g., nn.LayerNorm for flattened output
        flatten: bool = True,
        bias: bool = True,
        strict_vid_size: bool = True,
        dynamic_img_pad: bool = False,  # applies to T/H/W in 3D as dynamic padding
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.vid_size, self.grid_size, self.num_patches = self._init_vid_size(vid_size)

        self.flatten = flatten
        self.strict_vid_size = strict_vid_size
        self.dynamic_img_pad = dynamic_img_pad

        # Project patches with a single Conv3d (kernel=stride=patch_size)
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )

        # By convention (like timm 2D), if flatten=True, norm is usually LayerNorm(embed_dim).
        # If flatten=False (NCDHW), users may prefer nn.BatchNorm3d(embed_dim) or Identity.
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_vid_size(self, vid_size: Optional[Union[int, Tuple[int, int, int]]]):
        if vid_size is None:
            return None, None, None
        vid_size = to_3tuple(vid_size)  # (T, H, W)
        grid_size = tuple(s // p for s, p in zip(vid_size, self.patch_size))
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        return vid_size, grid_size, num_patches

    def set_input_size(
        self,
        vid_size: Optional[Union[int, Tuple[int, int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    ):
        # Optional helper if you want to change sizes after init (weights are NOT resampled here).
        if patch_size is not None:
            new_ps = to_3tuple(patch_size)
            if new_ps != self.patch_size:
                self.patch_size = new_ps
                # Rebuild projection (without resampling old weights).
                old = self.proj
                self.proj = nn.Conv3d(
                    old.in_channels,
                    old.out_channels,
                    kernel_size=new_ps,
                    stride=new_ps,
                    bias=old.bias is not None,
                )
                # NOTE: if you need weight resampling, add a custom routine here.

        vsize = vid_size or self.vid_size
        if vsize is not None:
            self.vid_size, self.grid_size, self.num_patches = self._init_vid_size(vsize)

    def feat_ratio(self, as_scalar: bool = True) -> Union[int, Tuple[int, int, int]]:
        return max(self.patch_size) if as_scalar else self.patch_size

    def dynamic_feat_size(self, vid_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        pt, ph, pw = self.patch_size
        if self.dynamic_img_pad:
            return (
                math.ceil(vid_size[0] / pt),
                math.ceil(vid_size[1] / ph),
                math.ceil(vid_size[2] / pw),
            )
        else:
            return (
                vid_size[0] // pt,
                vid_size[1] // ph,
                vid_size[2] // pw,
            )

    def forward(self, x: torch.Tensor):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        if self.vid_size is not None:
            if self.strict_vid_size:
                assert T == self.vid_size[0], (
                    f"Input T ({T}) != expected ({self.vid_size[0]})."
                )
                assert H == self.vid_size[1], (
                    f"Input H ({H}) != expected ({self.vid_size[1]})."
                )
                assert W == self.vid_size[2], (
                    f"Input W ({W}) != expected ({self.vid_size[2]})."
                )
            elif not self.dynamic_img_pad:
                pt, ph, pw = self.patch_size
                assert T % pt == 0, f"T={T} must be divisible by pt={pt}."
                assert H % ph == 0, f"H={H} must be divisible by ph={ph}."
                assert W % pw == 0, f"W={W} must be divisible by pw={pw}."

        if self.dynamic_img_pad:
            pt, ph, pw = self.patch_size
            pad_t = (pt - T % pt) % pt
            pad_h = (ph - H % ph) % ph
            pad_w = (pw - W % pw) % pw
            # F.pad for 5D expects pad in (W, H, T) order: (w_left, w_right, h_left, h_right, t_left, t_right)
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))

        x = self.proj(x)  # [B, embed_dim, T', H', W']

        if self.flatten:
            # Flatten spatial-temporal dims and return [B, N, C]
            x = x.flatten(2).transpose(1, 2)  # [B, C, N] -> [B, N, C]
            x = self.norm(x)  # e.g., LayerNorm over last dim
        else:
            # Keep [B, C, T', H', W']; norm is expected to be Identity or BatchNorm3d
            x = self.norm(x)

        return x


def DDTModulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        shift: Tensor of shape (B, L, D)
        scale: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * (1 + scale) + shift,
        with shift and scale repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * (1 + scale) + shift


def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        gate: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * gate,
        with gate repeated to match L_x if necessary.
    """
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        # print(f"gate shape: {gate.shape}, x shape: {x.shape}")
        gate = gate.repeat_interleave(repeat, dim=1)
    # apply modulation
    return x * gate


class LightningDDTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including:
    - ROPE
    - QKNorm
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        **block_kwargs,
    ):
        super().__init__()

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            self.norm3 = RMSNorm(hidden_size)
        if not use_rmsnorm:
            self.norm_context = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
        else:
            self.norm_context = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs,
        )

        self.cross_attn = NormCrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs,
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, context_latents=None, feat_rope=None):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)  # (B, 1, C)
        if self.wo_shift:
            scale_msa, gate_msa, scale_ca, gate_ca, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=-1)
            )
            shift_msa = None
            shift_mlp = None
        else:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_ca,
                scale_ca,
                gate_ca,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.adaLN_modulation(c).chunk(9, dim=-1)
        # self attention (target latents)
        x = x + DDTGate(
            self.attn(DDTModulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope),
            gate_msa,
        )
        # cross attention with context latents
        x = x + DDTGate(
            self.cross_attn(
                DDTModulate(self.norm2(x), shift_ca, scale_ca),
                context_latents=self.norm_context(context_latents),
                rope=feat_rope,
            ),
            gate_ca,
        )
        # Final MLP
        x = x + DDTGate(
            self.mlp(DDTModulate(self.norm3(x), shift_mlp, scale_mlp)), gate_mlp
        )
        return x


class DDTFinalLayer(nn.Module):
    """
    The final layer of DDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)  # no gate
        x = self.linear(x)
        return x


class PositionalEncoding3D(nn.Module):
    """
    Flexible 3D positional encoding with multiple initialization strategies.

    Args:
        embed_dim: Embedding dimension
        grid_size: (T, H, W) tuple of patch grid dimensions
        mode: One of:
            - 'learned': Random init, fully learnable
            - 'sincos_frozen': Sin-cos init, frozen
            - 'sincos_learnable': Sin-cos init, learnable
        init_std: Std for learned mode initialization
    """

    def __init__(
        self,
        embed_dim: int,
        grid_size: Tuple[int, int, int],
        mode: str = "learned",
        init_std: float = 0.02,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.mode = mode

        T, H, W = grid_size
        num_patches = T * H * W

        # Generate the embedding tensor
        if mode == "learned":
            pe = torch.empty(num_patches, embed_dim)
            nn.init.trunc_normal_(pe, std=init_std)
            self.pe = nn.Parameter(pe)
        elif mode in ("sincos_frozen", "sincos_learnable"):
            pe = self._make_sincos_3d(T, H, W, embed_dim)
            if mode == "sincos_frozen":
                self.register_buffer("pe", pe)
            else:
                self.pe = nn.Parameter(pe)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'learned', 'sincos_frozen', or 'sincos_learnable'"
            )

    def _make_sincos_3d(self, T: int, H: int, W: int, D: int) -> torch.Tensor:
        """Generate 3D sinusoidal positional embeddings."""
        # Split dimensions: time gets remainder to ensure even splits for spatial
        dim_spatial_each = (D // 6) * 2
        dim_t = D - 2 * dim_spatial_each

        def sincos_1d(seq_len: int, dim: int) -> torch.Tensor:
            pos = torch.arange(seq_len, dtype=torch.float32)
            omega = torch.arange(dim // 2, dtype=torch.float32)
            omega = 1.0 / (10000 ** (2 * omega / dim))
            out = pos[:, None] * omega[None, :]
            return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

        emb_t = sincos_1d(T, dim_t)
        emb_h = sincos_1d(H, dim_spatial_each)
        emb_w = sincos_1d(W, dim_spatial_each)

        # Create 3D grid and index
        grid_t, grid_h, grid_w = torch.meshgrid(
            torch.arange(T), torch.arange(H), torch.arange(W), indexing="ij"
        )

        pe = torch.cat(
            [
                emb_t[grid_t.flatten()],
                emb_h[grid_h.flatten()],
                emb_w[grid_w.flatten()],
            ],
            dim=-1,
        )

        return pe  # [T*H*W, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor [B, N, D]."""
        return x + self.pe.unsqueeze(0)


class DiTwDDTHead(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        noise_input_size: int = 1,
        patch_size: Union[list, int] = 1,
        in_channels: int = 768,
        hidden_size=[1152, 2048],
        depth=[28, 2],
        num_heads: Union[list[int], int] = [16, 16],
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        # num_classes=1000,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_pos_embed: bool = True,
        pos_embed_mode: str = "learned",
        spatial_dimension: int = 32,
        backbone_patch_size: int = 16,
        action_dim: int = 0,
        action_conditioning: str = "adaln",
        proprio_dim: int = 0,
        action_emb_dim: int = 10,
        proprio_emb_dim: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.action_dim = action_dim
        self.action_conditioning = action_conditioning
        self.proprio_dim = proprio_dim
        self.action_emb_dim = action_emb_dim
        self.proprio_emb_dim = proprio_emb_dim

        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]
        self.num_heads = (
            [num_heads, num_heads] if isinstance(num_heads, int) else list(num_heads)
        )
        self.num_decoder_blocks = depth[1]
        self.num_encoder_blocks = depth[0]
        self.num_blocks = depth[0] + depth[1]
        self.use_pos_embed = use_pos_embed
        self.pos_embed_mode = pos_embed_mode
        self.use_rope = use_rope
        # analyze patch size
        if isinstance(patch_size, int) or isinstance(patch_size, float):
            patch_size = [patch_size, patch_size]  # patch size for s , x embed
        assert len(patch_size) == 2, (
            f"patch size should be a list of two numbers, but got {patch_size}"
        )
        self.patch_size = patch_size
        self.s_patch_size = patch_size[0]
        self.x_patch_size = patch_size[1]
        spatial_dimension_patch = spatial_dimension // backbone_patch_size
        s_channel_per_token = in_channels * self.s_patch_size * self.s_patch_size
        s_patch_size = self.s_patch_size

        x_channel_per_token = in_channels * self.x_patch_size * self.x_patch_size
        x_patch_size = self.x_patch_size

        self.s_embedder = PatchEmbed3D(
            vid_size=(
                noise_input_size,
                spatial_dimension_patch,
                spatial_dimension_patch,
            ),
            patch_size=(s_patch_size, s_patch_size, s_patch_size),
            in_chans=in_channels,
            embed_dim=self.encoder_hidden_size,
            bias=True,
        )

        self.s_context_embedder = PatchEmbed3D(
            vid_size=(input_size, spatial_dimension_patch, spatial_dimension_patch),
            patch_size=(s_patch_size, s_patch_size, s_patch_size),
            in_chans=in_channels,
            embed_dim=self.encoder_hidden_size,
            bias=True,
        )

        # These are the decoders for target_latents and context_latents
        self.x_embedder = PatchEmbed3D(
            vid_size=(
                noise_input_size,
                spatial_dimension_patch,
                spatial_dimension_patch,
            ),
            patch_size=(x_patch_size, x_patch_size, x_patch_size),
            in_chans=in_channels,
            embed_dim=self.decoder_hidden_size,
            bias=True,
        )

        self.x_context_embedder = PatchEmbed3D(
            vid_size=(input_size, spatial_dimension_patch, spatial_dimension_patch),
            patch_size=(x_patch_size, x_patch_size, x_patch_size),
            in_chans=in_channels,
            embed_dim=self.decoder_hidden_size,
            bias=True,
        )

        self.s_channel_per_token = s_channel_per_token
        self.x_channel_per_token = x_channel_per_token
        self.s_projector = (
            nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
            if self.encoder_hidden_size != self.decoder_hidden_size
            else nn.Identity()
        )
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        # self.y_embedder = LabelEmbedder(
        #     num_classes, self.encoder_hidden_size, class_dropout_prob)
        # print(f"x_channel_per_token: {x_channel_per_token}, s_channel_per_token: {s_channel_per_token}")
        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size, 1, x_channel_per_token, use_rmsnorm=use_rmsnorm
        )

        if self.use_pos_embed:
            enc_grid = tuple(
                map(int, self.s_embedder.grid_size)
            )  # (T_target, H/P, W/P)
            enc_context_grid = tuple(
                map(int, self.s_context_embedder.grid_size)
            )  # (T_context, H/P, W/P)
            dec_grid = tuple(
                map(int, self.x_embedder.grid_size)
            )  # (T_target, H/P, W/P)
            dec_context_grid = tuple(
                map(int, self.x_context_embedder.grid_size)
            )  # (T_context, H/P, W/P)

            self.pos_embed_s_target = PositionalEncoding3D(
                self.encoder_hidden_size, enc_grid, mode=pos_embed_mode
            )
            self.pos_embed_s_context = PositionalEncoding3D(
                self.encoder_hidden_size, enc_context_grid, mode=pos_embed_mode
            )
            self.pos_embed_x_target = PositionalEncoding3D(
                self.decoder_hidden_size, dec_grid, mode=pos_embed_mode
            )
            self.pos_embed_x_context = PositionalEncoding3D(
                self.decoder_hidden_size, dec_context_grid, mode=pos_embed_mode
            )

        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]

        # --- replace the whole if self.use_rope: block with this ---
        if self.use_rope:
            enc_head_dim = self.encoder_hidden_size // enc_num_heads
            dec_head_dim = (
                self.decoder_hidden_size // dec_num_heads
            )  # by default dec_num_heads is 16. This seems very (too?) high
            self.enc_feat_rope = VideoRotaryEmbedding3D(
                head_dim=enc_head_dim,
                grid_size=tuple(
                    map(int, self.s_embedder.grid_size)
                ),  # (T,num_h_patches, num_w_patches)
            )
            self.dec_feat_rope = VideoRotaryEmbedding3D(
                head_dim=dec_head_dim,
                grid_size=tuple(
                    map(int, self.x_embedder.grid_size)
                ),  # (T,num_h_patches, num_w_patches)
            )
        else:
            self.enc_feat_rope = None
            self.dec_feat_rope = None

        self.blocks = nn.ModuleList(
            [
                LightningDDTBlock(
                    self.encoder_hidden_size
                    if i < self.num_encoder_blocks
                    else self.decoder_hidden_size,
                    enc_num_heads if i < self.num_encoder_blocks else dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu,
                    wo_shift=wo_shift,
                )
                for i in range(self.num_blocks)
            ]
        )

        # Action and proprio conditioning modules
        if self.action_dim > 0 or self.proprio_dim > 0:
            self._init_conditioning_modules()

        self.initialize_weights()

    def initialize_weights(self, xavier_uniform_init: bool = False):
        if xavier_uniform_init:

            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

            self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        def init_patch_embed(embedder):
            w = embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if embedder.proj.bias is not None:
                nn.init.constant_(embedder.proj.bias, 0)

        init_patch_embed(self.x_embedder)
        init_patch_embed(self.s_embedder)
        init_patch_embed(self.x_context_embedder)
        init_patch_embed(self.s_context_embedder)

        if isinstance(self.s_projector, nn.Linear):
            nn.init.xavier_uniform_(self.s_projector.weight)
            if self.s_projector.bias is not None:
                nn.init.constant_(self.s_projector.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        # if self.use_pos_embed:
        #     # Initialize (and freeze) pos_embed by sin-cos embedding:
        #     pos_embed = get_2d_sincos_pos_embed(
        #         self.pos_embed.shape[-1], int(self.s_embedder.num_patches ** 0.5))
        #     self.pos_embed.data.copy_(
        #         torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x has shape [B, T * num_h_patches * num_w_patches, D]
        """
        (T, num_h_patches, num_w_patches) = self.x_embedder.grid_size
        x = rearrange(
            x, "b (t h w) d -> b d t h w", t=T, h=num_h_patches, w=num_w_patches
        )
        return x

    def _init_conditioning_modules(self):
        """Initialize action and proprio conditioning modules.

        Uses shared Conv1d encoder (ProprioceptiveEmbedding) + model-specific
        linear projections to match each stage's hidden size.
        """
        from .proprio import ProprioceptiveEmbedding

        # --- Action modules ---
        if self.action_dim > 0:
            self.action_conv_encoder = ProprioceptiveEmbedding(
                in_chans=self.action_dim, emb_dim=self.action_emb_dim
            )
            self.action_projector = nn.Sequential(
                nn.Linear(self.action_emb_dim, self.encoder_hidden_size),
                nn.SiLU(),
                nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size),
            )
            if self.action_conditioning in ("concat", "cross_attn"):
                self.action_decoder_proj = nn.Sequential(
                    nn.Linear(self.action_emb_dim, self.decoder_hidden_size),
                    nn.SiLU(),
                    nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
                )
            elif self.action_conditioning == "adaln":
                self.action_adaln_decoder_proj = nn.Linear(
                    self.encoder_hidden_size, self.decoder_hidden_size
                )

        # --- Proprio modules (mirrors action modules) ---
        if self.proprio_dim > 0:
            self.proprio_conv_encoder = ProprioceptiveEmbedding(
                in_chans=self.proprio_dim, emb_dim=self.proprio_emb_dim
            )
            self.proprio_projector = nn.Sequential(
                nn.Linear(self.proprio_emb_dim, self.encoder_hidden_size),
                nn.SiLU(),
                nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size),
            )
            if self.action_conditioning in ("concat", "cross_attn"):
                self.proprio_decoder_proj = nn.Sequential(
                    nn.Linear(self.proprio_emb_dim, self.decoder_hidden_size),
                    nn.SiLU(),
                    nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
                )
            elif self.action_conditioning == "adaln":
                self.proprio_adaln_decoder_proj = nn.Linear(
                    self.encoder_hidden_size, self.decoder_hidden_size
                )

    def _encode_signal(self, raw, conv_encoder, projector):
        """Encode a signal (action or proprio) with Conv1d then project."""
        emb = conv_encoder(raw)  # [B, T, emb_dim]
        return projector(emb)  # [B, T, hidden_size]

    def _embed_adaln(self, raw, conv_encoder, projector, c):
        """AdaLN mode: mean-pool projected embedding and add to conditioning signal c."""
        emb = self._encode_signal(raw, conv_encoder, projector)  # [B, T, hidden]
        return c + emb.mean(dim=1)

    def _embed_concat(
        self,
        raw,
        conv_encoder,
        projector,
        tokens,
        context_tokens,
        num_ctx_frames,
        patches_per_frame,
    ):
        """Concat mode: tile per-frame embeddings across patches and add to tokens."""
        emb = self._encode_signal(raw, conv_encoder, projector)  # [B, T, hidden]
        ctx = emb[:, :num_ctx_frames]
        tgt = emb[:, num_ctx_frames:]

        ctx = ctx.unsqueeze(2).expand(-1, -1, patches_per_frame, -1)
        ctx = ctx.reshape(ctx.shape[0], -1, ctx.shape[-1])
        tgt = tgt.unsqueeze(2).expand(-1, -1, patches_per_frame, -1)
        tgt = tgt.reshape(tgt.shape[0], -1, tgt.shape[-1])

        return tokens + tgt, context_tokens + ctx

    def _embed_cross_attn(self, raw, conv_encoder, projector, context):
        """Cross-attn mode: append tokens to context for cross-attention."""
        emb = self._encode_signal(raw, conv_encoder, projector)  # [B, T, hidden]
        return torch.cat([context, emb], dim=1)

    def _apply_conditioning_adaln(self, actions, states, c):
        """Apply adaln conditioning from actions and proprio to c. Returns (c, decoder_residual)."""
        decoder_residual = None
        if actions is not None and self.action_dim > 0:
            c = self._embed_adaln(
                actions, self.action_conv_encoder, self.action_projector, c
            )
            enc_emb = self._encode_signal(
                actions, self.action_conv_encoder, self.action_projector
            )
            decoder_residual = self.action_adaln_decoder_proj(
                enc_emb.mean(dim=1)
            ).unsqueeze(1)
        if states is not None and self.proprio_dim > 0:
            c = self._embed_adaln(
                states, self.proprio_conv_encoder, self.proprio_projector, c
            )
            enc_emb = self._encode_signal(
                states, self.proprio_conv_encoder, self.proprio_projector
            )
            dec_res = self.proprio_adaln_decoder_proj(enc_emb.mean(dim=1)).unsqueeze(1)
            decoder_residual = (
                decoder_residual + dec_res if decoder_residual is not None else dec_res
            )
        return c, decoder_residual

    def forward(
        self, x, t, context_latents, actions=None, states=None, s=None, mask=None
    ):
        # x has shape [B, C, T_target, H, W]  -> Noise
        t = self.t_embedder(t)  # [B, hidden_dim]
        c = nn.functional.silu(t)  # [B, hidden_dim]

        # AdaLN conditioning modifies c before encoder blocks
        decoder_residual = None
        if self.action_conditioning == "adaln":
            c, decoder_residual = self._apply_conditioning_adaln(actions, states, c)

        # Encode target_latents and context_latents with 3D patchifier
        s_target = self.s_embedder(x)
        s_context = self.s_context_embedder(context_latents)

        # Concat conditioning: tile and add to encoder tokens
        if self.action_conditioning == "concat":
            enc_ppf = int(self.s_embedder.grid_size[1] * self.s_embedder.grid_size[2])
            num_ctx = int(self.s_context_embedder.grid_size[0])
            if actions is not None and self.action_dim > 0:
                s_target, s_context = self._embed_concat(
                    actions,
                    self.action_conv_encoder,
                    self.action_projector,
                    s_target,
                    s_context,
                    num_ctx,
                    enc_ppf,
                )
            if states is not None and self.proprio_dim > 0:
                s_target, s_context = self._embed_concat(
                    states,
                    self.proprio_conv_encoder,
                    self.proprio_projector,
                    s_target,
                    s_context,
                    num_ctx,
                    enc_ppf,
                )

        # Positional encoding
        if self.use_pos_embed:
            s_target = self.pos_embed_s_target(s_target)
            s_context = self.pos_embed_s_context(s_context)

        # Cross-attn conditioning: append tokens to context
        if self.action_conditioning == "cross_attn":
            if actions is not None and self.action_dim > 0:
                s_context = self._embed_cross_attn(
                    actions, self.action_conv_encoder, self.action_projector, s_context
                )
            if states is not None and self.proprio_dim > 0:
                s_context = self._embed_cross_attn(
                    states, self.proprio_conv_encoder, self.proprio_projector, s_context
                )

        # Encoder blocks
        for i in range(self.num_encoder_blocks):
            s_target = self.blocks[i](
                s_target, c, context_latents=s_context, feat_rope=self.enc_feat_rope
            )

        # Prepare for decoding
        t = t.unsqueeze(1).repeat(1, s_target.shape[1], 1)
        s_target = nn.functional.silu(t + s_target)
        s_target = self.s_projector(s_target)

        # Inject adaln decoder residual
        if decoder_residual is not None:
            s_target = s_target + decoder_residual

        x_target = self.x_embedder(x)
        x_context = self.x_context_embedder(context_latents)

        # Concat conditioning for decoder stage
        if self.action_conditioning == "concat":
            dec_ppf = int(self.x_embedder.grid_size[1] * self.x_embedder.grid_size[2])
            num_ctx = int(self.x_context_embedder.grid_size[0])
            if actions is not None and self.action_dim > 0:
                x_target, x_context = self._embed_concat(
                    actions,
                    self.action_conv_encoder,
                    self.action_decoder_proj,
                    x_target,
                    x_context,
                    num_ctx,
                    dec_ppf,
                )
            if states is not None and self.proprio_dim > 0:
                x_target, x_context = self._embed_concat(
                    states,
                    self.proprio_conv_encoder,
                    self.proprio_decoder_proj,
                    x_target,
                    x_context,
                    num_ctx,
                    dec_ppf,
                )

        if self.use_pos_embed:
            x_target = self.pos_embed_x_target(x_target)
            x_context = self.pos_embed_x_context(x_context)

        # Cross-attn conditioning for decoder stage
        if self.action_conditioning == "cross_attn":
            if actions is not None and self.action_dim > 0:
                x_context = self._embed_cross_attn(
                    actions,
                    self.action_conv_encoder,
                    self.action_decoder_proj,
                    x_context,
                )
            if states is not None and self.proprio_dim > 0:
                x_context = self._embed_cross_attn(
                    states,
                    self.proprio_conv_encoder,
                    self.proprio_decoder_proj,
                    x_context,
                )

        # Decode with DDT blocks
        for i in range(self.num_encoder_blocks, self.num_blocks):
            x_target = self.blocks[i](
                x_target,
                s_target,
                context_latents=x_context,
                feat_rope=self.dec_feat_rope,
            )

        # Critical part
        x_target = self.final_layer(x_target, s_target)  # [B, N, D]
        x_target = self.unpatchify(x_target)
        return x_target

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(0, 1)):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        acombined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, (
            "cfg_interval should be (min, max) with min < max"
        )
        t = t[: len(t) // 2]  # get t for the conditional half
        half_eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)).view(
                -1, *[1] * (len(cond_eps.shape) - 1)
            ),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps),
            cond_eps,
        )
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward_with_autoguidance(
        self, x, t, y, cfg_scale, additional_model_forward, cfg_interval=(0, 1)
    ):
        """
        Forward pass of LightningDiT, but also contain the forward pass for the additional model
        """
        model_out = self.forward(x, t, y)
        ag_model_out = additional_model_forward(x, t, y)
        eps = model_out[:, : self.in_channels]
        ag_eps = ag_model_out[:, : self.in_channels]

        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, (
            "cfg_interval should be (min, max) with min < max"
        )
        eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)).view(
                -1, *[1] * (len(eps.shape) - 1)
            ),
            ag_eps + cfg_scale * (eps - ag_eps),
            eps,
        )

        return eps
