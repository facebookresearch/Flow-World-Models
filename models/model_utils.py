import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from typing import *
from math import pi
from collections.abc import Callable
import numpy as np

from .sdpa import scaled_dot_product_attention_with_fallback


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), (
        "invalid dimensions for broadcastable concatentation"
    )
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


######### FRANCOIS MODIFICATIONS FOR VIDEO ROPE #########


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # identical to Qwen2-VL: rotate the last-dim in 2D blocks
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list,
    unsqueeze_dim: int = 2,  # our q,k are [B, H, N, D], so broadcast along dim=2
):
    """
    Qwen2-VL's multimodal 3D RoPE application, adapted to our tensor layout.

    Args:
        q, k: shape [B, H, N, D_head]
        cos, sin: shape [3, B, N, D_head]  (3 = [t, h, w] axes)
        mrope_section: list of 3 ints (per-axis chunk size in HALF-dim), e.g. [d_t//2, d_h//2, d_w//2]
        unsqueeze_dim: where to unsqueeze for broadcasting to q/k (2 for [B,H,N,D])
    """
    # Qwen multiplies by 2 because RoPE works on pairs (cos,sin) across the last dim
    mrope_section = mrope_section * 2  # now counts full dim per axis (cos/sin paired)

    # Reorder chunks along last dim as in Qwen2-VL:
    # interleave axis-specific frequency bands so each axis rotates its own sub-chunk
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


########## END FRANCOIS MODIFICATIONS FOR VIDEO ROPE #########


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum("..., f -> ... f", t, freqs)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)

        freqs_w = torch.einsum("..., f -> ... f", t, freqs)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

        # print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], (
            f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        )
        t_left, t, t_right = (
            t[..., :start_index],
            t[..., start_index:end_index],
            t[..., end_index:],
        )
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        # print('======== shape of rope freq', self.freqs_cos.shape,freqs_sin.shape, '========')

    def forward(self, t):
        # print('======== shape of t', t.shape, '========')
        _, _, Lt, _ = t.shape  # B, num_heads, L, dim
        L, _ = self.freqs_cos.shape  # L, dim
        repeat = Lt // L
        freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin
        if repeat != 1:
            freqs_cos = freqs_cos.repeat_interleave(repeat, dim=0)
            freqs_sin = freqs_sin.repeat_interleave(repeat, dim=0)
            # print('======== shape of rope freq', freqs_cos.shape,freqs_sin.shape, '========')
            # print(f'======== repeat {repeat} times ========')
            # print(f'======== shape of t {t.shape} ========')
            # # assert the repeat is twice
            # #assert repeat == 2, f'repeat should be 2, but got {repeat}'
            # # check the content of the repeated freqs
            # # the content at odd index should be the same as the content at even index
            # assert torch.allclose(freqs_cos[::2], freqs_cos[1::2]), 'repeated freqs_cos are not the same'
            # assert torch.allclose(freqs_sin[::2], freqs_sin[1::2]), 'repeated freqs_sin are not the same'
        # apply repeated freqs
        return t * freqs_cos + rotate_half(t) * freqs_sin


class VideoRotaryEmbedding3D(nn.Module):
    """
    Minimal 3D RoPE (time, height, width) for tensors shaped [B, H, N, D_head].
    Matches Qwen2-VL mixing: splits frequency bands along the last dim and
    interleaves axis-specific bands: t, h, w, t, h, w, ...
    """

    def __init__(
        self,
        head_dim: int,
        grid_size: Tuple[int, int, int],
        base: float = 10000.0,
        axis_split_half: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.D = head_dim
        self.T, self.H, self.W = map(int, grid_size)
        N = self.T * self.H * self.W
        assert self.D % 2 == 0, "head_dim must be even for RoPE pairs"
        half = self.D // 2

        # split HALF-dim across (t,h,w) as evenly as possible (sum == half)
        if axis_split_half is None:
            Dt = half // 3
            Dh = (half - Dt) // 2
            Dw = half - Dt - Dh
        else:
            Dt, Dh, Dw = axis_split_half
            assert Dt + Dh + Dw == half, "axis_split_half must sum to D/2"

        self.mrope_section_half = (Dt, Dh, Dw)  # per-axis HALF-dim
        self.mrope_section_full = [2 * Dt, 2 * Dh, 2 * Dw]  # full dims (cos/sin paired)

        # inv_freq length == D/2
        inv = torch.arange(half, dtype=torch.float32) / float(half)
        inv_freq = base**-inv  # [D/2]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # precompute per-token axis indices for flattened order (t, then h, then w)
        t_ids = torch.arange(self.T).repeat_interleave(self.H * self.W)  # [N]
        hw = torch.arange(self.H * self.W)
        h_ids = (hw // self.W).repeat(self.T)  # [N]
        w_ids = (hw % self.W).repeat(self.T)  # [N]
        self.register_buffer("t_ids", t_ids.long(), persistent=False)
        self.register_buffer("h_ids", h_ids.long(), persistent=False)
        self.register_buffer("w_ids", w_ids.long(), persistent=False)

    def _axis_cos_sin(
        self, pos_ids: torch.Tensor, dtype: torch.dtype, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pos_ids: [N]
        freqs = torch.outer(
            pos_ids.to(device=device, dtype=self.inv_freq.dtype), self.inv_freq
        )  # [N, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [N, D]
        return emb.cos().to(dtype), emb.sin().to(dtype)  # each [N, D]

    def _mix_axes(
        self, cos_3: torch.Tensor, sin_3: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cos_3, sin_3: [3, N, D] with axis order [t, h, w].
        Interleave along last dim with chunks sized self.mrope_section_full repeating (t,h,w,t,h,w,...).
        Returns mixed cos/sin of shape [N, D].
        """
        N, D = cos_3.shape[1], cos_3.shape[2]
        sizes = self.mrope_section_full  # Sizes for [time, h, w]
        # split the full D into repeating chunks [Dt*2, Dh*2, Dw*2, Dt*2, ...]
        # number of chunks:
        k = (D + sum(sizes) - 1) // sum(sizes)
        split_sizes = (sizes * k)[:D]  # may overrun; slice trims last chunk
        # torch.split requires sum == D, so correct the last piece:
        if sum(split_sizes) != D:
            split_sizes[-1] += D - sum(split_sizes)

        cos_chunks = list(cos_3.split(split_sizes, dim=-1))  # list of [3, N, chunk]
        sin_chunks = list(sin_3.split(split_sizes, dim=-1))

        # pick axis i%3 for chunk i
        mixed_cos = torch.cat(
            [c[i % 3] for i, c in enumerate(cos_chunks)], dim=-1
        )  # [N, D]
        mixed_sin = torch.cat(
            [s[i % 3] for i, s in enumerate(sin_chunks)], dim=-1
        )  # [N, D]
        return mixed_cos, mixed_sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, N, D]  -> returns rotated x with 3D RoPE.
        We broadcast cos/sin over heads, so no per-head differences.
        """
        B, _, N, D = x.shape
        device, dtype = x.device, x.dtype
        assert D == self.D and N == self.T * self.H * self.W, (
            "Token dim or length mismatch vs grid_size/head_dim."
        )

        # cos/sin per axis
        cos_t, sin_t = self._axis_cos_sin(self.t_ids, dtype, device)  # [N, D]
        cos_h, sin_h = self._axis_cos_sin(self.h_ids, dtype, device)
        cos_w, sin_w = self._axis_cos_sin(self.w_ids, dtype, device)
        cos_3 = torch.stack([cos_t, cos_h, cos_w], dim=0)  # [3, N, D]
        sin_3 = torch.stack([sin_t, sin_h, sin_w], dim=0)  # [3, N, D]

        # Qwen2-VL style interleaving of frequency bands across axes
        cos_mix, sin_mix = self._mix_axes(cos_3, sin_3)  # each [N, D]

        # broadcast to [B, 1, N, D] so it matches [B, H, N, D]
        cos_mix = cos_mix.unsqueeze(0).unsqueeze(1).expand(B, 1, N, D)
        sin_mix = sin_mix.unsqueeze(0).unsqueeze(1).expand(B, 1, N, D)

        return (x * cos_mix) + (rotate_half(x) * sin_mix)


class RelativePositionBias2D(nn.Module):
    """
    2D relative positional bias for full self-attention.
    Creates a learnable bias table of size (2*H-1) (2*W-1) per head,
    and a fixed index map to look up bias for any pair of token positions.
    """

    def __init__(self, height: int, width: int, num_heads: int):
        super().__init__()
        self.height = height
        self.width = width
        self.num_heads = num_heads

        # Create a bias table: one bias for every possible relative offset
        # in y ∈ [-(H-1)..(H-1)] and x ∈ [-(W-1)..(W-1)]
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * height - 1) * (2 * width - 1), num_heads)
        )
        # Precompute a (H*W)×(H*W) index matrix of which bias entry each pair (i,j) uses
        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        # meshgrid of absolute coords, shape (H*W, 2)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        ).view(-1, 2)

        # Compute all pairwise relative coords
        relative_coords = coords[:, None, :] - coords[None, :, :]  # shape (HW, HW, 2)
        # shift to positive
        relative_coords[..., 0] += height - 1  # y
        relative_coords[..., 1] += width - 1  # x

        # flatten 2D index into single index: idx = y*(2W-1) + x
        relative_index = (
            relative_coords[..., 0] * (2 * width - 1) + relative_coords[..., 1]
        )
        # register as buffer so it’s on the right device / dtype
        self.register_buffer("relative_index", relative_index.long())

    def forward(self):
        """
        Returns:
           bias: Tensor of shape (1, num_heads, HW, HW)
        to be added to the raw attention logits before softmax.
        """
        # Lookup and reshape to (HW, HW, num_heads)
        bias = self.relative_bias_table[
            self.relative_index.view(-1)
        ]  # (HW*HW, num_heads)
        bias = bias.view(
            self.height * self.width, self.height * self.width, self.num_heads
        )  # (HW, HW, heads)
        # permute to (heads, HW, HW) and add batch-dim
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, HW, HW)
        return bias


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class NormAttention(nn.Module):
    """
    Attention module of LightningDiT.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        if use_rmsnorm:
            norm_layer = RMSNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # --- RoPE injection ---
        if rope is not None:
            # Back-compat: allow a callable (your previous VisionRoPE) OR a dict with Qwen2-VL video RoPE tensors.
            if callable(rope):
                q = rope(q)
                k = rope(k)
            else:
                raise ValueError("Expected 'rope' to be a callable RoPE function.")

        if self.fused_attn:
            q = q.to(v.dtype)
            k = k.to(v.dtype)  # rope may change the q,k's dtype
            x = scaled_dot_product_attention_with_fallback(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NormCrossAttention(nn.Module):
    """
    Cross-attention module based on LightningDiT's NormAttention.
    Queries are generated from x, while keys and values come from context_latents.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int = None,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        if use_rmsnorm:
            norm_layer = RMSNorm

        # Context dimension defaults to input dimension if not specified
        context_dim = context_dim or dim

        # Separate linear layers for Q (from x) and KV (from context_latents)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, context_latents: torch.Tensor, rope=None
    ) -> torch.Tensor:
        B, N, C = x.shape
        _, N_ctx, _ = context_latents.shape

        # Generate queries from x
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Generate keys and values from context_latents
        kv = (
            self.kv(context_latents)
            .reshape(B, N_ctx, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        # Apply normalization
        q, k = self.q_norm(q), self.k_norm(k)

        # --- RoPE injection ---
        if rope is not None:
            if callable(rope):
                q = rope(q)
                k = rope(k)
            else:
                raise ValueError("Expected 'rope' to be a callable RoPE function.")

        if self.fused_attn:
            q = q.to(v.dtype)
            k = k.to(v.dtype)  # rope may change the q,k's dtype
            x = scaled_dot_product_attention_with_fallback(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GaussianFourierEmbedding(nn.Module):
    """
    Gaussian Fourier Embedding for timesteps.
    """

    embedding_size: int = 256
    scale: float = 1.0

    def __init__(self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        self.W = nn.Parameter(
            torch.normal(0, self.scale, (embedding_size,)), requires_grad=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        with torch.no_grad():
            W = self.W  # stop gradient manually
        t = t[:, None] * W[None, :] * 2 * torch.pi
        # Concatenate sine and cosine transformations
        t_embed = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)
        t_embed = self.mlp(t_embed)
        return t_embed


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
