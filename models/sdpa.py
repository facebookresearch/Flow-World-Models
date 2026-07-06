import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


_FAST_SDPA_DTYPES = {torch.float16, torch.bfloat16}
_FAST_SDPA_BACKENDS = (
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
)


def _can_use_fast_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return (
        q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and q.dtype == k.dtype == v.dtype
        and q.dtype in _FAST_SDPA_DTYPES
    )


def scaled_dot_product_attention_with_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
):
    backends = [SDPBackend.MATH]
    if _can_use_fast_sdpa(q, k, v):
        backends = [*_FAST_SDPA_BACKENDS, SDPBackend.MATH]

    last_error = None
    for backend in backends:
        try:
            with sdpa_kernel(backend):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
        except RuntimeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("No SDPA backend is available.")
