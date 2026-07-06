import torch.nn as nn

from .sdpa import scaled_dot_product_attention_with_fallback


class MultiHeadAttention(nn.Module):
    """Multi-head attention using PyTorch SDPA with efficient backend."""

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key=None, value=None, attn_mask=None, rope=None):
        if query.dim() != 3:
            raise ValueError("query must be of shape [B, L, D]")
        bsz, q_len, _ = query.shape
        k_len = key.shape[1]
        v_len = value.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            if isinstance(rope, tuple):
                rope_q, rope_k = rope
            else:
                rope_q = rope_k = rope
            if rope_q is not None:
                q = rope_q(q)
            if rope_k is not None:
                k = rope_k(k)

        dropout_p = self.dropout if self.training else 0.0

        attn = scaled_dot_product_attention_with_fallback(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

        attn = attn.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.out_proj(attn)


class FeedForward(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerDecoderBlock(nn.Module):
    """Pre-norm transformer decoder block with self- and cross-attention."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout=dropout)

        self.cross_attn_norm_q = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout=dropout)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target_tokens,
        query_positional_encoding=None,
        context_tokens=None,
        context_positional_encoding=None,
        self_attn_mask=None,
        cross_attn_mask=None,
    ):
        if context_tokens is None:
            raise ValueError("context_tokens must be provided")

        # Self-attention follows DETR: add learned positional encoding to queries and keys only.
        self_query = (
            target_tokens + query_positional_encoding
            if query_positional_encoding is not None
            else target_tokens
        )
        self_key = (
            target_tokens + query_positional_encoding
            if query_positional_encoding is not None
            else target_tokens
        )
        self_value = target_tokens

        self_attn_out = self.self_attn(
            self.self_attn_norm(self_query),
            key=self.self_attn_norm(self_key),
            value=self.self_attn_norm(self_value),
            attn_mask=self_attn_mask,
        )
        target_tokens = target_tokens + self.dropout(self_attn_out)

        # Cross-attention uses context positional encodings on keys only.
        cross_query = (
            target_tokens + query_positional_encoding
            if query_positional_encoding is not None
            else target_tokens
        )
        cross_key = (
            context_tokens + context_positional_encoding
            if context_positional_encoding is not None
            else context_tokens
        )
        cross_value = context_tokens

        cross_attn_out = self.cross_attn(
            self.cross_attn_norm_q(cross_query),
            key=cross_key,
            value=cross_value,
            attn_mask=cross_attn_mask,
        )
        target_tokens = target_tokens + self.dropout(cross_attn_out)
        return target_tokens + self.dropout(self.ff(self.ff_norm(target_tokens)))
