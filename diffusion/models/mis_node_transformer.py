import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


class SwiGLU(nn.Module):
    """SwiGLU feedforward block."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embeddings + MLP."""

    def __init__(self, hidden_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        # t: [B]
        half = dim // 2
        if half == 0:
            return torch.zeros((t.shape[0], dim), device=t.device, dtype=torch.float32)
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / float(half)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class MaskedSelfAttention(nn.Module):
    """Self-attention with optional adjacency bias + padding mask."""

    def __init__(self, dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

        # learnable weight for adjacency/structure bias
        self.structure_bias_weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        adj_bias: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]
            adj_bias: optional [B, N, N] or [B, 1, N, N] or [B, H, N, N]
            node_mask: optional [B, N] bool (True for real nodes)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, Hd]

                # Build additive attention mask for SDPA.
        attn_mask = None

        # adjacency / structure bias
        if adj_bias is not None:
            if adj_bias.dim() == 3:
                adj = adj_bias.unsqueeze(1)  # [B,1,N,N]
            elif adj_bias.dim() == 4:
                adj = adj_bias
            else:
                raise ValueError("adj_bias must be 3D [B,N,N] or 4D [B,*,N,N]")

            # broadcast to heads if needed
            if adj.shape[1] == 1 and self.num_heads != 1:
                adj = adj.expand(-1, self.num_heads, -1, -1)

            if adj.dtype != q.dtype:
                adj = adj.to(dtype=q.dtype)

            adj = adj * self.structure_bias_weight
            attn_mask = adj

        # padding mask: disallow attending to padded keys
        if node_mask is not None:
            if node_mask.dtype != torch.bool:
                node_mask = node_mask.bool()
            key_mask = node_mask[:, None, None, :]  # [B,1,1,N]
            pad = (~key_mask).to(dtype=q.dtype) * (-1e9)
            attn_mask = pad if attn_mask is None else (attn_mask + pad)

        # SDPA will dispatch to FlashAttention / memory-efficient kernels when available.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj(out)

        # zero out padded queries
        if node_mask is not None:
            out = out * node_mask[:, :, None].to(out.dtype)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MaskedSelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        hidden_dim = int(dim * mlp_ratio * 2 / 3)
        self.mlp = SwiGLU(dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        adj_bias: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), adj_bias=adj_bias, node_mask=node_mask)
        x = x + self.mlp(self.norm2(x))
        if node_mask is not None:
            x = x * node_mask[:, :, None].to(x.dtype)
        return x


class MISNodeTransformer(nn.Module):
    """A MIS-specialized node transformer.

    - No coordinate relative bias (MIS has no geometric prior).
    - Explicitly conditions on x_t by concatenation.
    - Supports padding mask.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        depth: int = 16,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.node_embed = nn.Linear(input_dim, embed_dim)
        self.time_embed = TimestepEmbedder(embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.norm = RMSNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        t: torch.Tensor,
        x_t_node: torch.Tensor,
        adj_bias: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, F_static]
            t: [B]
            x_t_node: [B, N] (noisy node probabilities/indicators)
            adj_bias: [B, N, N]
            node_mask: [B, N] bool
        Returns:
            logits: [B, N]
        """
        if x_t_node.dim() != 2:
            raise ValueError("x_t_node must have shape [B, N]")
        if node_features.shape[:2] != x_t_node.shape:
            raise ValueError("node_features and x_t_node must match on [B, N]")

        x_in = torch.cat([node_features, x_t_node.unsqueeze(-1)], dim=-1)
        h = self.node_embed(x_in)
        t_emb = self.time_embed(t)
        h = h + t_emb.unsqueeze(1)
        if node_mask is not None:
            h = h * node_mask[:, :, None].to(h.dtype)

        for blk in self.blocks:
            h = blk(h, adj_bias=adj_bias, node_mask=node_mask)

        h = self.norm(h)
        logits = self.head(h).squeeze(-1)
        if node_mask is not None:
            logits = logits.masked_fill(~node_mask, -1e9)
        return logits


def get_mis_node_transformer(param_args, input_dim: int) -> MISNodeTransformer:
    """Factory aligned with other model factories."""
    return MISNodeTransformer(
        input_dim=input_dim,
        embed_dim=getattr(param_args, "hidden_dim", 128),
        depth=getattr(param_args, "num_layers", 16),
        num_heads=getattr(param_args, "num_heads", 4),
        mlp_ratio=4.0,
    )
