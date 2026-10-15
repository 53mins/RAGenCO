import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed

class SwiGLU(nn.Module):
    """
    SwiGLU Activation.
    """
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TimestepEmbedder(nn.Module):
    """
    Sinusoidal Timestep Embeddings.
    """
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # t: [B]
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# ---------- Relative position bias module for TSP ----------
class RelativePositionBias(nn.Module):
    """
    Compute a learnable relative position bias for point (coordinate) data.
    (Used for TSP, where coords provide useful geometric prior.)
    """
    def __init__(self, num_heads, hidden_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads)
        )

    def forward(self, coords):
        # coords: [B, N, 2]
        if coords is None:
            return None
        B, N, C = coords.shape
        assert C >= 2, "coords must have at least 2 dimensions (x,y)"
        # pairwise relative vector: [B, N, N, 2]
        rel = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = torch.sqrt((rel ** 2).sum(-1, keepdim=True) + 1e-6)
        inp = torch.cat([rel, dist], dim=-1)  # [B, N, N, 3]
        flat = inp.view(B * N * N, 3)
        out = self.mlp(flat)  # [B*N*N, num_heads]
        out = out.view(B, N, N, self.num_heads).permute(0, 3, 1, 2).contiguous()  # [B, num_heads, N, N]
        return out

# ---------- Attention variant supporting rel-pos bias + adj/structure bias ----------
class NodeCentricAttentionRelBias(nn.Module):
    """
    Lightweight Attention with optional relative position bias and adjacency/structure bias.
    Works for symmetric (TSP) and asymmetric (ATSP if adj_bias is asymmetric) cases.
    """
    def __init__(self, dim, num_heads, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        
        # learnable weights for structure bias and relative positional bias
        self.structure_bias_weight = nn.Parameter(torch.tensor(1.0))
        self.rel_bias_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, rel_pos_bias=None, adj_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D]

        # Build additive attention mask (float), equivalent to adding bias to attention logits.
        attn_mask = None

        if rel_pos_bias is not None:
            rb = rel_pos_bias
            if rb.dtype != q.dtype:
                rb = rb.to(dtype=q.dtype)
            attn_mask = rb * self.rel_bias_weight

        if adj_bias is not None:
            if adj_bias.dim() == 3:
                adj = adj_bias.unsqueeze(1)  # [B,1,N,N]
            elif adj_bias.dim() == 4:
                adj = adj_bias
            else:
                raise ValueError("adj_bias must be 3D [B,N,N] or 4D [B,1,N,N] or [B,num_heads,N,N]")

            if adj.shape[1] == 1 and self.num_heads != 1:
                adj = adj.expand(-1, self.num_heads, -1, -1)

            if adj.dtype != q.dtype:
                adj = adj.to(dtype=q.dtype)

            adj = adj * self.structure_bias_weight
            attn_mask = adj if attn_mask is None else (attn_mask + adj)

        # SDPA will dispatch to FlashAttention / memory-efficient kernels when available.
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)
        return x

# ---------- Transformer block ----------
class RAGenCOTransformerBlockRel(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = NodeCentricAttentionRelBias(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio * 2 / 3)) 

    def forward(self, x, rel_pos_bias=None, adj_bias=None):
        x = x + self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, adj_bias=adj_bias)
        x = x + self.mlp(self.norm2(x))
        return x

# ---------- Main Transformer ----------
class RAGenCOTransformer(nn.Module):
    """
    Lightweight denoising backbone used by RAGenCO.

    The model predicts task logits; task-specific projected control states and
    best-of-B branching are handled by the Lightning modules.
    """
    def __init__(
        self, 
        input_dim=2, 
        embed_dim=256, 
        depth=16, 
        num_heads=8, 
        mlp_ratio=4.0,
        edge_asymmetric: bool = False,
        use_cvrp_bias: bool = False,   # NEW: optional CVRP bias
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.edge_asymmetric = edge_asymmetric
        self.use_cvrp_bias = use_cvrp_bias
        
        self.node_embed = nn.Linear(input_dim, embed_dim)
        self.time_embed = TimestepEmbedder(embed_dim)

        # For TSP (coordinate-based), compute pairwise rel-pos bias via coords MLP
        if not self.edge_asymmetric:
            self.rel_pos_module = RelativePositionBias(num_heads=num_heads, hidden_dim=min(128, embed_dim))
            self.blocks = nn.ModuleList([
                RAGenCOTransformerBlockRel(embed_dim, num_heads, mlp_ratio)
                for _ in range(depth)
            ])
            # no cost-derived bias in TSP branch
            self.use_cost_bias = False
        else:
            # For ATSP: learned asymmetric relative bias from cost matrix
            self.blocks = nn.ModuleList([
                RAGenCOTransformerBlockRel(embed_dim, num_heads, mlp_ratio)
                for _ in range(depth)
            ])
            bias_hidden = min(128, embed_dim)
            self.rel_bias_mlp = nn.Sequential(
                nn.Linear(3, bias_hidden),
                nn.SiLU(),
                nn.Linear(bias_hidden, num_heads)
            )
            nn.init.zeros_(self.rel_bias_mlp[-1].weight)
            nn.init.zeros_(self.rel_bias_mlp[-1].bias)
            self.rel_bias_scale = nn.Parameter(torch.tensor(0.01))
            self.use_cost_bias = True

        self.norm = RMSNorm(embed_dim)

        # CVRP bias MLP (optional) - maps edge-level CVRP-specific features to per-head biases
        # Input features per edge: [demand_i, demand_j, demand_diff, is_depot_j, distance_ij]
        if self.use_cvrp_bias:
            bias_hidden = min(128, embed_dim)
            self.cvrp_bias_mlp = nn.Sequential(
                nn.Linear(5, bias_hidden),
                nn.SiLU(),
                nn.Linear(bias_hidden, num_heads)
            )
            # small initialization (so disabling effect early)
            nn.init.zeros_(self.cvrp_bias_mlp[-1].weight)
            nn.init.zeros_(self.cvrp_bias_mlp[-1].bias)
            self.cvrp_bias_scale = nn.Parameter(torch.tensor(0.01))

        # heads for edge output remain (separate src/dst for asymmetric case)
        if self.edge_asymmetric:
            self.head_proj_edge_src = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.head_proj_edge_dst = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.head_proj_edge = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        
        # node-level head for MIS/node-target tasks
        self.head_proj_node = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 1)
        )

    def _compute_cost_derived_bias(self, x, x_t_adj=None, eps=1e-6, clamp_val=10.0):
        """
        Compute asymmetric per-head relative bias from cost matrix robustly.
        """
        B, N, F = x.shape
        cost_mat = None
        if F == N:
            cost_mat = x
        if cost_mat is None and (x_t_adj is not None) and (x_t_adj.dim() == 3) and (x_t_adj.shape[1] == N and x_t_adj.shape[2] == N):
            cost_mat = x_t_adj
        if cost_mat is None:
            return None

        cost_mat = cost_mat.float()
        flat = cost_mat.view(B, -1)
        mean = flat.mean(dim=1).view(B, 1, 1)
        std = flat.std(dim=1).view(B, 1, 1).clamp(min=eps)
        norm_cost = (cost_mat - mean) / std  # [B, N, N]
        norm_cost = torch.nan_to_num(norm_cost, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
        norm_cost = norm_cost.clamp(-clamp_val, clamp_val)

        cost_trans = norm_cost.transpose(1, 2)
        cost_diff = norm_cost - cost_trans
        cost_feats = torch.stack([norm_cost, cost_trans, cost_diff], dim=-1)  # [B,N,N,3]

        Bn, Nn, Nn2, Ff = cost_feats.shape
        flat_feats = cost_feats.view(Bn * Nn * Nn2, Ff)  # [B*N*N, 3]
        out = self.rel_bias_mlp(flat_feats)  # [B*N*N, num_heads]
        num_heads = out.shape[-1]
        out = out.view(Bn, Nn, Nn2, num_heads).permute(0, 3, 1, 2).contiguous()  # [B, num_heads, N, N]

        rel_bias = out * self.rel_bias_scale
        rel_bias = torch.nan_to_num(rel_bias, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
        rel_bias = rel_bias.clamp(-clamp_val, clamp_val)
        return rel_bias

    def _compute_cvrp_bias(self, x, coords=None, eps=1e-6, clamp_val=10.0):
        """
        Compute per-head CVRP-specific bias from node features.
        Input node features x: [B, N, F] with F >= 4 (x,y,demand_norm,is_depot)
        coords: [B, N, 2] optionally used to compute distances
        Returns:
            cvrp_bias: [B, num_heads, N, N] or None
        Edge features used: [d_i, d_j, d_i - d_j, is_depot_j, dist_ij]
        """
        if (not self.use_cvrp_bias) or (self.cvrp_bias_mlp is None):
            return None
        B, N, F = x.shape
        if F < 4:
            return None

        # Extract demands and depot flag
        demand = x[..., 2].float()  # [B, N]
        is_depot = x[..., 3].float()  # [B, N]

        # compute pairwise features
        # demand_i, demand_j
        di = demand.unsqueeze(2).repeat(1, 1, N)  # [B, N, N]
        dj = demand.unsqueeze(1).repeat(1, N, 1)  # [B, N, N]
        d_diff = di - dj

        # is_depot_j
        is_depot_j = is_depot.unsqueeze(1).repeat(1, N, 1)  # [B, N, N]

        # distance if coords provided
        if coords is not None:
            # coords: [B, N, 2]
            rel = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B,N,N,2]
            dist = torch.sqrt((rel ** 2).sum(-1) + 1e-6)  # [B,N,N]
        else:
            dist = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)

        # Stack features -> [B, N, N, 5]
        feats = torch.stack([di, dj, d_diff, is_depot_j, dist], dim=-1)
        Bn, Nn, Nn2, Ff = feats.shape
        flat_feats = feats.view(Bn * Nn * Nn2, Ff)  # [B*N*N, 5]
        out = self.cvrp_bias_mlp(flat_feats)  # [B*N*N, num_heads]
        num_heads = out.shape[-1]
        out = out.view(Bn, Nn, Nn2, num_heads).permute(0, 3, 1, 2).contiguous()  # [B, num_heads, N, N]

        rel_bias = out * self.cvrp_bias_scale
        rel_bias = torch.nan_to_num(rel_bias, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
        rel_bias = rel_bias.clamp(-clamp_val, clamp_val)
        return rel_bias

    def forward(self, x, t, x_t_adj=None, target_type="edge"):
        """
        Args:
            x: Node features [B, N, F]. For CVRP: F >= 4 (x,y,demand_norm,is_depot)
            t: Timesteps [B]
            x_t_adj: Structure bias or noisy adjacency (if available) [B, N, N]
            target_type: "edge" (default) or "node"
        """
        B, N, _ = x.shape
        coords = None
        if (not self.edge_asymmetric) and (x.shape[-1] >= 2):
            coords = x[..., :2]
        else:
            coords = None

        h = self.node_embed(x)
        t_emb = self.time_embed(t)
        h = h + t_emb.unsqueeze(1)

        # compute biases
        if not self.edge_asymmetric:
            rel_bias = self.rel_pos_module(coords) if coords is not None else None

            # compute optional CVRP bias and combine with x_t_adj
            cvrp_bias = None
            if self.use_cvrp_bias:
                cvrp_bias = self._compute_cvrp_bias(x, coords=coords)
            # Build adj_bias passed into blocks:
            # - x_t_adj may be None or [B,N,N] probabilities; convert to [B,1,N,N] for addition if needed
            adj_bias_for_blocks = None
            if (x_t_adj is not None) and (x_t_adj.dim() == 3):
                adj_bias_for_blocks = x_t_adj.unsqueeze(1)  # [B,1,N,N]
            elif (x_t_adj is not None) and (x_t_adj.dim() == 4):
                adj_bias_for_blocks = x_t_adj

            # If cvrp_bias exists, add into adj_bias_for_blocks as per-head bias
            if cvrp_bias is not None:
                if adj_bias_for_blocks is None:
                    adj_bias_for_blocks = cvrp_bias
                else:
                    # If adj_bias_for_blocks has shape [B,1,N,N], broadcast and add
                    if adj_bias_for_blocks.shape[1] == 1:
                        adj_bias_for_blocks = adj_bias_for_blocks + cvrp_bias
                    else:
                        adj_bias_for_blocks = adj_bias_for_blocks + cvrp_bias

            for block in self.blocks:
                h = block(h, rel_pos_bias=rel_bias, adj_bias=adj_bias_for_blocks)
        else:
            # ATSP: compute learned asymmetric bias from cost matrix
            adj_cost_bias = self._compute_cost_derived_bias(x, x_t_adj=x_t_adj)
            for block in self.blocks:
                h = block(h, rel_pos_bias=None, adj_bias=adj_cost_bias)

        h = self.norm(h)

        if target_type == "edge":
            if self.edge_asymmetric:
                h_src = self.head_proj_edge_src(h)  # [B, N, D]
                h_dst = self.head_proj_edge_dst(h)  # [B, N, D]
            else:
                h_src = self.head_proj_edge(h)  # [B, N, D]
                h_dst = h_src
            logits = torch.matmul(h_src, h_dst.transpose(1, 2))
            logits = logits * (self.embed_dim ** -0.5)
            return logits
        elif target_type == "node":
            node_logit = self.head_proj_node(h)  # [B, N, 1]
            node_logit = node_logit.squeeze(-1)  # [B, N]
            return node_logit
        else:
            raise ValueError(f"Unsupported target_type: {target_type}")

def get_ragenco_model(param_args, input_dim=2, edge_asymmetric: bool = False, use_cvrp_bias: bool = False):
    """Factory function."""
    return RAGenCOTransformer(
        input_dim=input_dim,
        embed_dim=getattr(param_args, 'hidden_dim', 256),
        depth=getattr(param_args, 'num_layers', 16),
        num_heads=getattr(param_args, 'num_heads', 8),
        mlp_ratio=4.0,
        edge_asymmetric=edge_asymmetric,
        use_cvrp_bias=use_cvrp_bias,
    )
