"""Graphormer-style Shortest-Path Distance (SPD) Multi-Head Attention."""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShortestPathDistanceAttention(nn.Module):
    """Multi-Head Attention biased by topological Shortest-Path Distance (Ying et al., 2021).
    
    Attention formula:
      A_ij = Softmax( (q_i · k_j^T) / sqrt(d_k) + b_SPD(i, j) + Mask_ij )
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_spd: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_spd = max_spd
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Learnable spatial SPD bias table: (max_spd + 1) buckets across n_heads
        self.spd_bias_table = nn.Embedding(max_spd + 1, n_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        spd_matrix: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node feature tensor [B, N, d_model]
            spd_matrix: LongTensor of pairwise shortest path distances [B, N, N]
            attention_mask: BoolTensor [B, N] (True for valid nodes)
            
        Returns:
            out: Contextual node tensor [B, N, d_model]
            attn_weights: Attention probability maps [B, n_heads, N, N]
        """
        B, N, C = x.shape
        
        # Linear projections
        q = self.q_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, N, d_k]
        k = self.k_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, N, d_k]
        v = self.v_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, N, d_k]
        
        # Base dot product scores: Q @ K^T / sqrt(d_k) -> [B, h, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Topological SPD spatial bias: [B, N, N] -> [B, N, N, h] -> [B, h, N, N]
        # Clamp spd_matrix to valid range [0, max_spd]
        clamped_spd = spd_matrix.clamp(0, self.max_spd)
        spd_bias = self.spd_bias_table(clamped_spd).permute(0, 3, 1, 2)
        
        # Add spatial bias
        scores = scores + spd_bias
        
        # Mask out padding nodes
        if attention_mask is not None:
            # Mask shape: [B, 1, 1, N]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float("-inf"))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights_drop = self.dropout(attn_weights)
        
        # Context aggregation
        context = torch.matmul(attn_weights_drop, v)  # [B, h, N, d_k]
        context = context.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]
        
        return self.out_proj(context), attn_weights
