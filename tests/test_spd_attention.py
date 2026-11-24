"""Unit tests for Graphormer SPD-biased Attention."""

import torch
import pytest
from src.spd_attention import ShortestPathDistanceAttention

def test_spd_attention_forward():
    B = 4
    N = 16
    d_model = 64
    n_heads = 4
    max_spd = 8
    
    attn = ShortestPathDistanceAttention(d_model=d_model, n_heads=n_heads, max_spd=max_spd)
    
    x = torch.randn(B, N, d_model)
    spd_matrix = torch.randint(0, max_spd + 1, (B, N, N))
    mask = torch.ones((B, N), dtype=torch.bool)
    mask[:, -3:] = False  # Mask last 3 nodes
    
    out, weights = attn(x, spd_matrix, attention_mask=mask)
    
    assert out.shape == (B, N, d_model)
    assert weights.shape == (B, n_heads, N, N)
    assert not torch.isnan(out).any()
    
    # Attention weights over valid nodes sum to 1
    valid_sums = weights[:, :, :, :N-3].sum(dim=-1)
    assert torch.allclose(valid_sums[:, :, :N-3], torch.ones(B, n_heads, N-3), atol=1e-4)
