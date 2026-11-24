"""Unit tests for BioPathTransformerModel."""

import torch
import pytest
from src.biopath_transformer import BioPathTransformerModel
from src.config import MODEL_CONFIG

def test_biopath_forward():
    B = 4
    N = 16
    cfg = dict(MODEL_CONFIG)
    cfg["d_model"] = 64
    cfg["max_graph_nodes"] = 16
    cfg["n_heads"] = 2
    cfg["num_layers"] = 2
    cfg["dim_feedforward"] = 128
    
    model = BioPathTransformerModel(cfg)
    
    token_ids = torch.randint(0, cfg["num_metabolites"], (B, N))
    spd_matrix = torch.randint(0, 9, (B, N, N))
    mask = torch.ones((B, N), dtype=torch.bool)
    mask[:, -4:] = False
    
    preds = model(token_ids, spd_matrix, attention_mask=mask)
    
    assert "flux" in preds
    assert "oncometabolite_surge" in preds
    assert preds["flux"].shape == (B, 1)
    assert preds["oncometabolite_surge"].shape == (B, 1)
    assert not torch.isnan(preds["flux"]).any()

def test_extract_attention():
    B = 2
    N = 12
    cfg = dict(MODEL_CONFIG)
    cfg["d_model"] = 32
    cfg["max_graph_nodes"] = 12
    cfg["n_heads"] = 2
    cfg["num_layers"] = 2
    
    model = BioPathTransformerModel(cfg)
    token_ids = torch.randint(0, cfg["num_metabolites"], (B, N))
    spd_matrix = torch.randint(0, 9, (B, N, N))
    mask = torch.ones((B, N), dtype=torch.bool)
    
    res = model.extract_attention(token_ids, spd_matrix, attention_mask=mask)
    assert "layer_attention_weights" in res
    assert res["layer_attention_weights"].shape == (2, B, 2, N, N)
