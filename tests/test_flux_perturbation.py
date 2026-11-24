"""Unit tests for metabolic perturbation engine."""

import torch
import pytest
from src.metabolic_tokenizer import MetabolicTokenizer
from src.biopath_transformer import BioPathTransformerModel
from src.flux_perturbation import MetabolicFluxPerturbation
from src.config import MODEL_CONFIG

def test_idh1_r132h_simulation():
    tokenizer = MetabolicTokenizer(max_nodes=16, max_spd=8)
    cfg = dict(MODEL_CONFIG)
    cfg["d_model"] = 32
    cfg["max_graph_nodes"] = 16
    cfg["n_heads"] = 2
    cfg["num_layers"] = 1
    
    model = BioPathTransformerModel(cfg)
    pert = MetabolicFluxPerturbation(model, tokenizer)
    
    res = pert.simulate_idh1_r132h()
    assert res["alteration"] == "IDH1_R132H"
    assert "wt_flux" in res
    assert "mut_flux" in res
    assert "oncometabolite_fold_surge" in res
    assert res["oncometabolite_fold_surge"] >= 1.0

def test_sdh_deficiency_simulation():
    tokenizer = MetabolicTokenizer(max_nodes=16, max_spd=8)
    cfg = dict(MODEL_CONFIG)
    cfg["d_model"] = 32
    cfg["max_graph_nodes"] = 16
    cfg["n_heads"] = 2
    cfg["num_layers"] = 1
    
    model = BioPathTransformerModel(cfg)
    pert = MetabolicFluxPerturbation(model, tokenizer)
    
    res = pert.simulate_sdh_deficiency()
    assert res["alteration"] == "SDH_Deficiency"
    assert "succinate_surge" in res
