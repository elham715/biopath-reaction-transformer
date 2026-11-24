"""Unit tests for MetabolicGraph and Shortest-Path Distance computation."""

import numpy as np
import pytest
from src.metabolic_graph import MetabolicGraph
from src.config import NUM_METABOLITES

def test_metabolic_graph_initialization():
    graph = MetabolicGraph(max_spd=8)
    assert graph.num_nodes == NUM_METABOLITES
    assert graph.adj_matrix.shape == (NUM_METABOLITES, NUM_METABOLITES)

def test_spd_properties():
    graph = MetabolicGraph(max_spd=8)
    spd = graph.get_spd_matrix()
    
    # Diagonal must be 0
    assert np.all(np.diag(spd) == 0)
    # Symmetry
    assert np.all(spd == spd.T)
    # Direct reactions have distance 1 (e.g. Glucose -> Glucose_6P)
    u = graph.metabolite_to_id["Glucose"]
    v = graph.metabolite_to_id["Glucose_6P"]
    assert spd[u, v] == 1

def test_tca_pathway_distances():
    graph = MetabolicGraph(max_spd=8)
    spd = graph.get_spd_matrix()
    
    citrate_id = graph.metabolite_to_id["Citrate"]
    isocitrate_id = graph.metabolite_to_id["Isocitrate"]
    succinate_id = graph.metabolite_to_id["Succinate"]
    
    assert spd[citrate_id, isocitrate_id] == 1
    # Succinate is downstream of Citrate
    assert 1 < spd[citrate_id, succinate_id] <= 8
