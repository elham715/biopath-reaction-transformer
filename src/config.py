"""Configuration and biochemical registry for BioPath Metabolic Reaction Transformer."""

from typing import Dict, List, Tuple, Any

# ============================================================================
# 1. Core Human Metabolite Registry
# ============================================================================
METABOLITE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Glucose": {"formula": "C6H12O6", "pathway": "Glycolysis", "carbon_count": 6},
    "Glucose_6P": {"formula": "C6H11O9P", "pathway": "Glycolysis", "carbon_count": 6},
    "Fructose_6P": {"formula": "C6H11O9P", "pathway": "Glycolysis", "carbon_count": 6},
    "Fructose_1_6BP": {"formula": "C6H10O12P2", "pathway": "Glycolysis", "carbon_count": 6},
    "Glyceraldehyde_3P": {"formula": "C3H5O6P", "pathway": "Glycolysis", "carbon_count": 3},
    "Phosphoenolpyruvate": {"formula": "C3H2O6P", "pathway": "Glycolysis", "carbon_count": 3},
    "Pyruvate": {"formula": "C3H3O3", "pathway": "Glycolysis_TCA_Hub", "carbon_count": 3},
    "Lactate": {"formula": "C3H5O3", "pathway": "Warburg_Glycolysis", "carbon_count": 3},
    "Acetyl_CoA": {"formula": "C23H38N7O17P3S", "pathway": "TCA_Cycle", "carbon_count": 2},
    "Citrate": {"formula": "C6H5O7", "pathway": "TCA_Cycle", "carbon_count": 6},
    "Isocitrate": {"formula": "C6H5O7", "pathway": "TCA_Cycle", "carbon_count": 6},
    "Alpha_Ketoglutarate": {"formula": "C5H4O5", "pathway": "TCA_Cycle", "carbon_count": 5},
    "2_Hydroxyglutarate": {"formula": "C5H6O5", "pathway": "Oncometabolite_Shunt", "carbon_count": 5},
    "Succinyl_CoA": {"formula": "C25H40N7O19P3S", "pathway": "TCA_Cycle", "carbon_count": 4},
    "Succinate": {"formula": "C4H4O4", "pathway": "TCA_Cycle", "carbon_count": 4},
    "Fumarate": {"formula": "C4H2O4", "pathway": "TCA_Cycle", "carbon_count": 4},
    "Malate": {"formula": "C4H4O5", "pathway": "TCA_Cycle", "carbon_count": 4},
    "Oxaloacetate": {"formula": "C4H2O5", "pathway": "TCA_Cycle", "carbon_count": 4},
    "Glutamine": {"formula": "C5H10N2O3", "pathway": "Glutaminolysis", "carbon_count": 5},
    "Glutamate": {"formula": "C5H8NO4", "pathway": "Glutaminolysis", "carbon_count": 5},
    "Carbamoyl_Phosphate": {"formula": "CH2NO5P", "pathway": "Urea_Cycle", "carbon_count": 1},
    "Citrulline": {"formula": "C6H13N3O3", "pathway": "Urea_Cycle", "carbon_count": 6},
    "Argininosuccinate": {"formula": "C10H18N4O6", "pathway": "Urea_Cycle", "carbon_count": 10},
    "Arginine": {"formula": "C6H14N4O2", "pathway": "Urea_Cycle", "carbon_count": 6},
    "Urea": {"formula": "CH4N2O", "pathway": "Urea_Cycle", "carbon_count": 1}
}

METABOLITE_NAMES = list(METABOLITE_REGISTRY.keys())
NUM_METABOLITES = len(METABOLITE_NAMES)

# ============================================================================
# 2. Canonical Metabolic Enzyme Registry (Official Enzyme Commission EC Numbers)
# ============================================================================
ENZYME_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Hexokinase": {"ec": "2.7.1.1", "gene": "HK2", "role": "Glycolysis Trapping"},
    "Phosphofructokinase_1": {"ec": "2.7.1.11", "gene": "PFKM", "role": "Glycolytic Pacemaker"},
    "Pyruvate_Kinase": {"ec": "2.7.1.40", "gene": "PKM2", "role": "Warburg Switch"},
    "Lactate_Dehydrogenase": {"ec": "1.1.1.27", "gene": "LDHA", "role": "Aerobic Glycolysis Regeneration"},
    "Pyruvate_Dehydrogenase": {"ec": "1.2.4.1", "gene": "PDHA1", "role": "Mitochondrial Entry"},
    "Citrate_Synthase": {"ec": "2.3.3.1", "gene": "CS", "role": "TCA Condensation"},
    "Aconitase": {"ec": "4.2.1.3", "gene": "ACO2", "role": "Isomerization"},
    "IDH1_WildType": {"ec": "1.1.1.42", "gene": "IDH1", "role": "Isocitrate Oxidative Decarboxylation"},
    "IDH1_R132H_Mutant": {"ec": "1.1.1.42-mut", "gene": "IDH1", "role": "Neomorphic Oncometabolite 2-HG Generator"},
    "Alpha_KGDH": {"ec": "1.2.4.2", "gene": "OGDH", "role": "Rate-Limiting Decarboxylation"},
    "Succinate_Dehydrogenase": {"ec": "1.3.5.1", "gene": "SDHA", "role": "Complex II / Tumor Suppressor"},
    "Fumarase": {"ec": "4.2.1.2", "gene": "FH", "role": "Hydration / Tumor Suppressor"},
    "Malate_Dehydrogenase": {"ec": "1.1.1.37", "gene": "MDH2", "role": "Oxaloacetate Regeneration"},
    "Glutaminase": {"ec": "3.5.1.2", "gene": "GLS", "role": "Glutamine Anaplerosis"},
    "Argininosuccinate_Synthase": {"ec": "6.3.4.5", "gene": "ASS1", "role": "Urea Precursor Condensation"},
    "Argininosuccinate_Lyase": {"ec": "4.3.2.1", "gene": "ASL", "role": "Fumarate & Arginine Cleavage"}
}

ENZYME_NAMES = list(ENZYME_REGISTRY.keys())
NUM_ENZYMES = len(ENZYME_NAMES)

# ============================================================================
# 3. Canonical Biochemical Reactions & Directed Topologies
# ============================================================================
CANONICAL_REACTIONS: List[Dict[str, Any]] = [
    # Glycolysis Cascade
    {"substrate": "Glucose", "product": "Glucose_6P", "enzyme": "Hexokinase", "cofactor": "ATP->ADP", "delta_G0": -16.7},
    {"substrate": "Glucose_6P", "product": "Fructose_6P", "enzyme": "Hexokinase", "cofactor": "None", "delta_G0": +1.7},
    {"substrate": "Fructose_6P", "product": "Fructose_1_6BP", "enzyme": "Phosphofructokinase_1", "cofactor": "ATP->ADP", "delta_G0": -14.2},
    {"substrate": "Fructose_1_6BP", "product": "Glyceraldehyde_3P", "enzyme": "Phosphofructokinase_1", "cofactor": "None", "delta_G0": +23.8},
    {"substrate": "Glyceraldehyde_3P", "product": "Phosphoenolpyruvate", "enzyme": "Pyruvate_Kinase", "cofactor": "NAD+->NADH", "delta_G0": -3.2},
    {"substrate": "Phosphoenolpyruvate", "product": "Pyruvate", "enzyme": "Pyruvate_Kinase", "cofactor": "ADP->ATP", "delta_G0": -31.4},
    {"substrate": "Pyruvate", "product": "Lactate", "enzyme": "Lactate_Dehydrogenase", "cofactor": "NADH->NAD+", "delta_G0": -25.1},
    
    # TCA Cycle Cascade
    {"substrate": "Pyruvate", "product": "Acetyl_CoA", "enzyme": "Pyruvate_Dehydrogenase", "cofactor": "NAD+->NADH", "delta_G0": -33.4},
    {"substrate": "Acetyl_CoA", "product": "Citrate", "enzyme": "Citrate_Synthase", "cofactor": "CoA", "delta_G0": -31.5},
    {"substrate": "Citrate", "product": "Isocitrate", "enzyme": "Aconitase", "cofactor": "None", "delta_G0": +6.3},
    {"substrate": "Isocitrate", "product": "Alpha_Ketoglutarate", "enzyme": "IDH1_WildType", "cofactor": "NADP+->NADPH", "delta_G0": -8.4},
    {"substrate": "Alpha_Ketoglutarate", "product": "Succinyl_CoA", "enzyme": "Alpha_KGDH", "cofactor": "NAD+->NADH", "delta_G0": -30.1},
    {"substrate": "Succinyl_CoA", "product": "Succinate", "enzyme": "Alpha_KGDH", "cofactor": "GDP->GTP", "delta_G0": -2.9},
    {"substrate": "Succinate", "product": "Fumarate", "enzyme": "Succinate_Dehydrogenase", "cofactor": "FAD->FADH2", "delta_G0": +0.0},
    {"substrate": "Fumarate", "product": "Malate", "enzyme": "Fumarase", "cofactor": "H2O", "delta_G0": -3.8},
    {"substrate": "Malate", "product": "Oxaloacetate", "enzyme": "Malate_Dehydrogenase", "cofactor": "NAD+->NADH", "delta_G0": +29.7},
    
    # Oncometabolite Shunts (Cancer Pathology)
    {"substrate": "Alpha_Ketoglutarate", "product": "2_Hydroxyglutarate", "enzyme": "IDH1_R132H_Mutant", "cofactor": "NADPH->NADP+", "delta_G0": -18.2},
    {"substrate": "Glutamine", "product": "Glutamate", "enzyme": "Glutaminase", "cofactor": "H2O", "delta_G0": -12.5},
    {"substrate": "Glutamate", "product": "Alpha_Ketoglutarate", "enzyme": "Glutaminase", "cofactor": "NAD+->NADH", "delta_G0": -5.1},
    
    # Urea Cycle Cascade
    {"substrate": "Carbamoyl_Phosphate", "product": "Citrulline", "enzyme": "Argininosuccinate_Synthase", "cofactor": "None", "delta_G0": -14.0},
    {"substrate": "Citrulline", "product": "Argininosuccinate", "enzyme": "Argininosuccinate_Synthase", "cofactor": "ATP->AMP", "delta_G0": -12.8},
    {"substrate": "Argininosuccinate", "product": "Arginine", "enzyme": "Argininosuccinate_Lyase", "cofactor": "Yields Fumarate", "delta_G0": +11.7},
    {"substrate": "Arginine", "product": "Urea", "enzyme": "Argininosuccinate_Lyase", "cofactor": "H2O", "delta_G0": -15.5}
]

# ============================================================================
# 4. Oncometabolite Cancer Phenotypes
# ============================================================================
ONCOMETABOLITE_SHUNTS: Dict[str, Dict[str, Any]] = {
    "IDH1_R132H": {
        "alteration": "IDH1 Arg132His missense mutation",
        "oncometabolite": "2_Hydroxyglutarate (D-2-HG)",
        "mechanism": "Neomorphic NADPH-dependent reduction of alpha-KG to (R)-2-hydroxyglutarate; competitive inhibitor of alpha-KG-dependent dioxygenases (TET2, KDM)",
        "cancer_types": "Glioblastoma (80%), Acute Myeloid Leukemia (20%), Cholangiocarcinoma (15%)"
    },
    "SDH_Deficiency": {
        "alteration": "SDHA/B/C/D loss-of-function mutation",
        "oncometabolite": "Succinate",
        "mechanism": "Blockade of Complex II; succinate accumulation spills into cytosol, inhibiting prolyl hydroxylases (PHD), stabilizing HIF-1alpha (Pseudohypoxia)",
        "cancer_types": "Paraganglioma, Pheochromocytoma, Gastrointestinal Stromal Tumors (GIST)"
    },
    "FH_Deficiency": {
        "alteration": "Fumarate Hydratase biallelic inactivation",
        "oncometabolite": "Fumarate",
        "mechanism": "Fumarate hydratase loss; excess fumarate reacts with cysteine residues (succination), inactivating KEAP1 and activating NRF2 antioxidant pathway",
        "cancer_types": "Hereditary Leiomyomatosis and Renal Cell Cancer (HLRCC)"
    }
}

METABOLIC_PATHWAYS = ["Glycolysis", "TCA_Cycle", "Oncometabolite_Shunt", "Urea_Cycle"]

# ============================================================================
# 5. Model Architecture & Training Configurations
# ============================================================================
MODEL_CONFIG: Dict[str, Any] = {
    "vocab_size": NUM_METABOLITES + 10,  # 25 metabolites + special tokens
    "num_metabolites": NUM_METABOLITES + 10,
    "num_enzymes": NUM_ENZYMES,
    "max_graph_nodes": 32,
    "d_model": 128,
    "n_heads": 4,
    "num_layers": 3,
    "dim_feedforward": 256,
    "dropout": 0.10,
    "max_spd_distance": 8,     # Shortest-Path Distance truncation threshold
    "spd_bias_dim": 8,
}

TRAINING_CONFIG: Dict[str, Any] = {
    "batch_size": 32,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "num_epochs": 15,
    "seed": 42,
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "test_ratio": 0.10,
}
