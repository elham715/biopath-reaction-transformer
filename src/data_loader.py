"""Data loading and metabolic cascade benchmark generation."""

import os
import csv
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .config import METABOLITE_NAMES, METABOLIC_PATHWAYS, CANONICAL_REACTIONS, MODEL_CONFIG, TRAINING_CONFIG
from .metabolic_tokenizer import MetabolicTokenizer

def generate_metabolic_benchmark(
    num_samples: int = 400,
    random_seed: int = 42,
    csv_path: Optional[str] = None
) -> List[Dict]:
    """Generates synthetic biochemical reaction cascades with flux and oncometabolite targets."""
    np.random.seed(random_seed)
    records = []
    
    # 1. Glycolysis Subgraph
    glycolysis_nodes = [
        "Glucose", "Glucose_6P", "Fructose_6P", "Fructose_1_6BP",
        "Glyceraldehyde_3P", "Phosphoenolpyruvate", "Pyruvate", "Lactate"
    ]
    # 2. TCA Subgraph
    tca_nodes = [
        "Pyruvate", "Acetyl_CoA", "Citrate", "Isocitrate", "Alpha_Ketoglutarate",
        "Succinyl_CoA", "Succinate", "Fumarate", "Malate", "Oxaloacetate"
    ]
    # 3. Oncometabolite Shunt Subgraph
    onco_nodes = [
        "Glutamine", "Glutamate", "Alpha_Ketoglutarate", "2_Hydroxyglutarate",
        "Succinyl_CoA", "Succinate", "Fumarate", "Malate"
    ]
    # 4. Urea Cycle Subgraph
    urea_nodes = [
        "Carbamoyl_Phosphate", "Citrulline", "Argininosuccinate", "Arginine", "Urea"
    ]
    
    pathways = [
        ("Glycolysis", glycolysis_nodes, 4.2, 0.2),
        ("TCA_Cycle", tca_nodes, 5.8, 0.1),
        ("Oncometabolite_Shunt", onco_nodes, 2.1, 3.4),  # High oncometabolite surge!
        ("Urea_Cycle", urea_nodes, 3.5, 0.05)
    ]
    
    while len(records) < num_samples:
        p_name, p_nodes, base_flux, base_onco = pathways[np.random.choice(len(pathways))]
        
        # Subsample or perturb nodes
        sub_len = np.random.randint(max(3, len(p_nodes) - 2), len(p_nodes) + 1)
        # Preserve order
        chosen_indices = sorted(np.random.choice(len(p_nodes), size=sub_len, replace=False))
        active_metabs = [p_nodes[i] for i in chosen_indices]
        
        # Flux variation
        flux = float(base_flux * (len(active_metabs) / len(p_nodes)) + np.random.normal(0, 0.2))
        onco = float(base_onco + (0.5 if "2_Hydroxyglutarate" in active_metabs or "Lactate" in active_metabs else 0.0) + np.random.normal(0, 0.15))
        
        records.append({
            "pathway": p_name,
            "metabolites": active_metabs,
            "flux": max(0.1, flux),
            "oncometabolite_surge": max(0.0, onco)
        })
        
    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pathway", "flux", "oncometabolite_surge", "metabolites"])
            for r in records:
                writer.writerow([r["pathway"], f"{r['flux']:.4f}", f"{r['oncometabolite_surge']:.4f}", ";".join(r["metabolites"])])
                
    return records


class MetabolicPathwayDataset(Dataset):
    """PyTorch Dataset for metabolic graphs."""
    
    def __init__(self, records: List[Dict], tokenizer: MetabolicTokenizer):
        self.records = records
        self.tokenizer = tokenizer
        
        self.encoded = []
        for r in records:
            ids, mask, spd = self.tokenizer.encode_pathway(r["metabolites"])
            flux_target = torch.tensor([r["flux"]], dtype=torch.float32)
            onco_target = torch.tensor([r["oncometabolite_surge"]], dtype=torch.float32)
            self.encoded.append((ids, mask, spd, flux_target, onco_target))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoded[idx]


def create_dataloaders(
    records: List[Dict],
    tokenizer: MetabolicTokenizer,
    batch_size: int = 32,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    np.random.seed(seed)
    indices = np.random.permutation(len(records))
    
    n_total = len(records)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_records = [records[i] for i in indices[:n_train]]
    val_records = [records[i] for i in indices[n_train:n_train + n_val]]
    test_records = [records[i] for i in indices[n_train + n_val:]]
    
    train_loader = DataLoader(MetabolicPathwayDataset(train_records, tokenizer), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MetabolicPathwayDataset(val_records, tokenizer), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(MetabolicPathwayDataset(test_records, tokenizer), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
