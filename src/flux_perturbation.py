"""Metabolic enzyme perturbation and oncometabolite shift simulator."""

from typing import Dict, List, Tuple, Any, Optional
import torch
import numpy as np
from .metabolic_tokenizer import MetabolicTokenizer
from .biopath_transformer import BioPathTransformerModel
from .config import CANONICAL_REACTIONS, ONCOMETABOLITE_SHUNTS

class MetabolicFluxPerturbation:
    """Simulates enzyme knockouts, oncometabolite shunts, and pharmacological inhibition."""
    
    def __init__(self, model: BioPathTransformerModel, tokenizer: MetabolicTokenizer, device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def simulate_idh1_r132h(self) -> Dict[str, Any]:
        """Simulates the canonical IDH1 R132H oncogenic mutation.
        
        Wild-Type Pathway:
          Isocitrate -> Alpha-Ketoglutarate -> Succinyl-CoA -> Succinate -> Fumarate -> Malate -> Oxaloacetate
        Mutant Neomorphic Shunt:
          Isocitrate -> Alpha-Ketoglutarate -> 2-Hydroxyglutarate (D-2-HG)
        """
        # Wild-type TCA sequence
        wt_metabolites = [
            "Citrate", "Isocitrate", "Alpha_Ketoglutarate",
            "Succinyl_CoA", "Succinate", "Fumarate", "Malate", "Oxaloacetate"
        ]
        # IDH1 mutant sequence with oncometabolite shunt
        mut_metabolites = [
            "Citrate", "Isocitrate", "Alpha_Ketoglutarate", "2_Hydroxyglutarate",
            "Succinyl_CoA", "Succinate", "Fumarate", "Malate", "Oxaloacetate"
        ]
        
        with torch.no_grad():
            # Encode WT
            wt_ids, wt_mask, wt_spd = self.tokenizer.encode_pathway(wt_metabolites)
            wt_preds = self.model(
                wt_ids.unsqueeze(0).to(self.device),
                wt_spd.unsqueeze(0).to(self.device),
                wt_mask.unsqueeze(0).to(self.device)
            )
            
            # Encode Mutant
            mut_ids, mut_mask, mut_spd = self.tokenizer.encode_pathway(mut_metabolites)
            mut_preds = self.model(
                mut_ids.unsqueeze(0).to(self.device),
                mut_spd.unsqueeze(0).to(self.device),
                mut_mask.unsqueeze(0).to(self.device)
            )
            
        wt_flux = wt_preds["flux"].item()
        wt_2hg = wt_preds["oncometabolite_surge"].item()
        
        mut_flux = mut_preds["flux"].item()
        mut_2hg = mut_preds["oncometabolite_surge"].item()
        
        return {
            "alteration": "IDH1_R132H",
            "wt_flux": wt_flux,
            "mut_flux": mut_flux,
            "flux_inhibition_pct": max(0.0, float((wt_flux - mut_flux) / max(abs(wt_flux), 1e-4) * 100)),
            "wt_2hg_level": wt_2hg,
            "mut_2hg_level": mut_2hg,
            "oncometabolite_fold_surge": float(np.exp(max(0.0, mut_2hg - wt_2hg))),
            "clinical_annotation": ONCOMETABOLITE_SHUNTS["IDH1_R132H"]
        }

    def simulate_sdh_deficiency(self) -> Dict[str, Any]:
        """Simulates Succinate Dehydrogenase (Complex II) inactivation."""
        # Intact TCA
        intact_tca = ["Succinyl_CoA", "Succinate", "Fumarate", "Malate", "Oxaloacetate"]
        # SDH blocked (Succinate accumulation, downstream depletion)
        blocked_tca = ["Succinyl_CoA", "Succinate", "Malate", "Oxaloacetate"]
        
        with torch.no_grad():
            w_ids, w_mask, w_spd = self.tokenizer.encode_pathway(intact_tca)
            w_preds = self.model(w_ids.unsqueeze(0).to(self.device), w_spd.unsqueeze(0).to(self.device), w_mask.unsqueeze(0).to(self.device))
            
            b_ids, b_mask, b_spd = self.tokenizer.encode_pathway(blocked_tca)
            b_preds = self.model(b_ids.unsqueeze(0).to(self.device), b_spd.unsqueeze(0).to(self.device), b_mask.unsqueeze(0).to(self.device))
            
        return {
            "alteration": "SDH_Deficiency",
            "intact_flux": w_preds["flux"].item(),
            "blocked_flux": b_preds["flux"].item(),
            "succinate_surge": float(np.exp(b_preds["oncometabolite_surge"].item() - w_preds["oncometabolite_surge"].item())),
            "clinical_annotation": ONCOMETABOLITE_SHUNTS["SDH_Deficiency"]
        }
