"""Multi-task CPU trainer for BioPath Metabolic Transformer."""

import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .biopath_transformer import BioPathTransformerModel
from .config import TRAINING_CONFIG

class BioPathModelTrainer:
    """Trains and evaluates the BioPath Transformer on CPU."""
    
    def __init__(
        self,
        model: BioPathTransformerModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or TRAINING_CONFIG
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 5e-4),
            weight_decay=self.config.get("weight_decay", 1e-4)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get("num_epochs", 15),
            eta_min=1e-6
        )
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_flux_spearman": [],
            "val_onco_spearman": []
        }
        self.best_score = -1.0
        self.best_model_weights = None

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for ids, mask, spd, flux_target, onco_target in self.train_loader:
            ids = ids.to(self.device)
            mask = mask.to(self.device)
            spd = spd.to(self.device)
            flux_target = flux_target.to(self.device)
            onco_target = onco_target.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(ids, spd, attention_mask=mask)
            
            loss_flux = self.criterion(preds["flux"], flux_target)
            loss_onco = self.criterion(preds["oncometabolite_surge"], onco_target)
            loss = loss_flux + loss_onco
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        return total_loss / max(n_batches, 1)

    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        pred_flux, true_flux = [], []
        pred_onco, true_onco = [], []
        n_batches = 0
        
        with torch.no_grad():
            for ids, mask, spd, flux_target, onco_target in loader:
                ids = ids.to(self.device)
                mask = mask.to(self.device)
                spd = spd.to(self.device)
                flux_target = flux_target.to(self.device)
                onco_target = onco_target.to(self.device)
                
                preds = self.model(ids, spd, attention_mask=mask)
                loss_flux = self.criterion(preds["flux"], flux_target)
                loss_onco = self.criterion(preds["oncometabolite_surge"], onco_target)
                loss = loss_flux + loss_onco
                
                total_loss += loss.item()
                n_batches += 1
                
                pred_flux.extend(preds["flux"].squeeze(-1).cpu().numpy().tolist())
                true_flux.extend(flux_target.squeeze(-1).cpu().numpy().tolist())
                
                pred_onco.extend(preds["oncometabolite_surge"].squeeze(-1).cpu().numpy().tolist())
                true_onco.extend(onco_target.squeeze(-1).cpu().numpy().tolist())
                
        pf, tf = np.array(pred_flux), np.array(true_flux)
        po, to = np.array(pred_onco), np.array(true_onco)
        
        rho_flux, _ = spearmanr(pf, tf) if len(pf) > 1 and np.std(pf) > 1e-5 else (0.0, 0.0)
        rho_onco, _ = spearmanr(po, to) if len(po) > 1 and np.std(po) > 1e-5 else (0.0, 0.0)
        
        return {
            "loss": float(total_loss / max(n_batches, 1)),
            "flux_spearman": float(rho_flux),
            "onco_spearman": float(rho_onco),
            "pred_flux": pf,
            "true_flux": tf,
            "pred_onco": po,
            "true_onco": to
        }

    def fit(self, verbose: bool = True) -> Dict[str, Any]:
        num_epochs = self.config.get("num_epochs", 15)
        start_time = time.time()
        
        if verbose:
            print("=" * 72)
            print(f"🚀 Training BioPath Transformer ({num_epochs} Epochs on CPU)")
            print("=" * 72)
            
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()
            elapsed = time.time() - t0
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_flux_spearman"].append(val_metrics["flux_spearman"])
            self.history["val_onco_spearman"].append(val_metrics["onco_spearman"])
            
            avg_score = (val_metrics["flux_spearman"] + val_metrics["onco_spearman"]) / 2.0
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_model_weights = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                
            if verbose:
                print(
                    f"Epoch {epoch:02d}/{num_epochs:02d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Flux ρ: {val_metrics['flux_spearman']:.4f} | "
                    f"Onco ρ: {val_metrics['onco_spearman']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )
                
        total_time = time.time() - start_time
        if self.best_model_weights:
            self.model.load_state_dict(self.best_model_weights)
            
        test_metrics = self.evaluate(self.test_loader)
        if verbose:
            print("=" * 72)
            print(f"🏁 Training Complete in {total_time:.1f}s")
            print(f"📊 Test Results -> Loss: {test_metrics['loss']:.4f} | "
                  f"Flux ρ: {test_metrics['flux_spearman']:.4f} | Onco ρ: {test_metrics['onco_spearman']:.4f}")
            print("=" * 72)
            
        return {
            "history": self.history,
            "test_metrics": test_metrics,
            "total_time_seconds": total_time
        }
