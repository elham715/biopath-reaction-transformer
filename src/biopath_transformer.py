"""BioPath Relational Topological Transformer Model."""

from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
from .spd_attention import ShortestPathDistanceAttention
from .config import MODEL_CONFIG

class BioPathBlock(nn.Module):
    """Pre-LN Transformer Block biased by Shortest-Path Distance (SPD)."""
    
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, max_spd: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ShortestPathDistanceAttention(d_model=d_model, n_heads=n_heads, max_spd=max_spd, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        spd_matrix: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_x = self.norm1(x)
        attn_out, weights = self.attn(norm_x, spd_matrix, attention_mask=attention_mask)
        x = x + self.dropout1(attn_out)
        
        norm_x2 = self.norm2(x)
        ff_out = self.linear2(self.act(self.linear1(norm_x2)))
        x = x + self.dropout2(ff_out)
        return x, weights


class BioPathTransformerModel(nn.Module):
    """Predicts metabolic reaction fluxes and oncometabolite accumulation shifts."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = config or MODEL_CONFIG
        
        self.d_model = cfg.get("d_model", 128)
        self.max_nodes = cfg.get("max_graph_nodes", 32)
        n_heads = cfg.get("n_heads", 4)
        num_layers = cfg.get("num_layers", 3)
        dim_ff = cfg.get("dim_feedforward", 256)
        max_spd = cfg.get("max_spd_distance", 8)
        dropout = cfg.get("dropout", 0.10)
        
        # 1. Metabolite Node Embeddings
        vocab_size = cfg.get("vocab_size", cfg.get("num_metabolites", 35))
        self.node_embedding = nn.Embedding(vocab_size, self.d_model)
        
        # 2. Stacked SPD-biased Graphormer Blocks
        self.blocks = nn.ModuleList([
            BioPathBlock(
                d_model=self.d_model,
                n_heads=n_heads,
                dim_feedforward=dim_ff,
                max_spd=max_spd,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)
        
        # 3. Multi-task Output Prediction Heads
        # Head A: Downstream Pathway Flux (umol/min)
        self.flux_head = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1)
        )
        
        # Head B: Oncometabolite Accumulation Score (e.g. 2-HG / Succinate fold surge)
        self.oncometabolite_head = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1)
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        spd_matrix: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            token_ids: LongTensor of node token IDs [B, N]
            spd_matrix: LongTensor of shortest-path distances [B, N, N]
            attention_mask: BoolTensor [B, N] (True for valid nodes)
            
        Returns:
            Dict containing 'flux' [B, 1] and 'oncometabolite_surge' [B, 1]
        """
        x = self.node_embedding(token_ids)  # [B, N, d_model]
        
        for block in self.blocks:
            x, _ = block(x, spd_matrix, attention_mask=attention_mask)
            
        x = self.final_norm(x)
        
        # Multimodal Pooling (Mean + Max over valid nodes)
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1).float()
            mean_pool = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)
            x_masked = x.masked_fill(~attention_mask.unsqueeze(-1), -1e9)
            max_pool, _ = x_masked.max(dim=1)
        else:
            mean_pool = x.mean(dim=1)
            max_pool, _ = x.max(dim=1)
            
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2 * d_model]
        
        flux = self.flux_head(pooled)
        onco = self.oncometabolite_head(pooled)
        
        return {
            "flux": flux,
            "oncometabolite_surge": onco
        }

    def extract_attention(
        self,
        token_ids: torch.Tensor,
        spd_matrix: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Extracts layer-wise attention weights across the metabolic network."""
        self.eval()
        with torch.no_grad():
            x = self.node_embedding(token_ids)
            all_weights = []
            for block in self.blocks:
                x, weights = block(x, spd_matrix, attention_mask=attention_mask)
                all_weights.append(weights)
                
            preds = self.forward(token_ids, spd_matrix, attention_mask=attention_mask)
            return {
                "layer_attention_weights": torch.stack(all_weights, dim=0),
                "flux": preds["flux"],
                "oncometabolite_surge": preds["oncometabolite_surge"]
            }
