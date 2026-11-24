"""Publication-grade visualization suite for BioPath Reaction Transformer."""

import os
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt

BG_COLOR = "#0F111A"
CARD_BG = "#1A1D2B"
TEXT_COLOR = "#E6EDF3"
MUTED_TEXT = "#8B949E"
CYAN_ACCENT = "#38BDF8"
GOLD_ACCENT = "#FACC15"
PURPLE_ACCENT = "#C084FC"
GREEN_ACCENT = "#4ADE80"
RED_ACCENT = "#F87171"
GRID_COLOR = "#2D3748"

class BioPathVisualizer:
    """Renders 300 DPI publication-grade figures in 3B1B dark aesthetic."""
    
    def __init__(self, output_dir: str = "figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._apply_style()

    def _apply_style(self):
        plt.rcParams.update({
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": BG_COLOR,
            "axes.edgecolor": "#30363D",
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "font.family": "serif",
            "font.size": 11,
        })

    def plot_spd_attention_heatmap(
        self,
        attn_matrix: np.ndarray,
        node_labels: List[str],
        filename: str = "spd_attention_heatmap.png"
    ) -> str:
        """Plots Graphormer Shortest-Path Distance attention matrix across metabolic nodes."""
        fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
        
        n_nodes = len(node_labels)
        sub_attn = attn_matrix[:n_nodes, :n_nodes]
        
        cax = ax.imshow(sub_attn, cmap="viridis", interpolation="nearest")
        cbar = fig.colorbar(cax, ax=ax, fraction=0.035, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        cbar.set_label("SPD-Biased Attention Weight $\\mathbf{A}_{ij}$", color=TEXT_COLOR, size=11, labelpad=10)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_COLOR)
        
        ax.set_xticks(np.arange(n_nodes))
        ax.set_xticklabels(node_labels, rotation=45, ha="right", fontsize=9, color=CYAN_ACCENT)
        ax.set_yticks(np.arange(n_nodes))
        ax.set_yticklabels(node_labels, fontsize=9, color=TEXT_COLOR)
        
        ax.set_title("Graphormer Topological Attention (SPD-Biased Metabolic Flux)", fontsize=12, fontweight="bold", color=GOLD_ACCENT, pad=16)
        ax.set_xlabel("Target Metabolite Node", color=TEXT_COLOR, fontsize=10, labelpad=8)
        ax.set_ylabel("Source Metabolite Node", color=TEXT_COLOR, fontsize=10, labelpad=8)
        
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
        plt.close(fig)
        return out_path

    def plot_oncometabolite_surge(
        self,
        perturbation_results: Dict[str, Any],
        filename: str = "oncometabolite_flux_shift.png"
    ) -> str:
        """Plots oncometabolite accumulation under oncogenic enzyme mutations."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        
        # Panel 1: IDH1 R132H 2-HG Accumulation
        labels1 = ["Wild-Type TCA", "IDH1 R132H (Mutant)"]
        values1 = [perturbation_results["wt_2hg_level"], perturbation_results["mut_2hg_level"]]
        bars1 = ax1.bar(labels1, values1, color=["#64748B", RED_ACCENT], width=0.5, edgecolor="#FFFFFF", linewidth=0.5)
        ax1.set_title("IDH1 R132H Neomorphic Shunt (2-HG)", fontsize=11, fontweight="bold", color=GOLD_ACCENT)
        ax1.set_ylabel("Oncometabolite Level (D-2-HG Score)", fontsize=10)
        ax1.grid(axis="y", linestyle="--", alpha=0.3)
        
        for bar in bars1:
            h = bar.get_height()
            ax1.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                         textcoords="offset points", ha="center", va="bottom", fontsize=10, color=TEXT_COLOR)
                         
        # Panel 2: Pathway Flux Comparison
        labels2 = ["WT TCA Flux", "Mutant TCA Flux"]
        values2 = [perturbation_results["wt_flux"], perturbation_results["mut_flux"]]
        bars2 = ax2.bar(labels2, values2, color=[CYAN_ACCENT, GOLD_ACCENT], width=0.5, edgecolor="#FFFFFF", linewidth=0.5)
        ax2.set_title("TCA Cycle Forward Flux Inhibition", fontsize=11, fontweight="bold", color=GOLD_ACCENT)
        ax2.set_ylabel(r"Downstream Pathway Flux ($\mu\mathrm{mol/min}$)", fontsize=10)
        ax2.grid(axis="y", linestyle="--", alpha=0.3)
        
        for bar in bars2:
            h = bar.get_height()
            ax2.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                         textcoords="offset points", ha="center", va="bottom", fontsize=10, color=TEXT_COLOR)
                         
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
        plt.close(fig)
        return out_path

    def plot_predicted_vs_actual_flux(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        spearman_rho: float,
        filename: str = "flux_predicted_vs_actual.png"
    ) -> str:
        """Plots true vs predicted metabolic flux values."""
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        
        ax.scatter(y_true, y_pred, color=CYAN_ACCENT, alpha=0.8, edgecolors="#FFFFFF", s=50, linewidths=0.4)
        min_val = min(y_true.min(), y_pred.min()) - 0.2
        max_val = max(y_true.max(), y_pred.max()) + 0.2
        ax.plot([min_val, max_val], [min_val, max_val], color=GOLD_ACCENT, linestyle="--", linewidth=1.5, label="Identity ($y = x$)")
        
        ax.set_title("BioPath: Metabolic Reaction Flux Prediction", fontsize=12, fontweight="bold", color=GOLD_ACCENT, pad=12)
        ax.set_xlabel(r"True Pathway Flux ($\mu\mathrm{mol/min}$)", fontsize=10)
        ax.set_ylabel("Predicted Pathway Flux", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        ax.text(0.06, 0.90, f"Spearman $\\rho$: {spearman_rho:.4f}", transform=ax.transAxes,
                fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD_BG, edgecolor=CYAN_ACCENT, alpha=0.9))
                
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
        plt.close(fig)
        return out_path
