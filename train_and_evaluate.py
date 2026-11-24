#!/usr/bin/env python3
"""Automated Training, Metabolic Perturbation, and Visualization Driver for BioPath Transformer."""

import os
import argparse
import time
import torch
import numpy as np

from src.config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    METABOLITE_NAMES,
    METABOLIC_PATHWAYS,
    ONCOMETABOLITE_SHUNTS
)
from src.metabolic_tokenizer import MetabolicTokenizer
from src.biopath_transformer import BioPathTransformerModel
from src.data_loader import generate_metabolic_benchmark, create_dataloaders
from src.trainer import BioPathModelTrainer
from src.flux_perturbation import MetabolicFluxPerturbation
from src.visualizer import BioPathVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate BioPath Topological Transformer.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--samples", type=int, default=400, help="Number of benchmark samples (default: 400)")
    parser.add_argument("--save-dir", type=str, default="figures", help="Directory for generated figures (default: figures)")
    parser.add_argument("--data-file", type=str, default="data/metabolic_benchmark.csv", help="Path to benchmark CSV")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory for model weights")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    print("=" * 76)
    print("⚗️  BIOPATH REACTION TRANSFORMER: SHORTEST-PATH DISTANCE (SPD) ATTENTION")
    print("   Ying et al. (NeurIPS / Graphormer) & Schwaller et al. (Nature Machine Intelligence)")
    print("=" * 76)
    
    # 1. Dataset Generation
    print(f"\n[1/5] Synthesizing metabolic reaction cascades ({args.samples} samples)...")
    records = generate_metabolic_benchmark(
        num_samples=args.samples,
        random_seed=42,
        csv_path=args.data_file
    )
    print(f"      Saved curated benchmark to: {args.data_file}")
    
    # 2. Tokenizer & Dataloaders
    print("\n[2/5] Initializing metabolic topological tokenizer & data loaders...")
    tokenizer = MetabolicTokenizer(
        max_nodes=MODEL_CONFIG["max_graph_nodes"],
        max_spd=MODEL_CONFIG["max_spd_distance"]
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        records=records,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        train_ratio=TRAINING_CONFIG["train_ratio"],
        val_ratio=TRAINING_CONFIG["val_ratio"],
        seed=TRAINING_CONFIG["seed"]
    )
    print(f"      Train samples: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    # 3. Model Architecture
    print("\n[3/5] Instantiating BioPath Graphormer Transformer...")
    cfg = dict(MODEL_CONFIG)
    model = BioPathTransformerModel(cfg)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total Trainable Parameters: {total_params:,} (~{total_params/1e6:.2f}M)")
    print(f"      Shortest-Path Distance Truncation: {cfg['max_spd_distance']} hops")
    print(f"      Running on Device: CPU (Intel Core i5 Optimized)")
    
    # 4. Training Loop
    print("\n[4/5] Executing high-efficiency training loop...")
    train_cfg = dict(TRAINING_CONFIG)
    train_cfg["num_epochs"] = args.epochs
    train_cfg["learning_rate"] = args.lr
    train_cfg["batch_size"] = args.batch_size
    
    trainer = BioPathModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=train_cfg,
        device=torch.device("cpu")
    )
    
    train_results = trainer.fit(verbose=True)
    test_metrics = train_results["test_metrics"]
    
    best_ckpt_path = os.path.join(args.ckpt_dir, "best_biopath_model.pt")
    torch.save(model.state_dict(), best_ckpt_path)
    print(f"\n      Saved best model checkpoint to: {best_ckpt_path}")
    
    # 5. Perturbation Simulation & Visualization
    print("\n[5/5] Simulating Oncogenic IDH1 R132H Perturbation & Rendering Figures...")
    visualizer = BioPathVisualizer(output_dir=args.save_dir)
    pert = MetabolicFluxPerturbation(model=model, tokenizer=tokenizer)
    
    # Figure A: Predicted vs Actual Flux Scatter Plot
    scatter_path = visualizer.plot_predicted_vs_actual_flux(
        y_true=test_metrics["true_flux"],
        y_pred=test_metrics["pred_flux"],
        spearman_rho=test_metrics["flux_spearman"],
        filename="flux_predicted_vs_actual.png"
    )
    print(f"      [Figure 1/3] Flux Correlation:         {scatter_path}")
    
    # Figure B: IDH1 R132H Simulation & 2-HG Surge
    idh1_res = pert.simulate_idh1_r132h()
    bar_path = visualizer.plot_oncometabolite_surge(
        perturbation_results=idh1_res,
        filename="oncometabolite_flux_shift.png"
    )
    print(f"      [Figure 2/3] Oncometabolite Shift:     {bar_path}")
    
    # Figure C: SPD-Biased Attention Heatmap
    tca_sample = ["Citrate", "Isocitrate", "Alpha_Ketoglutarate", "Succinate", "Fumarate", "Malate", "Oxaloacetate"]
    ids, mask, spd = tokenizer.encode_pathway(tca_sample)
    attn_res = model.extract_attention(ids.unsqueeze(0), spd.unsqueeze(0), attention_mask=mask.unsqueeze(0))
    # Average across heads in last layer: [1, n_heads, N, N] -> [N, N]
    last_layer_attn = attn_res["layer_attention_weights"][-1, 0].mean(dim=0).cpu().numpy()
    
    heatmap_path = visualizer.plot_spd_attention_heatmap(
        attn_matrix=last_layer_attn,
        node_labels=tca_sample,
        filename="spd_attention_heatmap.png"
    )
    print(f"      [Figure 3/3] SPD Attention Heatmap:    {heatmap_path}")
    
    print("\n" + "=" * 76)
    print("🏆 ALL EXECUTION MILESTONES COMPLETE")
    print(f"   • Flux Test Spearman Rank ρ:       {test_metrics['flux_spearman']:.4f}")
    print(f"   • Oncometabolite Test Spearman ρ:  {test_metrics['onco_spearman']:.4f}")
    print(f"   • IDH1 R132H 2-HG Fold Surge:      {idh1_res['oncometabolite_fold_surge']:.2f}x")
    print(f"   • Generated Artifacts:             {args.save_dir}/")
    print("=" * 76)

if __name__ == "__main__":
    main()
