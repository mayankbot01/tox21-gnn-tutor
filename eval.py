"""
eval.py — ROC-AUC evaluation for Tox21 multi-label classification.

Usage
-----
  python eval.py --checkpoint best_model.pt

Teaching notes
--------------
Why ROC-AUC and not accuracy?
  Tox21 is heavily imbalanced: most molecules are non-toxic (label=0).
  A model that always predicts "non-toxic" gets ~90%+ accuracy but
  is completely useless. ROC-AUC measures how well the model *ranks*
  toxic molecules above non-toxic ones, regardless of threshold.
  An AUC of 0.5 = random guessing; 1.0 = perfect.

Per-task AUC matters:
  Some endpoints (assays) are harder than others due to fewer positives
  or noisier data. Reporting per-task AUC lets you identify which
  toxicity endpoints the model handles well vs. poorly.

Edge case: skipped tasks
  If a split contains only one class for a task (all 0 or all 1),
  ROC-AUC is undefined. We skip those tasks from the macro average
  and report NaN for them.
"""

import argparse
import json
from typing import Dict, Tuple

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from data import (
    load_tox21, random_split, get_loaders,
    compute_pos_weight, get_label_mask, TOX21_TASKS, NUM_TASKS
)
from model import build_model


# ------------------------------------------------------------------ #
#  Core AUC computation                                               #
# ------------------------------------------------------------------ #

def compute_roc_auc(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute per-task ROC-AUC and macro average, handling missing labels.

    Parameters
    ----------
    probs   : [N, 12] float tensor (sigmoid of logits, range 0..1)
    targets : [N, 12] float tensor (0.0, 1.0, or NaN)

    Returns
    -------
    macro_auc   : float  (mean over valid tasks)
    per_task    : dict   {task_name: auc_score or NaN}
    """
    probs   = probs.numpy()   if isinstance(probs, torch.Tensor)   else np.array(probs)
    targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)

    per_task = {}
    valid_aucs = []

    for t, task_name in enumerate(TOX21_TASKS):
        y_true = targets[:, t]          # [N]
        y_pred = probs[:, t]            # [N]

        # Mask out NaN labels
        valid = ~np.isnan(y_true)
        y_true_v = y_true[valid]
        y_pred_v = y_pred[valid]

        # Need at least 2 classes to compute AUC
        if len(np.unique(y_true_v)) < 2:
            per_task[task_name] = float("nan")
            continue

        auc = roc_auc_score(y_true_v, y_pred_v)
        per_task[task_name] = round(float(auc), 4)
        valid_aucs.append(auc)

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return macro_auc, per_task


# ------------------------------------------------------------------ #
#  Inference on a DataLoader                                          #
# ------------------------------------------------------------------ #

@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model on all batches; return stacked probs and targets."""
    model.eval()
    all_probs, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        probs  = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_targets.append(batch.y.cpu())

    return torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)


# ------------------------------------------------------------------ #
#  Pretty-print results                                               #
# ------------------------------------------------------------------ #

def print_results(macro_auc: float, per_task: Dict[str, float]) -> None:
    print(f"\n{'Task':<20} {'ROC-AUC':>8}")
    print("-" * 31)
    for task, auc in per_task.items():
        marker = "  (skipped)" if np.isnan(auc) else ""
        val_str = f"{auc:.4f}" if not np.isnan(auc) else "  N/A  "
        print(f"{task:<20} {val_str:>8}{marker}")
    print("-" * 31)
    print(f"{'Macro Average':<20} {macro_auc:>8.4f}")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Tox21 GNN checkpoint")
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    p.add_argument("--data_root",  type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output",     type=str, default="metrics.json")
    p.add_argument("--scaffold_split", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval.py] Device: {device}")

    # Load checkpoint
    print(f"[eval.py] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    in_channels = ckpt.get("in_channels", 9)

    model = build_model(
        in_channels=in_channels,
        hidden_channels=ckpt_args.get("hidden", 128),
        num_tasks=NUM_TASKS,
        dropout=0.0,  # no dropout during eval
        num_layers=ckpt_args.get("num_layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[eval.py] Loaded model from epoch {ckpt.get('epoch', '?')}, "
          f"val AUC={ckpt.get('val_auc', float('nan')):.4f}")

    # Data
    dataset = load_tox21(root=args.data_root)
    if args.scaffold_split:
        from data import scaffold_split
        _, _, test_set = scaffold_split(dataset, seed=args.seed)
    else:
        _, _, test_set = random_split(dataset, seed=args.seed)

    _, _, test_loader = get_loaders(
        test_set, test_set, test_set,  # val/train dummy
        batch_size=args.batch_size
    )
    # Re-get just test_loader properly
    from torch_geometric.loader import DataLoader
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Evaluate
    probs, targets = collect_predictions(model, test_loader, device)
    macro_auc, per_task = compute_roc_auc(probs, targets)

    print_results(macro_auc, per_task)

    # Save metrics
    metrics = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch"),
        "macro_roc_auc": round(macro_auc, 4),
        "per_task_roc_auc": per_task,
    }
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[eval.py] Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
