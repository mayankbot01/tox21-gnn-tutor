"""
train.py — Training and validation loop for Tox21 GNN.

Usage
-----
  python train.py
  python train.py --epochs 50 --lr 0.001 --hidden 128 --batch_size 64

Teaching notes
--------------
Training loop recap:
  For each epoch:
    1. Set model to train mode (enables Dropout + BatchNorm training behavior)
    2. Loop over batches from train_loader
    3. Forward pass -> get logits
    4. Compute masked BCE loss
    5. Backward pass -> compute gradients
    6. Optimizer step -> update weights
    7. Evaluate on val set (model in eval mode)
    8. Save checkpoint if val ROC-AUC improves

Overfitting signs:
  - Train loss keeps decreasing but val AUC plateaus or drops
  - Gap between train and val AUC keeps growing
  Remedies: increase dropout, reduce model size, add weight decay, early stop.

Early stopping:
  We stop training if val ROC-AUC doesn't improve for `patience` epochs.
  This prevents wasting compute and overfitting.
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from data import (
    load_tox21, random_split, get_loaders,
    compute_pos_weight, get_label_mask, NUM_TASKS
)
from model import build_model
from eval import compute_roc_auc


# ------------------------------------------------------------------ #
#  Argument parsing                                                    #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Train Tox21 GNN")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hidden",     type=int,   default=128)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--num_layers", type=int,   default=3)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--patience",   type=int,   default=10,
                   help="Early stopping patience (epochs without improvement)")
    p.add_argument("--checkpoint", type=str,   default="best_model.pt")
    p.add_argument("--data_root",  type=str,   default="./data")
    p.add_argument("--log",        type=str,   default="training_log.csv")
    p.add_argument("--scaffold_split", action="store_true",
                   help="Use scaffold split instead of random split (harder, more realistic)")
    return p.parse_args()


# ------------------------------------------------------------------ #
#  Masked loss                                                         #
# ------------------------------------------------------------------ #

def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute binary cross-entropy loss, ignoring NaN labels.

    Steps
    -----
    1. Build mask: True where label is known (not NaN).
    2. Replace NaN in targets with 0 (value doesn't matter; masked out).
    3. Compute per-element BCE loss (reduction='none') so we get a
       [B, 12] tensor of individual losses.
    4. Zero out masked (unknown) positions.
    5. Average only over known positions.

    Why BCEWithLogitsLoss and not BCELoss?
      BCEWithLogitsLoss = sigmoid + BCE in one numerically stable op.
      It avoids float precision issues that can cause NaN gradients.
    """
    mask = get_label_mask(targets)  # [B, 12] bool
    targets_clean = targets.clone()
    targets_clean[~mask] = 0.0     # fill NaN with 0 (masked away anyway)

    criterion = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=pos_weight.to(device)
    )
    loss_per_elem = criterion(logits, targets_clean)  # [B, 12]
    loss_per_elem = loss_per_elem * mask.float()      # zero out unknown

    n_valid = mask.sum().float()
    if n_valid == 0:
        # Edge case: entire batch has no known labels (shouldn't happen)
        return torch.tensor(0.0, requires_grad=True, device=device)
    return loss_per_elem.sum() / n_valid


# ------------------------------------------------------------------ #
#  Training epoch                                                      #
# ------------------------------------------------------------------ #

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """
    Run one training epoch. Returns mean loss over all batches.

    grad_clip: gradient clipping norm threshold.
      Why? If logits explode early in training, gradients can become
      very large and cause NaN weights. Clipping keeps them bounded.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)

        # Forward
        logits = model(batch.x, batch.edge_index, batch.batch)  # [B, 12]
        y = batch.y  # [B, 12]

        # Loss
        loss = masked_bce_loss(logits, y, pos_weight, device)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ------------------------------------------------------------------ #
#  Validation epoch                                                    #
# ------------------------------------------------------------------ #

@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    pos_weight: torch.Tensor,
    device: torch.device,
):
    """
    Run validation. Returns (mean_loss, macro_roc_auc, per_task_auc).

    @torch.no_grad(): disables gradient computation for speed + memory.
    We also set model.eval() to disable Dropout and use running
    BatchNorm statistics.
    """
    model.eval()
    all_logits, all_targets = [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        y = batch.y

        loss = masked_bce_loss(logits, y, pos_weight, device)
        total_loss += loss.item()
        n_batches += 1

        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    all_logits  = torch.cat(all_logits,  dim=0)  # [N_val, 12]
    all_targets = torch.cat(all_targets, dim=0)  # [N_val, 12]

    probs = torch.sigmoid(all_logits)
    macro_auc, per_task_auc = compute_roc_auc(probs, all_targets)

    return total_loss / max(n_batches, 1), macro_auc, per_task_auc


# ------------------------------------------------------------------ #
#  Main training loop                                                  #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train.py] Using device: {device}")

    # Data
    dataset = load_tox21(root=args.data_root)
    if args.scaffold_split:
        from data import scaffold_split
        train_set, val_set, test_set = scaffold_split(dataset, seed=args.seed)
    else:
        train_set, val_set, test_set = random_split(dataset, seed=args.seed)

    pos_weight = compute_pos_weight(train_set).to(device)
    train_loader, val_loader, _ = get_loaders(
        train_set, val_set, test_set, batch_size=args.batch_size
    )

    # Sanity check: verify first batch doesn't crash
    sample_batch = next(iter(train_loader))
    in_channels = sample_batch.x.size(1)  # typically 9 for Tox21
    print(f"[train.py] Node feature dim: {in_channels}")

    # Model
    model = build_model(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        num_tasks=NUM_TASKS,
        dropout=args.dropout,
        num_layers=args.num_layers,
    ).to(device)
    print(f"[train.py] Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # Logging
    log_rows = []
    best_val_auc = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"\n{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>9} | {'Val AUC':>8} | {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, pos_weight, device
        )
        val_loss, val_auc, per_task_auc = validate(
            model, val_loader, pos_weight, device
        )
        scheduler.step(val_auc)

        elapsed = time.time() - t0
        print(
            f"{epoch:>6} | {train_loss:>11.4f} | {val_loss:>9.4f} | "
            f"{val_auc:>8.4f} | {elapsed:>5.1f}s"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_auc_macro": round(val_auc, 6),
        })

        # Save best checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "args": vars(args),
                    "in_channels": in_channels,
                },
                args.checkpoint,
            )
            print(f"         [*] New best! Saved checkpoint to {args.checkpoint}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[train.py] Early stopping at epoch {epoch}. "
                      f"Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}.")
                break

    # Save training log
    with open(args.log, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n[train.py] Training log saved to {args.log}")
    print(f"[train.py] Best val ROC-AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    print(f"[train.py] Run `python eval.py --checkpoint {args.checkpoint}` for test metrics.")


if __name__ == "__main__":
    main()
