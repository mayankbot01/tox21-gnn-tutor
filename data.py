"""
data.py — Dataset loading, train/val/test splits, masking utilities.

Teaching notes
--------------
Tox21 via MoleculeNet (PyG)
  - ~7,831 molecules
  - 12 binary classification tasks (toxicity assays)
  - Some labels are NaN ("missing") because not every molecule was tested
    on every assay. We MUST mask these out before computing loss.

Split strategies
  Random split:   Fast; molecules in train/val/test overlap chemically.
                  OK for learning; gives optimistic metrics.
  Scaffold split: Splits by molecular scaffold (the core ring system).
                  Trains on one set of chemical scaffolds, tests on
                  different ones — better measure of generalization.
                  Preferred in research papers.

We implement both. Random split is used by default for speed.
"""

import os
import random
from typing import Tuple, List

import numpy as np
import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]
NUM_TASKS = len(TOX21_TASKS)  # 12


# ------------------------------------------------------------------ #
#  Load dataset                                                        #
# ------------------------------------------------------------------ #

def load_tox21(root: str = "./data") -> MoleculeNet:
    """
    Download and load the Tox21 dataset.

    Each element `data` in the returned dataset is a
    torch_geometric.data.Data object with:
      data.x           — atom features, shape [num_atoms, 9]
      data.edge_index  — bond connectivity, shape [2, num_bonds*2]
      data.edge_attr   — bond features (optional), shape [num_bonds*2, 3]
      data.y           — labels, shape [1, 12] (NaN where unknown)
      data.smiles      — SMILES string for the molecule

    Returns
    -------
    dataset : MoleculeNet
    """
    print(f"[data.py] Loading Tox21 from {os.path.abspath(root)} ...")
    dataset = MoleculeNet(root=root, name="Tox21")
    print(f"[data.py] Loaded {len(dataset)} molecules, "
          f"{dataset.num_node_features} atom features, "
          f"{NUM_TASKS} tasks.")
    return dataset


def print_sample_info(dataset: MoleculeNet, idx: int = 0) -> None:
    """Print shapes for one sample — use as a sanity check."""
    data = dataset[idx]
    print(f"Sample [{idx}]")
    print(f"  x (atom features)  : {data.x.shape}    dtype={data.x.dtype}")
    print(f"  edge_index         : {data.edge_index.shape} dtype={data.edge_index.dtype}")
    print(f"  y (labels)         : {data.y.shape}    dtype={data.y.dtype}")
    print(f"  SMILES             : {data.smiles}")
    print(f"  Label values       : {data.y}")


# ------------------------------------------------------------------ #
#  Splits                                                             #
# ------------------------------------------------------------------ #

def random_split(
    dataset: MoleculeNet,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Randomly split dataset into train / val / test.

    train_frac + val_frac + (1 - train_frac - val_frac) = 1.0
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)

    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]

    train_set = dataset[torch.tensor(train_idx)]
    val_set   = dataset[torch.tensor(val_idx)]
    test_set  = dataset[torch.tensor(test_idx)]

    print(f"[data.py] Random split: train={len(train_set)}, "
          f"val={len(val_set)}, test={len(test_set)}")
    return train_set, val_set, test_set


def scaffold_split(
    dataset: MoleculeNet,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Scaffold split: groups molecules by their Murcko scaffold (the core
    ring system). The largest scaffolds go to training; smaller / unique
    scaffolds go to val/test. This creates a harder, more realistic split.

    Requires: rdkit
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        raise ImportError(
            "rdkit is required for scaffold split. "
            "Install with: pip install rdkit"
        )

    # Step 1: compute scaffold for each molecule
    scaffolds = {}  # scaffold_smiles -> list of indices
    for idx, data in enumerate(dataset):
        smiles = data.smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
        except Exception:
            scaffold = smiles  # fallback: treat molecule as its own scaffold
        scaffolds.setdefault(scaffold, []).append(idx)

    # Step 2: sort scaffold groups by size (descending) for determinism
    scaffold_sets = sorted(
        scaffolds.values(), key=lambda x: (len(x), x[0]), reverse=True
    )

    # Step 3: assign scaffold groups to splits
    n = len(dataset)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_idx, val_idx, test_idx = [], [], []
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) <= n_train:
            train_idx.extend(scaffold_set)
        elif len(val_idx) + len(scaffold_set) <= n_val:
            val_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)

    train_set = dataset[torch.tensor(train_idx)]
    val_set   = dataset[torch.tensor(val_idx)]
    test_set  = dataset[torch.tensor(test_idx)]

    print(f"[data.py] Scaffold split: train={len(train_set)}, "
          f"val={len(val_set)}, test={len(test_set)}")
    return train_set, val_set, test_set


# ------------------------------------------------------------------ #
#  DataLoaders                                                         #
# ------------------------------------------------------------------ #

def get_loaders(
    train_set, val_set, test_set,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap splits in PyG DataLoaders."""
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


# ------------------------------------------------------------------ #
#  Missing-label mask utilities                                        #
# ------------------------------------------------------------------ #

def get_label_mask(y: torch.Tensor) -> torch.Tensor:
    """
    Return a boolean mask of the same shape as y.
    True  = label is known (train/evaluate on this entry).
    False = label is NaN  (skip this entry in loss and metrics).

    Example
    -------
    y    = tensor([[1., 0., nan, 1., ...]]) shape [B, 12]
    mask = tensor([[True, True, False, True, ...]]) shape [B, 12]
    """
    return ~torch.isnan(y)


def compute_pos_weight(dataset: MoleculeNet) -> torch.Tensor:
    """
    Compute per-task positive class weights for BCEWithLogitsLoss.

    Formula: pos_weight[t] = num_negative[t] / num_positive[t]

    Why? If 95% of labels for a task are 0 (non-toxic) and only 5% are 1
    (toxic), the model can get 95% accuracy by always predicting 0. That's
    useless! pos_weight upweights the loss on positive (toxic) examples so
    the model is penalized harder for missing them.
    """
    all_y = []
    for data in dataset:
        all_y.append(data.y)  # shape [1, 12]
    all_y = torch.cat(all_y, dim=0)  # [N, 12]

    pos_weight = []
    for t in range(NUM_TASKS):
        col = all_y[:, t]
        valid = col[~torch.isnan(col)]
        n_pos = (valid == 1).sum().float()
        n_neg = (valid == 0).sum().float()
        # Avoid division by zero; default to 1.0 if no positives
        w = (n_neg / n_pos) if n_pos > 0 else torch.tensor(1.0)
        pos_weight.append(w)

    pos_weight = torch.stack(pos_weight)  # [12]
    print(f"[data.py] pos_weight per task: {pos_weight.round(decimals=1).tolist()}")
    return pos_weight


def print_label_stats(dataset, name: str = "dataset") -> None:
    """Print fraction of positives per task — sanity check."""
    all_y = []
    for data in dataset:
        all_y.append(data.y)
    all_y = torch.cat(all_y, dim=0)  # [N, 12]

    print(f"\nLabel stats for {name} ({len(dataset)} molecules):")
    print(f"  {'Task':<20} {'Pos%':>6}  {'N_valid':>8}")
    for i, task in enumerate(TOX21_TASKS):
        col = all_y[:, i]
        valid = col[~torch.isnan(col)]
        pct = 100.0 * (valid == 1).sum() / len(valid) if len(valid) > 0 else 0.0
        print(f"  {task:<20} {pct:>5.1f}%  {len(valid):>8}")


# ------------------------------------------------------------------ #
#  Standalone test                                                     #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    dataset = load_tox21()
    print_sample_info(dataset, idx=0)
    train_set, val_set, test_set = random_split(dataset)
    print_label_stats(train_set, name="train")
    pw = compute_pos_weight(train_set)
    train_loader, val_loader, test_loader = get_loaders(
        train_set, val_set, test_set, batch_size=64
    )
    batch = next(iter(train_loader))
    mask = get_label_mask(batch.y)
    print(f"\nFirst batch:")
    print(f"  batch.x.shape         : {batch.x.shape}")
    print(f"  batch.edge_index.shape: {batch.edge_index.shape}")
    print(f"  batch.y.shape         : {batch.y.shape}")
    print(f"  mask (valid labels)   : {mask.sum().item()} / {mask.numel()}")
    print("[data.py] All sanity checks passed!")
