"""
predict.py — Predict Tox21 toxicity for a single SMILES string.

Usage
-----
  python predict.py --smiles "CC(=O)Nc1ccc(O)cc1"   # Paracetamol
  python predict.py --smiles "C1=CC=CC=C1"           # Benzene
  python predict.py --smiles "CCO"                   # Ethanol

Teaching notes
--------------
SMILES (Simplified Molecular-Input Line-Entry System) is a text notation
for molecules. Examples:
  - "O"           = water
  - "CCO"         = ethanol (CH3-CH2-OH)
  - "c1ccccc1"    = benzene (aromatic ring)

This script:
  1. Parses the SMILES string with RDKit to get atoms + bonds.
  2. Featurises each atom (same features used during training).
  3. Builds a PyG Data object (x, edge_index).
  4. Runs a forward pass through the loaded model.
  5. Applies sigmoid to get probabilities in [0, 1].
  6. Prints per-task predictions with a simple toxic / non-toxic label.
"""

import argparse

import torch
import numpy as np

from data import TOX21_TASKS, NUM_TASKS
from model import build_model


# ------------------------------------------------------------------ #
#  Atom featurisation (must match training-time features)             #
# ------------------------------------------------------------------ #

def atom_features(atom) -> list:
    """
    Encode one atom as a fixed-length feature vector.
    These are the same 9 features PyG's MoleculeNet uses for Tox21.

    Feature vector:
      [0] atomic_num_onehot (6 dims: C, N, O, F, P, S, Cl, Br, I, other)
      Simplified here to match PyG's internal featuriser.

    NOTE: For exact compatibility with MoleculeNet's featurisation,
    we use PyG's own `from_smiles` utility which replicates the same
    atom features used at dataset-creation time.
    """
    # We delegate to PyG's torch_geometric.utils.smiles.from_smiles
    # which uses the identical featurisation as MoleculeNet.
    pass  # handled in smiles_to_data below


def smiles_to_data(smiles: str):
    """
    Convert a SMILES string to a PyG Data object.

    Uses torch_geometric.utils.smiles.from_smiles which replicates
    the exact atom features used in MoleculeNet's Tox21 loader.

    Returns None if SMILES is invalid.
    """
    try:
        from torch_geometric.utils import smiles as pyg_smiles
        data = pyg_smiles.from_smiles(smiles)
        if data is None or data.x is None:
            raise ValueError("Could not parse SMILES.")
        data.x = data.x.float()
        return data
    except Exception as e:
        print(f"[predict.py] Error parsing SMILES '{smiles}': {e}")
        return None


# ------------------------------------------------------------------ #
#  Prediction                                                         #
# ------------------------------------------------------------------ #

@torch.no_grad()
def predict_smiles(
    smiles: str,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
):
    """
    Predict toxicity for a SMILES string.

    Returns
    -------
    probs : numpy array of shape [12] with probability of toxicity per task.
    """
    data = smiles_to_data(smiles)
    if data is None:
        return None

    data = data.to(device)

    # batch tensor: all nodes belong to the same (only) molecule -> batch=0
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    model.eval()
    logits = model(data.x, data.edge_index, batch)  # [1, 12]
    probs  = torch.sigmoid(logits).cpu().numpy().flatten()  # [12]
    return probs


def print_predictions(
    smiles: str,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> None:
    """Pretty-print per-task toxicity predictions."""
    print(f"\n{'='*55}")
    print(f"Molecule (SMILES): {smiles}")
    print(f"{'='*55}")
    print(f"{'Task':<20} {'Probability':>12}  {'Prediction':>12}")
    print(f"{'-'*55}")
    for task, prob in zip(TOX21_TASKS, probs):
        label = "TOXIC" if prob >= threshold else "non-toxic"
        flag  = " <-- !!" if prob >= threshold else ""
        print(f"{task:<20} {prob:>12.4f}  {label:>12}{flag}")
    print(f"{'-'*55}")
    macro_prob = probs.mean()
    print(f"{'Macro avg prob':<20} {macro_prob:>12.4f}")
    n_toxic = (probs >= threshold).sum()
    print(f"\nFlagged as TOXIC in {n_toxic} / {NUM_TASKS} assays (threshold={threshold})")
    print(f"{'='*55}\n")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Predict Tox21 toxicity from SMILES")
    p.add_argument(
        "--smiles", type=str, required=True,
        help="SMILES string, e.g. \"CC(=O)Nc1ccc(O)cc1\" for paracetamol"
    )
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="Decision threshold for TOXIC label (default 0.5)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"[predict.py] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args   = ckpt.get("args", {})
    in_channels = ckpt.get("in_channels", 9)

    model = build_model(
        in_channels=in_channels,
        hidden_channels=ckpt_args.get("hidden", 128),
        num_tasks=NUM_TASKS,
        dropout=0.0,
        num_layers=ckpt_args.get("num_layers", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[predict.py] Model loaded (epoch {ckpt.get('epoch', '?')})")

    # Predict
    probs = predict_smiles(args.smiles, model, device, args.threshold)
    if probs is None:
        print("[predict.py] Failed to parse SMILES. Check your input.")
        return

    print_predictions(args.smiles, probs, threshold=args.threshold)


if __name__ == "__main__":
    main()
