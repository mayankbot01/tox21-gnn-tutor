# tox21-gnn-tutor

> **AI Tutor + Pair-Programming Project** — Learn to build a Graph Neural Network (GNN) that predicts molecular toxicity using the Tox21 dataset and PyTorch Geometric.

---

## What This Project Does

Tox21 is a public dataset of ~8,000 molecules tested across **12 toxicity assays** (e.g., nuclear receptor signaling, stress response pathways). We represent each molecule as a **graph** (atoms = nodes, bonds = edges) and train a GNN to predict all 12 binary toxicity labels simultaneously.

---

## Project Structure

```
tox21-gnn-tutor/
├── README.md          # This file
├── requirements.txt   # Pinned dependencies
├── data.py            # Dataset loading, splits, masking utilities
├── model.py           # GNN model definition
├── train.py           # Training + validation loop
├── eval.py            # ROC-AUC computation per task
└── predict.py         # Predict toxicity from a single SMILES string
```

---

## Setup

### Option A — Google Colab (recommended for beginners)

```python
# Run this cell first in your Colab notebook
!pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install torch_geometric==2.4.0
!pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
!pip install rdkit scikit-learn pandas
```

### Option B — Local machine (conda)

```bash
conda env create -f environment.yml
conda activate tox21-gnn
```

### Option C — pip (CPU only, simpler)

```bash
pip install -r requirements.txt
```

---

## How to Train

```bash
python train.py --epochs 50 --lr 0.001 --hidden 128 --batch_size 64
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 50 | Number of training epochs |
| `--lr` | 0.001 | Learning rate (Adam) |
| `--hidden` | 128 | Hidden channel size in GNN layers |
| `--batch_size` | 64 | Molecules per batch |
| `--dropout` | 0.3 | Dropout rate |
| `--seed` | 42 | Random seed |
| `--checkpoint` | `best_model.pt` | Where to save best model |

Training will print per-epoch metrics:
```
Epoch 01 | Train Loss: 0.4821 | Val ROC-AUC (macro): 0.7234
Epoch 02 | Train Loss: 0.4103 | Val ROC-AUC (macro): 0.7401
...
```

---

## How to Evaluate

```bash
python eval.py --checkpoint best_model.pt
```

This prints per-task ROC-AUC and the macro average, and saves `metrics.json`.

---

## How to Predict a Single Molecule

```bash
python predict.py --smiles "CC(=O)Nc1ccc(O)cc1"  # Paracetamol
```

Output:
```
Molecule: CC(=O)Nc1ccc(O)cc1
Tox21 Task Predictions:
  NR-AR         : 0.032  (non-toxic)
  NR-AR-LBD     : 0.041  (non-toxic)
  NR-AhR        : 0.187  (non-toxic)
  ...
  SR-p53        : 0.091  (non-toxic)
Macro avg prob  : 0.089
```

---

## Concept Map (Read Before Coding)

### 1. Molecules as Graphs
A **molecule** is a set of atoms connected by bonds.
- **Nodes** = atoms (each carries features like atomic number, charge, etc.)
- **Edges** = bonds (single, double, aromatic, etc.)

For example, water (H2O) would be a graph with 3 nodes (O, H, H) and 2 edges.

### 2. `edge_index` in PyTorch Geometric
`edge_index` is a `[2, num_edges]` tensor where:
- Row 0 = source node indices
- Row 1 = destination node indices

Edges are stored **bidirectionally** (A→B and B→A both appear).

### 3. Node Features (`x`)
Each atom is described by a feature vector (e.g., atomic number, degree, hybridization). Shape: `[num_atoms, num_features]`.

### 4. Message Passing
In each GNN layer, every node **collects messages** from its neighbors, **aggregates** them (e.g., sum/mean), and **updates** its own representation. After `k` layers, each node "sees" a k-hop neighborhood.

### 5. Graph Pooling
To get a single embedding for the whole molecule (for classification), we **pool** all node embeddings (e.g., global mean pool) → one vector per molecule.

### 6. Multi-Label Classification
Tox21 has **12 assays**. Each molecule gets 12 binary labels. We output **12 logits** and apply `BCEWithLogitsLoss` independently per task.

### 7. Class Imbalance
Most molecules are non-toxic (label=0). Without correction, models learn to always predict 0. We use `pos_weight` in the loss to penalize missing toxic molecules more heavily.

---

## Checkpoints & Artifacts

- `best_model.pt` — saved when validation macro ROC-AUC improves
- `metrics.json` — per-task ROC-AUC on test set
- `training_log.csv` — epoch-by-epoch loss and val AUC

---

## Common Failure Checklist

- [ ] Loss is NaN → check masking, reduce LR, clip gradients
- [ ] Model predicts all zeros → increase `pos_weight`, verify positives in batch
- [ ] ROC-AUC suspiciously high → suspect data leakage, use scaffold split
- [ ] `edge_index` out of bounds → validate `edge_index.max() < x.size(0)`
- [ ] dtype errors → `x` must be float, `edge_index` must be long, `y` must be float

---

## License

MIT
