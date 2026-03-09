"""
model.py — GNN model for Tox21 multi-label toxicity prediction.

Teaching notes
--------------
This file defines a Graph Neural Network (GNN) that:
1. Applies several GIN (Graph Isomorphism Network) convolutional layers
   to update atom (node) embeddings by aggregating neighbor information
   ("message passing").
2. Pools all atom embeddings into a single molecule-level embedding
   using global mean pooling.
3. Passes the molecule embedding through a linear layer to produce
   12 raw scores ("logits"), one per Tox21 task.

Why GIN instead of GCN?
  GIN is theoretically more expressive than GCN — it can distinguish
  more graph structures. For molecular graphs this often gives ~1-3%
  better ROC-AUC with almost no extra cost.

Why logits (not probabilities)?
  BCEWithLogitsLoss (used in train.py) is numerically more stable
  when it receives raw logits rather than sigmoid-squeezed probabilities.
  To get probabilities, apply torch.sigmoid(logits) at inference time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class GINBlock(nn.Module):
    """
    One GIN layer = MLP applied after neighborhood aggregation.
    
    The MLP is: Linear -> BN -> ReLU -> Linear -> BN
    Batch normalization (BN) stabilizes training.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.BatchNorm1d(2 * out_channels),
            nn.ReLU(),
            nn.Linear(2 * out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
        )
        # eps=0 and train_eps=True: GIN learns whether to weight the
        # center node differently from its neighbors.
        self.conv = GINConv(mlp, train_eps=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class Tox21GNN(nn.Module):
    """
    Full GNN model for Tox21.

    Architecture
    ------------
    Input: atom feature matrix x  [N, in_channels]
           edge_index              [2, E]   (bidirectional)
           batch                   [N]      (which graph each node belongs to)

    1. GIN layer 1:  [N, in_channels] -> [N, hidden]
    2. GIN layer 2:  [N, hidden]      -> [N, hidden]
    3. GIN layer 3:  [N, hidden]      -> [N, hidden]
    4. Dropout on node embeddings
    5. Global mean pool: [N, hidden] -> [B, hidden]  (B = batch size)
    6. Linear: [B, hidden] -> [B, num_tasks=12]

    Parameters
    ----------
    in_channels : int
        Number of input atom features (9 for Tox21 via MoleculeNet).
    hidden_channels : int
        Width of hidden layers (default 128).
    num_tasks : int
        Number of output prediction heads (12 for Tox21).
    dropout : float
        Dropout probability applied after pooling.
    num_layers : int
        Number of GIN layers (2-4 recommended).
    """
    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 128,
        num_tasks: int = 12,
        dropout: float = 0.3,
        num_layers: int = 3,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_tasks = num_tasks

        # Build GIN layers
        self.convs = nn.ModuleList()
        self.convs.append(GINBlock(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GINBlock(hidden_channels, hidden_channels))

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_tasks),
        )

    def forward(self, x, edge_index, batch):
        """
        Forward pass.

        Parameters
        ----------
        x          : Tensor [N, in_channels]  — atom features
        edge_index : Tensor [2, E] (dtype=torch.long) — bond connectivity
        batch      : Tensor [N] (dtype=torch.long)    — graph membership

        Returns
        -------
        logits : Tensor [B, num_tasks]  — raw scores (NOT probabilities)
        """
        # ---- Sanity guard (helps debug) ----
        assert edge_index.dtype == torch.long, "edge_index must be torch.long"
        assert x.dtype == torch.float, "x (atom features) must be torch.float"
        if edge_index.numel() > 0:
            assert edge_index.max() < x.size(0), (
                f"edge_index.max()={edge_index.max()} >= num_nodes={x.size(0)}. "
                "Graph is malformed."
            )
        # ------------------------------------

        # Message passing
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # Graph pooling: aggregate all atoms into one molecule vector
        # x shape goes from [N, hidden] to [B, hidden]
        x = global_mean_pool(x, batch)

        # Classify
        logits = self.classifier(x)  # [B, num_tasks]
        return logits


def build_model(
    in_channels: int = 9,
    hidden_channels: int = 128,
    num_tasks: int = 12,
    dropout: float = 0.3,
    num_layers: int = 3,
) -> Tox21GNN:
    """Convenience factory — called from train.py and predict.py."""
    return Tox21GNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_tasks=num_tasks,
        dropout=dropout,
        num_layers=num_layers,
    )


if __name__ == "__main__":
    # Quick sanity check: build model and run a fake forward pass.
    import torch
    model = build_model()
    print(model)
    print(f"\nParameter count: {sum(p.numel() for p in model.parameters()):,}")

    # Fake batch: 5 molecules, each with ~10 atoms, 9 features
    N, B, F_in = 50, 5, 9
    x = torch.randn(N, F_in)
    edge_index = torch.randint(0, N, (2, 100)).long()
    batch = torch.arange(B).repeat_interleave(N // B).long()
    out = model(x, edge_index, batch)
    print(f"Output shape: {out.shape}  (expected [{B}, 12])")
