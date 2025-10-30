# ==============================================================
# graph_encoder.py — Multi-Asset Graph Encoder using torch_geometric
# ==============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv
    TORCH_GEO_AVAILABLE = True
except ImportError:
    TORCH_GEO_AVAILABLE = False


class AssetGraphEncoder(nn.Module):
    """
    Graph-based encoder for multi-asset correlation learning.
    Uses torch_geometric (GCN or GAT) if available, else defaults to linear fusion.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_assets: int = 1, use_gat: bool = True):
        super().__init__()
        self.num_assets = num_assets
        self.use_gat = use_gat and TORCH_GEO_AVAILABLE

        if TORCH_GEO_AVAILABLE:
            if self.use_gat:
                self.gcn1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
                self.gcn2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)
            else:
                self.gcn1 = GCNConv(input_dim, hidden_dim)
                self.gcn2 = GCNConv(hidden_dim, output_dim)
        else:
            self.fallback = nn.Sequential(
                nn.Linear(input_dim * num_assets, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x: torch.Tensor, edge_index=None):
        """
        x: Tensor (N_assets, F)
        edge_index: LongTensor (2, E) — if using torch_geometric
        """
        if not TORCH_GEO_AVAILABLE:
            x_flat = x.view(1, -1)
            return self.fallback(x_flat)

        if edge_index is None:
            # Fully connected graph
            src, dst = torch.meshgrid(torch.arange(self.num_assets, device=x.device),
                                      torch.arange(self.num_assets, device=x.device), indexing='ij')
            edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)

        h = F.elu(self.gcn1(x, edge_index))
        h = self.gcn2(h, edge_index)
        return h
