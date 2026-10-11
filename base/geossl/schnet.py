"""Self-contained SchNet encoder (vendored from GeoSSL / Geom3D).

Faithful re-implementation of `Geom3D/models/schnet.py` from chao1224/GeoSSL, with two
changes so it runs in the `aevplig` conda env with **no extra dependencies**:

  1. `radius_graph` is re-implemented with `torch.cdist` + a 32-nearest-neighbour cap,
     matching `torch_geometric.nn.radius_graph(..., max_num_neighbors=32)` semantics
     (that is the default GeoSSL's SchNet.forward uses). This removes the `torch_cluster`
     dependency. Our complex graphs are small (<=~550 atoms) so the dense distance
     matrix is cheap.
  2. `ase.data.atomic_masses` is replaced by a plain zeros buffer of the same length
     (only used on the `dipole=True` path, which LBA never takes) so `ase` is not needed.
     The pretrained checkpoint's `atomic_mass` buffer still loads cleanly (strict=True).

Everything else — InteractionBlock / CFConv / GaussianSmearing / ShiftedSoftplus,
the hidden sizes, the mean readout — is byte-identical to the upstream module, so the
released GeoSSL-DDM SchNet state_dict loads with strict=True.
"""
from math import pi as PI

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_scatter import scatter


def radius_graph_cdist(pos, r, batch=None, max_num_neighbors=32):
    """Drop-in for torch_geometric.nn.radius_graph (flow='source_to_target').

    For each target node i, connect the (up to) `max_num_neighbors` nearest source
    nodes j != i with ||pos_i - pos_j|| <= r, restricted to the same graph (batch).
    Returns edge_index [2, E] with edge_index[0]=source (j), edge_index[1]=target (i).
    """
    N = pos.size(0)
    device = pos.device
    if batch is None:
        batch = torch.zeros(N, dtype=torch.long, device=device)

    dist = torch.cdist(pos, pos)                       # (N, N)
    same = batch.unsqueeze(0) == batch.unsqueeze(1)    # same graph
    mask = same & (dist <= r)
    mask.fill_diagonal_(False)                         # no self-loops

    d = dist.clone()
    d[~mask] = float("inf")
    k = min(max_num_neighbors, N)
    knn_d, knn_j = torch.topk(d, k, dim=1, largest=False)   # k nearest per target row i
    valid = torch.isfinite(knn_d)
    rows_i = torch.arange(N, device=device).unsqueeze(1).expand(-1, k)[valid]  # target i
    cols_j = knn_j[valid]                                                       # source j
    edge_index = torch.stack([cols_j, rows_i], dim=0)  # [source, target]
    return edge_index


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(torch.nn.Module):
    """Continuous-filter conv, message passing done explicitly (no MessagePassing base)."""

    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super().__init__()
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        src, dst = edge_index[0], edge_index[1]            # source -> target
        msg = x[src] * W                                   # message from source j
        out = scatter(msg, dst, dim=0, dim_size=x.size(0), reduce="add")
        out = self.lin2(out)
        return out


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNet(torch.nn.Module):
    """GeoSSL SchNet. Returns the 128-d mean-pooled graph representation (pre-head)."""

    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=51,
        cutoff=10.0,
        node_class=9,
        readout="mean",
        max_num_neighbors=32,
    ):
        super().__init__()
        assert readout in ["add", "sum", "mean"]
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.readout = "add" if readout == "sum" else readout
        self.max_num_neighbors = max_num_neighbors

        # placeholder buffer (upstream uses ase.data.atomic_masses; unused for dipole=False)
        self.register_buffer("atomic_mass", torch.zeros(119))

        self.embedding = Embedding(node_class, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            self.interactions.append(
                InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            )

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels, hidden_channels)

    def forward(self, z, pos, batch=None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index = radius_graph_cdist(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        out = scatter(h, batch, dim=0, reduce=self.readout)   # [num_graphs, hidden]
        return out


def load_pretrained_schnet(weight_file, device="cpu"):
    """Instantiate the GeoSSL LBA SchNet and load the released GeoSSL-DDM weights."""
    model = SchNet(
        hidden_channels=128, num_filters=128, num_interactions=6,
        num_gaussians=51, cutoff=10.0, node_class=9, readout="mean",
    )
    sd = torch.load(weight_file, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(device)
