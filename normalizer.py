import torch
from torch_geometric.transforms import BaseTransform


class NormalizeAttrs(BaseTransform):
    """
    Trasforma data.x e data.edge_attr usando scaler pre-calcolati.
    scaler_node: sklearn scaler fit su tutte le node-features del train set
    scaler_edge: sklearn scaler fit su tutte le edge-features del train set
    """
    def __init__(self, scaler_node, scaler_edge):
        self.scaler_node = scaler_node
        self.scaler_edge = scaler_edge

    def __call__(self, data):
        # Normalizza x
        x = data.x.numpy() if isinstance(data.x, torch.Tensor) else data.x
        x_norm = self.scaler_node.transform(x)
        data.x = torch.from_numpy(x_norm).float()
        # Normalizza edge_attr
        ea = data.edge_attr.numpy() if isinstance(data.edge_attr, torch.Tensor) else data.edge_attr
        ea_norm = self.scaler_edge.transform(ea)
        data.edge_attr = torch.from_numpy(ea_norm).float()
        return data
