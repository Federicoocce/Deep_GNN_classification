# training_utils.py
import torch
from torch_geometric.data import Batch


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving"""

    def __init__(self, patience=50, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False

        if self.mode == 'max':
            improved = val_score > self.best_score + self.min_delta
        else:
            improved = val_score < self.best_score - self.min_delta

        if improved:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


def drop_edges(batch: Batch, drop_prob: float = 0.2) -> Batch:
    """
    Drop edges from a PyG Batch by processing each Data object individually.
    Returns a valid Batch usable with .to_data_list().
    """
    if drop_prob == 0.0:  # Optimization: if no drop, return original
        return batch

    data_list = batch.to_data_list()
    new_data_list = []

    for data in data_list:
        if data.edge_index is None or data.edge_index.size(1) == 0:
            new_data_list.append(data.clone())  # No edges to drop
            continue

        edge_index = data.edge_index
        num_edges = edge_index.size(1)

        keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_prob
        new_edge_index = edge_index[:, keep_mask]

        new_data = data.clone()
        new_data.edge_index = new_edge_index

        if 'edge_attr' in data and data.edge_attr is not None:
            new_data.edge_attr = data.edge_attr[keep_mask]

        new_data_list.append(new_data)

    return Batch.from_data_list(new_data_list)