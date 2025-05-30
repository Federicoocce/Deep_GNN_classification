# load_single_dataset.py
import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import argparse
import gzip
import os

# --- RWSE Hyperparameters ---
RWSE_MAX_K = 16


def get_rw_landing_probs(edge_index, num_nodes, k_max):
    if num_nodes == 0: return torch.zeros((0, k_max), device=edge_index.device)
    if edge_index.numel() == 0: return torch.zeros((num_nodes, k_max), device=edge_index.device)

    if num_nodes > 1000:
        print(f"Info: RWSE for graph with {num_nodes} nodes (dense method). This may be slow.")

    source, _ = edge_index[0], edge_index[1]
    deg = degree(source, num_nodes=num_nodes, dtype=torch.float)
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    try:
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    except RuntimeError as e:
        max_idx = edge_index.max().item() if edge_index.numel() > 0 else -1
        print(f"Error in to_dense_adj: {e}. Max_idx: {max_idx}, Num_nodes: {num_nodes}. Returning zeros for RWSE.")
        return torch.zeros((num_nodes, k_max), device=edge_index.device)

    P_dense = deg_inv.view(-1, 1) * adj
    rws_list = []
    if num_nodes == 0: return torch.zeros((0, k_max), device=edge_index.device)
    Pk = torch.eye(num_nodes, device=edge_index.device)

    for _ in range(1, k_max + 1):
        if Pk.numel() == 0 or P_dense.numel() == 0:
            return torch.zeros((num_nodes, k_max), device=edge_index.device)
        try:
            Pk = Pk @ P_dense
        except RuntimeError as e:
            print(
                f"RuntimeError during Pk @ P_dense: {e}. Shapes Pk:{Pk.shape}, P_dense:{P_dense.shape}. Returning zeros.")
            return torch.zeros((num_nodes, k_max), device=edge_index.device)
        rws_list.append(torch.diag(Pk))

    return torch.stack(rws_list, dim=1) if rws_list else torch.zeros((num_nodes, k_max), device=edge_index.device)


def process_graph_data(graph_dict, is_test_set=False, graph_idx_info=""):
    num_nodes = graph_dict.get('num_nodes', 0)
    if not isinstance(num_nodes, int) or num_nodes < 0: num_nodes = 0

    x = torch.zeros(num_nodes, 1, dtype=torch.long)  # Node features are just type 0

    raw_edge_index = graph_dict.get('edge_index', [])
    raw_edge_attr = graph_dict.get('edge_attr', [])
    edge_attr_dim = graph_dict.get('edge_attr_dim', 7)  # Make flexible, default to 7 for ppa
    # If edge_attr_dim comes from graph_dict, ensure it's used for empty tensor
    if not isinstance(edge_attr_dim, int) or edge_attr_dim <= 0: edge_attr_dim = 7

    if num_nodes == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor(raw_edge_index, dtype=torch.long)
        edge_attr = torch.tensor(raw_edge_attr, dtype=torch.float)
        if edge_index.numel() > 0 and edge_index.shape[0] != 2:
            print(f"Warning: Invalid edge_index shape for graph {graph_idx_info}. Clearing edges.")
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)

        # Validate edge_attr shape if it exists
        if edge_attr.numel() > 0:
            if edge_attr.shape[1] != edge_attr_dim:
                print(
                    f"Warning: Mismatch edge_attr_dim (expected {edge_attr_dim}, got {edge_attr.shape[1]}) for graph {graph_idx_info}. Attempting to adapt or clearing.")
                # Simple fix: if only one edge_attr_dim possible and it's wrong, clear.
                # Or, if edge_attr_dim was a fixed global, this indicates an issue.
                # For now, let's be strict if a specific dim was expected from graph_dict
                if 'edge_attr_dim' in graph_dict:  # If graph_dict specified a dim, it must match
                    edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)
                    if edge_index.shape[1] > 0:  # if edges exist, this is a problem
                        print(f"  Cleared edge_attr for {graph_idx_info} due to dim mismatch with specification.")

        if edge_attr.numel() > 0 and edge_index.shape[1] != edge_attr.shape[0]:
            print(
                f"Warning: Mismatch edge_index/edge_attr count for graph {graph_idx_info}. Clearing edge_attr if no edges, or both if inconsistent.")
            if edge_index.shape[1] == 0:
                edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)

    y_val_raw = graph_dict.get('y')
    y_val = -1
    if is_test_set:  # Test sets explicitly have no labels
        y_val = -1
    elif y_val_raw is not None:
        temp_y = y_val_raw
        while isinstance(temp_y, list):  # Handle [[label]] or [label]
            if len(temp_y) == 1:
                temp_y = temp_y[0]
            else:
                temp_y = -1;
                break  # Malformed y if list has >1 element
        if isinstance(temp_y, int):
            y_val = temp_y
        else:
            y_val = -1  # Malformed y

    if y_val == -1 and not is_test_set:
        # This warning is more critical if it was supposed to be a training/validation sample
        print(f"Warning: 'y' missing or malformed for TRAIN/VAL graph {graph_idx_info}. Using -1. Data issue.")
    y = torch.tensor([y_val], dtype=torch.long)

    data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes)

    if data_obj.num_nodes > 0 and data_obj.num_edges > 0:
        data_obj.rwse_pe = get_rw_landing_probs(data_obj.edge_index, data_obj.num_nodes, k_max=RWSE_MAX_K)
    else:
        data_obj.rwse_pe = torch.zeros((data_obj.num_nodes, RWSE_MAX_K))
    return data_obj


def load_json_from_gz(gz_file_path):
    """
    Load JSON data from a .gz file.

    Args:
        gz_file_path: Path to the .gz file containing JSON data

    Returns:
        Parsed JSON data or None if error
    """
    try:
        with gzip.open(gz_file_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {gz_file_path}: {e}")
        return None


def load_single_dataset(train_gz_path=None, test_gz_path=None, val_split_ratio=0.3, dataset_name="dataset"):
    """
    Load and process a single dataset from .gz files, creating train/val/test splits.

    Args:
        train_gz_path: Path to the training data .gz file (optional)
        test_gz_path: Path to the test data .gz file (optional)
        val_split_ratio: Ratio of training data to use for validation (default 0.1 = 10%)
        dataset_name: Name for logging purposes

    Returns:
        dict: Contains 'train', 'val', 'test' splits
              Format: {'train': [Data objects], 'val': [Data objects], 'test': [Data objects]}
    """
    print(f"-- Processing dataset: {dataset_name} --")

    dataset_splits = {'train': [], 'val': [], 'test': []}

    # --- Load Training Data (Optional) ---
    if train_gz_path is None:
        print(f"No training file provided for dataset {dataset_name}")
        raw_train_graphs = []
    elif not os.path.exists(train_gz_path):
        print(f"Warning: Training file not found: {train_gz_path}")
        raw_train_graphs = []
    else:
        print(f"Loading training data from: {train_gz_path}")
        train_json_list = load_json_from_gz(train_gz_path)

        raw_train_graphs = []
        if train_json_list is not None:
            for i, g_data in tqdm(enumerate(train_json_list), total=len(train_json_list),
                                  desc=f"  {dataset_name} Train Data"):
                if not isinstance(g_data, dict):
                    continue
                processed_graph = process_graph_data(g_data, is_test_set=False,
                                                     graph_idx_info=f"{dataset_name}_train_{i}")
                raw_train_graphs.append(processed_graph)
        else:
            print(f"Failed to load training data from {train_gz_path}")

    # --- Split Training Data into Train/Val ---
    if len(raw_train_graphs) == 0:
        print(f"No training graphs available for dataset {dataset_name}")
        dataset_splits['train'] = []
        dataset_splits['val'] = []
    elif len(raw_train_graphs) == 1:
        print(
            f"Only 1 training sample for dataset {dataset_name}. Using for training, validation will be empty.")
        dataset_splits['train'] = raw_train_graphs
        dataset_splits['val'] = []
    else:
        # Try stratified split
        train_labels = [g.y.item() for g in raw_train_graphs if g.y.item() != -1]

        can_stratify = False
        if train_labels:  # Check if train_labels is not empty
            unique_labels, counts = np.unique(train_labels, return_counts=True)
            if len(unique_labels) > 1 and all(c >= 2 for c in counts):
                can_stratify = True

        try:
            if can_stratify:
                dataset_splits['train'], dataset_splits['val'] = train_test_split(
                    raw_train_graphs, test_size=val_split_ratio, random_state=42, stratify=train_labels
                )
                print(f"Used stratified split for dataset {dataset_name}")
            else:
                dataset_splits['train'], dataset_splits['val'] = train_test_split(
                    raw_train_graphs, test_size=val_split_ratio, random_state=42
                )
                print(f"Used random split for dataset {dataset_name}")
        except ValueError as e:
            print(f"Warning: Split failed for dataset {dataset_name} ({e}). Using all for training.")
            dataset_splits['train'] = raw_train_graphs
            dataset_splits['val'] = []

    # --- Load Test Data (Optional) ---
    if test_gz_path is None:
        print(f"No test file provided for dataset {dataset_name}")
    elif not os.path.exists(test_gz_path):
        print(f"Warning: Test file not found: {test_gz_path}")
    else:
        print(f"Loading test data from: {test_gz_path}")
        test_json_list = load_json_from_gz(test_gz_path)

        if test_json_list is not None:
            for i, g_data in tqdm(enumerate(test_json_list), total=len(test_json_list),
                                  desc=f"  {dataset_name} Test Data"):
                if not isinstance(g_data, dict):
                    continue
                processed_graph = process_graph_data(g_data, is_test_set=True,
                                                     graph_idx_info=f"{dataset_name}_test_{i}")
                dataset_splits['test'].append(processed_graph)
        else:
            print(f"Failed to load test data from {test_gz_path}")

    print(f"Dataset {dataset_name} - Train: {len(dataset_splits['train'])}, "
          f"Val: {len(dataset_splits['val'])}, Test: {len(dataset_splits['test'])}")

    return dataset_splits


def print_split_summary(splits_dict, dataset_name="dataset"):
    """Print summary for dataset splits"""
    print(f"\n--- Data Split Summary ({dataset_name}) ---")

    train_count = len(splits_dict['train'])
    val_count = len(splits_dict['val'])
    test_count = len(splits_dict['test'])

    print(f"Train: {train_count}")
    print(f"Val:   {val_count}")
    print(f"Test:  {test_count}")
    print(f"Total: {train_count + val_count + test_count}")
    print("--------------------------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Loader for Single Dataset from .gz files")
    parser.add_argument('--train_path', type=str, required=True,
                        help="Path to training data .gz file")
    parser.add_argument('--test_path', type=str, default=None,
                        help="Path to test data .gz file (optional)")
    parser.add_argument('--val_split_ratio', type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument('--dataset_name', type=str, default="dataset",
                        help="Name for the dataset (for logging purposes)")
    args = parser.parse_args()

    # Load data splits
    print("Loading dataset splits...")
    splits = load_single_dataset(
        train_gz_path=args.train_path,
        test_gz_path=args.test_path,
        val_split_ratio=args.val_split_ratio,
        dataset_name=args.dataset_name
    )

    # Print summary
    print_split_summary(splits, args.dataset_name)

    # Show examples
    print(f"\n=== Dataset {args.dataset_name} Examples ===")

    # Show train example
    if splits['train']:
        print(f"First training graph:")
        sg = splits['train'][0]
        print(sg)
        print(
            f"  x: {sg.x.shape}, {sg.x.dtype} | rwse_pe: {sg.rwse_pe.shape if hasattr(sg, 'rwse_pe') and sg.rwse_pe is not None else 'N/A'}")
        print(
            f"  edge_index: {sg.edge_index.shape} | edge_attr: {sg.edge_attr.shape if sg.edge_attr is not None else 'N/A'}")
        print(f"  y: {sg.y} | num_nodes: {sg.num_nodes}")

    # Show val example
    if splits['val']:
        print(f"First validation graph:")
        sg = splits['val'][0]
        print(f"  y: {sg.y} | num_nodes: {sg.num_nodes}")

    # Show test example
    if splits['test']:
        print(f"First test graph:")
        sg = splits['test'][0]
        print(f"  y (should be -1): {sg.y} | num_nodes: {sg.num_nodes}")