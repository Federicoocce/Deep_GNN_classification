# load_my_data.py (Modified for separate A,B,C,D dataset splits)
import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tqdm import tqdm
import argparse

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
            # else: # Potentially more complex mismatch, could clear both
            #     edge_index = torch.empty((2,0), dtype=torch.long)
            #     edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)

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


def find_single_json_file(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Warning: Directory not found: {directory_path}.")
        return None
    try:
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        if len(json_files) == 1:
            return os.path.join(directory_path, json_files[0])
        elif len(json_files) == 0:
            print(f"Warning: No JSON file found in {directory_path}.")
            return None
        else:
            # Using the first one lexicographically if multiple exist
            print(
                f"Warning: Multiple JSON files found in {directory_path}. Using the first one: {sorted(json_files)[0]}.")
            return os.path.join(directory_path, sorted(json_files)[0])
    except Exception as e:
        print(f"Error accessing directory {directory_path}: {e}")
        return None


def process_single_dataset(dataset_name, original_dataset_base_path, val_split_ratio=0.1):
    """
    Process a single dataset and create train/val/test splits for it.

    Args:
        dataset_name: Name of the dataset (A, B, C, or D)
        original_dataset_base_path: Base path to original dataset directory
        val_split_ratio: Ratio of training data to use for validation

    Returns:
        dict: Contains 'train', 'val', 'test' splits for this dataset
    """
    print(f"-- Processing dataset: {dataset_name} --")

    dataset_splits = {'train': [], 'val': [], 'test': []}

    # --- Load Training Data ---
    train_json_container_path = os.path.join(original_dataset_base_path, dataset_name, 'train.json')
    actual_train_json_file = find_single_json_file(train_json_container_path)

    raw_train_graphs = []
    if actual_train_json_file:
        print(f"  Loading training data from: {actual_train_json_file}")
        try:
            with open(actual_train_json_file, 'r') as f:
                train_json_list = json.load(f)
            for i, g_data in tqdm(enumerate(train_json_list), total=len(train_json_list),
                                  desc=f"  {dataset_name} Train Data"):
                if not isinstance(g_data, dict):
                    continue
                g_data['_source_dataset'] = dataset_name
                processed_graph = process_graph_data(g_data, is_test_set=False,
                                                     graph_idx_info=f"{dataset_name}_train_{i}")
                raw_train_graphs.append(processed_graph)
        except Exception as e:
            print(f"  ERROR loading {actual_train_json_file}: {e}. Skipping.")
    else:
        print(f"  No training JSON file found for dataset {dataset_name}")

    # --- Split Training Data into Train/Val ---
    if len(raw_train_graphs) == 0:
        print(f"  WARNING: No training graphs loaded for dataset {dataset_name}")
        dataset_splits['train'] = []
        dataset_splits['val'] = []
    elif len(raw_train_graphs) == 1:
        print(
            f"  WARNING: Only 1 training sample for dataset {dataset_name}. Using for training, validation will be empty.")
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
                print(f"  Used stratified split for dataset {dataset_name}")
            else:
                dataset_splits['train'], dataset_splits['val'] = train_test_split(
                    raw_train_graphs, test_size=val_split_ratio, random_state=42
                )
                print(f"  Used random split for dataset {dataset_name}")
        except ValueError as e:
            print(f"  Warning: Split failed for dataset {dataset_name} ({e}). Using all for training.")
            dataset_splits['train'] = raw_train_graphs
            dataset_splits['val'] = []

    # --- Load Test Data ---
    test_json_container_path = os.path.join(original_dataset_base_path, dataset_name, 'test.json')
    actual_test_json_file = find_single_json_file(test_json_container_path)

    if actual_test_json_file:
        print(f"  Loading test data from: {actual_test_json_file}")
        try:
            with open(actual_test_json_file, 'r') as f:
                test_json_list = json.load(f)
            for i, g_data in tqdm(enumerate(test_json_list), total=len(test_json_list),
                                  desc=f"  {dataset_name} Test Data"):
                if not isinstance(g_data, dict):
                    continue
                g_data['_source_dataset'] = dataset_name
                processed_graph = process_graph_data(g_data, is_test_set=True,
                                                     graph_idx_info=f"{dataset_name}_test_{i}")
                dataset_splits['test'].append(processed_graph)
        except Exception as e:
            print(f"  ERROR loading {actual_test_json_file}: {e}. Skipping.")
    else:
        print(f"  No test JSON file found for dataset {dataset_name}")

    print(f"  Dataset {dataset_name} - Train: {len(dataset_splits['train'])}, "
          f"Val: {len(dataset_splits['val'])}, Test: {len(dataset_splits['test'])}")

    return dataset_splits


def get_data_splits(force_reprocess=False, val_split_ratio=0.1):
    """
    Load and process graph datasets, creating separate train/val/test splits for each dataset.
    Processes and saves each dataset individually to minimize memory usage.

    Args:
        force_reprocess: If True, reprocess data even if cached files exist
        val_split_ratio: Ratio of training data to use for validation (default 0.1 = 10%)

    Returns:
        dict: Contains separate splits for each dataset
              Format: {
                  'A': {'train': [...], 'val': [...], 'test': [...]},
                  'B': {'train': [...], 'val': [...], 'test': [...]},
                  'C': {'train': [...], 'val': [...], 'test': [...]},
                  'D': {'train': [...], 'val': [...], 'test': [...]}
              }
    """
    script_base_path = os.path.dirname(os.path.abspath(__file__))
    original_dataset_base_path = os.path.join(script_base_path, 'original_dataset')
    processed_dir = os.path.join(script_base_path, 'processed_data_separate')
    os.makedirs(processed_dir, exist_ok=True)

    dataset_names = ['A', 'B', 'C', 'D']

    # Define paths for processed files (separate files for each dataset)
    processed_paths = {}
    for ds_name in dataset_names:
        processed_paths[ds_name] = {
            'train': os.path.join(processed_dir, f'{ds_name}_train_graphs.pt'),
            'val': os.path.join(processed_dir, f'{ds_name}_val_graphs.pt'),
            'test': os.path.join(processed_dir, f'{ds_name}_test_graphs.pt')
        }

    # Check if all pre-processed files exist
    all_files_exist = all(
        os.path.exists(processed_paths[ds_name]['train']) and
        os.path.exists(processed_paths[ds_name]['val']) and
        os.path.exists(processed_paths[ds_name]['test'])
        for ds_name in dataset_names
    )

    if not force_reprocess and all_files_exist:
        print("Loading pre-processed data...")
        loaded_data = {}
        for ds_name in dataset_names:
            loaded_data[ds_name] = {
                'train': torch.load(processed_paths[ds_name]['train'], weights_only=False),
                'val': torch.load(processed_paths[ds_name]['val'], weights_only=False),
                'test': torch.load(processed_paths[ds_name]['test'], weights_only=False)
            }
        print_split_summary_separate(loaded_data)
        return loaded_data

    print("Processing data from JSONs in original_dataset/...")

    # Process each dataset separately to minimize memory usage
    all_splits = {}
    for ds_name in dataset_names:
        print(f"Processing dataset {ds_name}...")

        # Process single dataset
        dataset_splits = process_single_dataset(ds_name, original_dataset_base_path, val_split_ratio)

        # Save immediately to disk
        print(f"Saving dataset {ds_name} to disk...")
        if dataset_splits['train']:
            torch.save(dataset_splits['train'], processed_paths[ds_name]['train'])
        if dataset_splits['val']:
            torch.save(dataset_splits['val'], processed_paths[ds_name]['val'])
        if dataset_splits['test']:
            torch.save(dataset_splits['test'], processed_paths[ds_name]['test'])

        # Store minimal info for summary (just counts, not the actual data)
        all_splits[ds_name] = {
            'train': len(dataset_splits['train']) if dataset_splits['train'] else 0,
            'val': len(dataset_splits['val']) if dataset_splits['val'] else 0,
            'test': len(dataset_splits['test']) if dataset_splits['test'] else 0
        }

        # Clear from memory
        del dataset_splits

        # Optional: Force garbage collection
        import gc
        gc.collect()

        print(f"Dataset {ds_name} processed and saved. Memory cleared.")

    # Now load all datasets back for return (if you need them in memory)
    # If you don't need them all in memory at once, you could modify this part
    print("Loading all processed data for return...")
    final_data = {}
    for ds_name in dataset_names:
        final_data[ds_name] = {
            'train': torch.load(processed_paths[ds_name]['train'], weights_only=False),
            'val': torch.load(processed_paths[ds_name]['val'], weights_only=False),
            'test': torch.load(processed_paths[ds_name]['test'], weights_only=False)
        }

    print_split_summary_separate(final_data)
    return final_data

def print_split_summary_separate(splits_dict):
    """Print summary for separate dataset splits"""
    print("\n--- Data Split Summary (Separate Datasets) ---")
    dataset_names = ['A', 'B', 'C', 'D']

    total_train, total_val, total_test = 0, 0, 0

    for ds_name in dataset_names:
        if ds_name in splits_dict:
            train_count = len(splits_dict[ds_name]['train'])
            val_count = len(splits_dict[ds_name]['val'])
            test_count = len(splits_dict[ds_name]['test'])

            print(f"Dataset {ds_name}:")
            print(f"  Train: {train_count}")
            print(f"  Val:   {val_count}")
            print(f"  Test:  {test_count}")
            print(f"  Total: {train_count + val_count + test_count}")

            total_train += train_count
            total_val += val_count
            total_test += test_count
        else:
            print(f"Dataset {ds_name}: No data found")

    print(f"\nOverall Totals:")
    print(f"  Total Train: {total_train}")
    print(f"  Total Val:   {total_val}")
    print(f"  Total Test:  {total_test}")
    print(f"  Grand Total: {total_train + total_val + total_test}")
    print("--------------------------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Loader for Separate A,B,C,D Dataset Splits")
    parser.add_argument('--force_reprocess_data', action='store_true', help="Force re-processing of all data")
    parser.add_argument('--val_split_ratio', type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    args = parser.parse_args()

    # Load data splits
    print("Attempting to load separate data splits...")
    all_splits = get_data_splits(force_reprocess=args.force_reprocess_data, val_split_ratio=args.val_split_ratio)

    # Show examples from each dataset
    dataset_names = ['A', 'B', 'C', 'D']
    for ds_name in dataset_names:
        if ds_name in all_splits:
            print(f"\n=== Dataset {ds_name} Examples ===")

            # Show train example
            train_graphs = all_splits[ds_name]['train']
            if train_graphs:
                print(f"First training graph from dataset {ds_name}:")
                sg = train_graphs[0]
                print(sg)
                print(f"  Source Dataset: {getattr(sg, '_source_dataset', 'N/A')}")
                print(
                    f"  x: {sg.x.shape}, {sg.x.dtype} | rwse_pe: {sg.rwse_pe.shape if hasattr(sg, 'rwse_pe') and sg.rwse_pe is not None else 'N/A'}")
                print(
                    f"  edge_index: {sg.edge_index.shape} | edge_attr: {sg.edge_attr.shape if sg.edge_attr is not None else 'N/A'}")
                print(f"  y: {sg.y} | num_nodes: {sg.num_nodes}")

            # Show val example
            val_graphs = all_splits[ds_name]['val']
            if val_graphs:
                print(f"First validation graph from dataset {ds_name}:")
                sg = val_graphs[0]
                print(f"  y: {sg.y} | num_nodes: {sg.num_nodes}")

            # Show test example
            test_graphs = all_splits[ds_name]['test']
            if test_graphs:
                print(f"First test graph from dataset {ds_name}:")
                sg = test_graphs[0]
                print(f"  y (should be -1): {sg.y} | num_nodes: {sg.num_nodes}")
        else:
            print(f"\nDataset {ds_name}: No data available")