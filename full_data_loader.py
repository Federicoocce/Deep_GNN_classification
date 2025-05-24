# load_my_data.py (Modified for A,B,C,D datasets)
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
            print(f"RuntimeError during Pk @ P_dense: {e}. Shapes Pk:{Pk.shape}, P_dense:{P_dense.shape}. Returning zeros.")
            return torch.zeros((num_nodes, k_max), device=edge_index.device)
        rws_list.append(torch.diag(Pk))

    return torch.stack(rws_list, dim=1) if rws_list else torch.zeros((num_nodes, k_max), device=edge_index.device)

def process_graph_data(graph_dict, is_test_set=False, graph_idx_info=""):
    num_nodes = graph_dict.get('num_nodes', 0)
    if not isinstance(num_nodes, int) or num_nodes < 0: num_nodes = 0
    
    x = torch.zeros(num_nodes, 1, dtype=torch.long) # Node features are just type 0
    
    raw_edge_index = graph_dict.get('edge_index', [])
    raw_edge_attr = graph_dict.get('edge_attr', [])
    edge_attr_dim = graph_dict.get('edge_attr_dim', 7) # Make flexible, default to 7 for ppa
    # If edge_attr_dim comes from graph_dict, ensure it's used for empty tensor
    if not isinstance(edge_attr_dim, int) or edge_attr_dim <=0: edge_attr_dim = 7


    if num_nodes == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor(raw_edge_index, dtype=torch.long)
        edge_attr = torch.tensor(raw_edge_attr, dtype=torch.float)
        if edge_index.numel() > 0 and edge_index.shape[0] != 2:
            print(f"Warning: Invalid edge_index shape for graph {graph_idx_info}. Clearing edges.")
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)
        
        # Validate edge_attr shape if it exists
        if edge_attr.numel() > 0:
            if edge_attr.shape[1] != edge_attr_dim:
                 print(f"Warning: Mismatch edge_attr_dim (expected {edge_attr_dim}, got {edge_attr.shape[1]}) for graph {graph_idx_info}. Attempting to adapt or clearing.")
                 # Simple fix: if only one edge_attr_dim possible and it's wrong, clear.
                 # Or, if edge_attr_dim was a fixed global, this indicates an issue.
                 # For now, let's be strict if a specific dim was expected from graph_dict
                 if 'edge_attr_dim' in graph_dict: # If graph_dict specified a dim, it must match
                    edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)
                    if edge_index.shape[1] > 0: # if edges exist, this is a problem
                        print(f"  Cleared edge_attr for {graph_idx_info} due to dim mismatch with specification.")
                    
        if edge_attr.numel() > 0 and edge_index.shape[1] != edge_attr.shape[0]:
            print(f"Warning: Mismatch edge_index/edge_attr count for graph {graph_idx_info}. Clearing edge_attr if no edges, or both if inconsistent.")
            if edge_index.shape[1] == 0 :
                 edge_attr = torch.empty((0,edge_attr_dim), dtype=torch.float)
            # else: # Potentially more complex mismatch, could clear both
            #     edge_index = torch.empty((2,0), dtype=torch.long)
            #     edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)


    y_val_raw = graph_dict.get('y')
    y_val = -1 
    if is_test_set: # Test sets explicitly have no labels
        y_val = -1
    elif y_val_raw is not None:
        temp_y = y_val_raw
        while isinstance(temp_y, list): # Handle [[label]] or [label]
            if len(temp_y) == 1: temp_y = temp_y[0]
            else: temp_y = -1; break # Malformed y if list has >1 element
        if isinstance(temp_y, int): y_val = temp_y
        else: y_val = -1 # Malformed y
    
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
            print(f"Warning: Multiple JSON files found in {directory_path}. Using the first one: {sorted(json_files)[0]}.")
            return os.path.join(directory_path, sorted(json_files)[0])
    except Exception as e:
        print(f"Error accessing directory {directory_path}: {e}")
        return None

def get_data_splits(force_reprocess=False):
    script_base_path = os.path.dirname(os.path.abspath(__file__))
    original_dataset_base_path = os.path.join(script_base_path, 'original_dataset')
    processed_dir = os.path.join(script_base_path, 'processed_data_multids') # New unique dir
    os.makedirs(processed_dir, exist_ok=True)

    dataset_names = ['A', 'B', 'C', 'D']
    
    # Define paths for processed files
    processed_paths = {
        'all_train': os.path.join(processed_dir, 'all_train_graphs.pt'),
        'val': os.path.join(processed_dir, 'val_graphs.pt'),
        'tests': {name: os.path.join(processed_dir, f'test_{name}_graphs.pt') for name in dataset_names}
    }

    # Check if all pre-processed files exist
    all_files_exist = os.path.exists(processed_paths['all_train']) and \
                      os.path.exists(processed_paths['val']) and \
                      all(os.path.exists(p) for p in processed_paths['tests'].values())

    if not force_reprocess and all_files_exist:
        print("Loading pre-processed data...")
        loaded_data = {'train': torch.load(processed_paths['all_train'], weights_only=False),
                       'val': torch.load(processed_paths['val'], weights_only=False)}
        for name in dataset_names:
            loaded_data[f'test_{name}'] = torch.load(processed_paths['tests'][name], weights_only=False)
        print_split_summary(loaded_data)
        return loaded_data
    
    print("Processing data from JSONs in original_dataset/...")
    
    all_raw_train_graphs = []
    raw_test_graphs_per_dataset = {name: [] for name in dataset_names}

    for ds_name in dataset_names:
        print(f"-- Processing dataset: {ds_name} --")
        
        # --- Training data for this dataset ---
        train_json_container_path = os.path.join(original_dataset_base_path, ds_name, 'train.json')
        actual_train_json_file = find_single_json_file(train_json_container_path)
        
        if actual_train_json_file:
            print(f"  Loading training data from: {actual_train_json_file}")
            try:
                with open(actual_train_json_file, 'r') as f:
                    train_json_list = json.load(f)
                for i, g_data in tqdm(enumerate(train_json_list), total=len(train_json_list), desc=f"  {ds_name} Train Data"):
                    if not isinstance(g_data, dict): continue
                    # Add dataset origin info for debugging
                    g_data['_source_dataset'] = ds_name 
                    all_raw_train_graphs.append(process_graph_data(g_data, is_test_set=False, graph_idx_info=f"{ds_name}_train_{i}"))
            except Exception as e:
                print(f"  ERROR loading {actual_train_json_file}: {e}. Skipping.")
        else:
            print(f"  No training JSON file found for dataset {ds_name} in {train_json_container_path}")

        # --- Test data for this dataset ---
        test_json_container_path = os.path.join(original_dataset_base_path, ds_name, 'test.json')
        actual_test_json_file = find_single_json_file(test_json_container_path)

        if actual_test_json_file:
            print(f"  Loading test data from: {actual_test_json_file}")
            try:
                with open(actual_test_json_file, 'r') as f:
                    test_json_list = json.load(f)
                for i, g_data in tqdm(enumerate(test_json_list), total=len(test_json_list), desc=f"  {ds_name} Test Data"):
                    if not isinstance(g_data, dict): continue
                    g_data['_source_dataset'] = ds_name 
                    raw_test_graphs_per_dataset[ds_name].append(process_graph_data(g_data, is_test_set=True, graph_idx_info=f"{ds_name}_test_{i}"))
            except Exception as e:
                print(f"  ERROR loading {actual_test_json_file}: {e}. Skipping.")
        else:
            print(f"  No test JSON file found for dataset {ds_name} in {test_json_container_path}")

    # Prepare final splits dictionary
    final_splits = {}

    # Split combined training data into train and validation
    if not all_raw_train_graphs:
        print("ERROR: No training graphs loaded from any dataset. Cannot create train/val splits.")
        final_splits['train'] = []
        final_splits['val'] = []
    else:
        train_labels = [g.y.item() for g in all_raw_train_graphs if g.y.item() != -1] # Ensure y is not -1
        
        # Check if stratification is possible
        can_stratify = False
        if len(all_raw_train_graphs) > 1 and train_labels: # Check if train_labels is not empty
            unique_labels, counts = np.unique(train_labels, return_counts=True)
            if len(unique_labels) > 1 and all(c >= 2 for c in counts): # Need at least 2 samples per class for stratification
                can_stratify = True
        
        if can_stratify:
            try:
                final_splits['train'], final_splits['val'] = train_test_split(
                    all_raw_train_graphs, test_size=0.1, random_state=42, stratify=train_labels
                )
            except ValueError as e:
                print(f"Warning: Stratification failed ({e}). Using random split for validation.")
                final_splits['train'], final_splits['val'] = train_test_split(
                    all_raw_train_graphs, test_size=0.1, random_state=42
                )
        else:
            print("Warning: Conditions for stratification not met (e.g., too few samples, <2 classes, or single class in labels). Using random split for validation.")
            if len(all_raw_train_graphs) >= 2 : # train_test_split needs at least 2 samples
                 final_splits['train'], final_splits['val'] = train_test_split(
                    all_raw_train_graphs, test_size=0.1, random_state=42
                )
            elif len(all_raw_train_graphs) == 1:
                print("Warning: Only 1 training sample. Using it for training, validation will be empty.")
                final_splits['train'] = all_raw_train_graphs
                final_splits['val'] = []
            else: # Should not happen if all_raw_train_graphs was checked before
                final_splits['train'] = []
                final_splits['val'] = []


    # Add test sets to final_splits
    for ds_name in dataset_names:
        final_splits[f'test_{ds_name}'] = raw_test_graphs_per_dataset[ds_name]

    # Save processed data
    print(f"Saving processed data to {processed_dir}...")
    if final_splits.get('train'): torch.save(final_splits['train'], processed_paths['all_train'])
    if final_splits.get('val'): torch.save(final_splits['val'], processed_paths['val'])
    for ds_name in dataset_names:
        if final_splits.get(f'test_{ds_name}'):
            torch.save(final_splits[f'test_{ds_name}'], processed_paths['tests'][ds_name])
    
    print_split_summary(final_splits)
    return final_splits

def print_split_summary(splits_dict):
    print("\n--- Data Split Summary ---")
    if 'train' in splits_dict:
        print(f"Total combined training graphs: {len(splits_dict['train'])}")
    if 'val' in splits_dict:
        print(f"Total validation graphs: {len(splits_dict['val'])}")
    
    dataset_names = ['A', 'B', 'C', 'D']
    for name in dataset_names:
        key = f'test_{name}'
        if key in splits_dict:
            print(f"Test graphs for dataset {name}: {len(splits_dict[key])}")
    print("-------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Loader for A,B,C,D Datasets")
    parser.add_argument('--force_reprocess_data', action='store_true', help="Force re-processing of all data")
    args = parser.parse_args()
    
    # Example of how to call and use the function
    print("Attempting to load data splits...")
    all_splits = get_data_splits(force_reprocess=args.force_reprocess_data)

    train_graphs = all_splits.get('train', [])
    val_graphs = all_splits.get('val', [])
    
    if train_graphs:
        print("\nFirst combined training graph sample:")
        sg = train_graphs[0]
        print(sg)
        print(f"  Source Dataset (if available): {getattr(sg, '_source_dataset', 'N/A')}") # Example of accessing added info
        print(f"  x: {sg.x.shape}, {sg.x.dtype} | rwse_pe: {sg.rwse_pe.shape if hasattr(sg, 'rwse_pe') and sg.rwse_pe is not None else 'N/A'}")
        print(f"  edge_index: {sg.edge_index.shape} | edge_attr: {sg.edge_attr.shape if sg.edge_attr is not None else 'N/A'}, {sg.edge_attr.dtype if sg.edge_attr is not None else 'N/A'}")
        print(f"  y: {sg.y} | num_nodes: {sg.num_nodes}")

    for ds_name_iter in ['A', 'B', 'C', 'D']:
        test_graphs_current_ds = all_splits.get(f'test_{ds_name_iter}', [])
        if test_graphs_current_ds:
            print(f"\nFirst test graph sample for dataset {ds_name_iter}:")
            sg_test = test_graphs_current_ds[0]
            print(sg_test)
            print(f"  Source Dataset (if available): {getattr(sg_test, '_source_dataset', 'N/A')}")
            print(f"  x: {sg_test.x.shape}, {sg_test.x.dtype} | rwse_pe: {sg_test.rwse_pe.shape if hasattr(sg_test, 'rwse_pe') and sg_test.rwse_pe is not None else 'N/A'}")
            print(f"  edge_index: {sg_test.edge_index.shape} | edge_attr: {sg_test.edge_attr.shape if sg_test.edge_attr is not None else 'N/A'}, {sg_test.edge_attr.dtype if sg_test.edge_attr is not None else 'N/A'}")
            print(f"  y (should be -1 for test): {sg_test.y} | num_nodes: {sg_test.num_nodes}")
        else:
            print(f"\nNo test graphs found for dataset {ds_name_iter}.")