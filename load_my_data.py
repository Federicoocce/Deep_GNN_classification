# load_my_data.py (Simplified)
import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tqdm import tqdm
import argparse # For --force_reprocess_data

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
    
    x = torch.zeros(num_nodes, 1, dtype=torch.long)
    
    raw_edge_index = graph_dict.get('edge_index', [])
    raw_edge_attr = graph_dict.get('edge_attr', [])
    edge_attr_dim = 7 # Expected for ppa

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
        if edge_attr.numel() > 0 and edge_index.shape[1] != edge_attr.shape[0]:
            print(f"Warning: Mismatch edge_index/edge_attr count for graph {graph_idx_info}. Clearing edge_attr if no edges.")
            if edge_index.shape[1] == 0 :
                 edge_attr = torch.empty((0,edge_attr_dim), dtype=torch.float)

    y_val_raw = graph_dict.get('y')
    y_val = -1 # Placeholder for missing/test labels
    if y_val_raw is not None:
        temp_y = y_val_raw
        while isinstance(temp_y, list):
            if len(temp_y) == 1: temp_y = temp_y[0]
            else: temp_y = -1; break
        if isinstance(temp_y, int): y_val = temp_y
        else: y_val = -1
    
    if y_val == -1 and not is_test_set:
         print(f"Warning: 'y' missing/malformed for TRAIN/VAL graph {graph_idx_info}. Using -1. FIX THIS.")
    y = torch.tensor([y_val], dtype=torch.long)

    data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes)

    if data_obj.num_nodes > 0 and data_obj.num_edges > 0:
        data_obj.rwse_pe = get_rw_landing_probs(data_obj.edge_index, data_obj.num_nodes, k_max=RWSE_MAX_K)
    else:
        data_obj.rwse_pe = torch.zeros((data_obj.num_nodes, RWSE_MAX_K))
    return data_obj

def get_data_splits(force_reprocess=False):
    base_path = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_path, 'processed_data_simple') # Unique dir for this version
    os.makedirs(processed_dir, exist_ok=True)

    paths = {s: os.path.join(processed_dir, f'{s}_graphs.pt') for s in ['train', 'val', 'test']}

    if not force_reprocess and all(os.path.exists(p) for p in paths.values()):
        print("Loading pre-processed data...")
        return {s: torch.load(paths[s], weights_only=False) for s in ['train', 'val', 'test']}
    
    print("Processing data from JSONs...")
    data_splits = {'train': [], 'val': [], 'test': []}
    
    for split_name, is_test in [('train', False), ('test', True)]:
        json_path = os.path.join(base_path, f'{split_name}.json')
        raw_graphs = []
        try:
            with open(json_path, 'r') as f:
                json_list = json.load(f)
        except Exception as e:
            print(f"ERROR loading {json_path}: {e}. Skipping this split."); continue

        for i, g_data in tqdm(enumerate(json_list), total=len(json_list), desc=f"{split_name.capitalize()} Data"):
            if not isinstance(g_data, dict): continue
            raw_graphs.append(process_graph_data(g_data, is_test_set=is_test, graph_idx_info=f"{split_name}_{i}"))
        
        if split_name == 'train':
            if not raw_graphs: print("ERROR: No graphs from train.json."); return {s:[] for s in paths}
            train_labels = [g.y.item() for g in raw_graphs if g.y.item() != -1]
            unique_labels, counts = np.unique(train_labels, return_counts=True)
            can_stratify = len(raw_graphs) > 1 and len(unique_labels) > 1 and (not train_labels or all(c >= 2 for c in counts))
            
            if can_stratify and train_labels:
                try:
                    data_splits['train'], data_splits['val'] = train_test_split(raw_graphs, test_size=0.1, random_state=42, stratify=train_labels)
                except ValueError: # Fallback if stratify fails
                    print("Warning: Stratification failed. Using random split for validation.")
                    data_splits['train'], data_splits['val'] = train_test_split(raw_graphs, test_size=0.1, random_state=42)
            else:
                print("Warning: Conditions for stratification not met. Using random split for validation.")
                data_splits['train'], data_splits['val'] = train_test_split(raw_graphs, test_size=0.1, random_state=42)
        else: # test split
            data_splits['test'] = raw_graphs

    print(f"Saving processed data to {processed_dir}...")
    for s_name, s_data in data_splits.items():
        if s_data: torch.save(s_data, paths[s_name])
    
    print_split_summary(data_splits)
    return data_splits['train'], data_splits['val'], data_splits['test']

def print_split_summary(splits_dict):
    for name, data_list in splits_dict.items():
        print(f"Loaded {len(data_list)} {name} graphs.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Loader Test")
    parser.add_argument('--force_reprocess_data', action='store_true', help="Force re-processing")
    args = parser.parse_args()
    
    splits = get_data_splits(force_reprocess=args.force_reprocess_data)
    train_graphs, val_graphs, test_graphs = splits.get('train',[]), splits.get('val',[]), splits.get('test',[])

    if train_graphs:
        print("\nFirst training graph sample:")
        sg = train_graphs[0]
        print(sg)
        print(f"  x: {sg.x.shape}, {sg.x.dtype} | rwse_pe: {sg.rwse_pe.shape if hasattr(sg, 'rwse_pe') and sg.rwse_pe is not None else 'N/A'}")
        print(f"  edge_index: {sg.edge_index.shape} | edge_attr: {sg.edge_attr.shape}, {sg.edge_attr.dtype}")
        print(f"  y: {sg.y} | num_nodes: {sg.num_nodes}")