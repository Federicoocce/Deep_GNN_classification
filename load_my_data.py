# load_my_data.py
import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, degree # Removed to_undirected as it's not used directly here
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tqdm import tqdm

# --- RWSE Hyperparameters ---
RWSE_MAX_K = 16  # Number of k-steps for RW, as in GNN+ Table 13 (RWSE-16)

def get_rw_landing_probs(edge_index, num_nodes, k_max):
    """
    Computes Random Walk landing probabilities for node features.
    Args:
        edge_index: PyG sparse representation of the graph
        num_nodes: Number of nodes in the graph
        k_max: Maximum k-step for which to compute the RW landings
    Returns:
        rw_landing: 2D Tensor with shape (num_nodes, k_max) with RW landing probs
    """
    if num_nodes == 0: # Handle empty graphs
        return torch.zeros((0, k_max), device=edge_index.device)
    if edge_index.numel() == 0: # Handle graphs with no edges
        return torch.zeros((num_nodes, k_max), device=edge_index.device)

    source, dest = edge_index[0], edge_index[1]
    
    deg = degree(source, num_nodes=num_nodes, dtype=torch.float)
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    # Using dense adjacency for matrix powers. Consider sparse for very large graphs.
    # Max_num_nodes might be an issue if individual graphs are huge.
    # However, to_dense_adj needs max_num_nodes if edge_index doesn't cover all nodes up to num_nodes-1.
    # If your num_nodes is accurate for each graph, it should be fine.
    try:
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    except RuntimeError as e:
        print(f"Error in to_dense_adj for graph with {num_nodes} nodes, edge_index shape {edge_index.shape}: {e}")
        print("This might happen if edge_index contains node indices >= num_nodes.")
        print(f"Max node index in edge_index: {edge_index.max().item() if edge_index.numel() > 0 else -1}")
        # Fallback or re-raise, for now, fallback to zeros.
        return torch.zeros((num_nodes, k_max), device=edge_index.device)


    P_dense = deg_inv.view(-1, 1) * adj

    rws_list = []
    if num_nodes > 0 : # Pk can only be eye if num_nodes > 0
      Pk = torch.eye(num_nodes, device=edge_index.device) # P^0 = I
    else: # Should be caught by earlier num_nodes == 0 check
      return torch.zeros((0,k_max), device=edge_index.device)


    for _ in range(1, k_max + 1): # k_step from 1 to k_max
        if Pk.numel() == 0 or P_dense.numel() == 0 : # Safety check if Pk or P_dense became empty
            # This case indicates an issue with an empty graph that wasn't caught earlier
            # or num_nodes was positive but became zero effective in Pk
            print(f"Warning: Pk or P_dense became empty for graph with num_nodes={num_nodes}. RWSE will be zeros.")
            return torch.zeros((num_nodes, k_max), device=edge_index.device)
        try:
            Pk = Pk @ P_dense 
        except RuntimeError as e:
            print(f"RuntimeError during Pk @ P_dense: {e}. Graph num_nodes: {num_nodes}, Pk shape: {Pk.shape}, P_dense shape: {P_dense.shape}")
            return torch.zeros((num_nodes, k_max), device=edge_index.device) # Fallback

        landing_probs_k = torch.diag(Pk)
        rws_list.append(landing_probs_k)

    if not rws_list:
        return torch.zeros((num_nodes, k_max), device=edge_index.device)

    rw_landing = torch.stack(rws_list, dim=1)
    return rw_landing


def process_graph_data(graph_dict, filename_for_warning="unknown.json", item_idx_for_warning=0):
    """Processes a single graph dictionary into a PyG Data object."""
    num_nodes = graph_dict.get('num_nodes', 0) # Default to 0 if not present

    # Ensure num_nodes is an int
    if not isinstance(num_nodes, int) or num_nodes < 0:
        print(f"Warning: Invalid num_nodes value ({num_nodes}) for graph {item_idx_for_warning} in {filename_for_warning}. Setting to 0.")
        num_nodes = 0
        
    x = torch.zeros(num_nodes, 1, dtype=torch.long) # If num_nodes is 0, x will be (0,1)

    # Edge index and attributes
    # Handle cases where these might be missing or empty for an empty graph
    raw_edge_index = graph_dict.get('edge_index', [])
    raw_edge_attr = graph_dict.get('edge_attr', [])

    if num_nodes == 0: # If graph is empty, edge_index and edge_attr should also be empty
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,7), dtype=torch.float) # Assuming 7 features for edge_attr
    else:
        edge_index = torch.tensor(raw_edge_index, dtype=torch.long)
        edge_attr = torch.tensor(raw_edge_attr, dtype=torch.float)
        # Validate shapes if not empty
        if edge_index.numel() > 0 and edge_index.shape[0] != 2:
            print(f"Warning: Edge index for graph {item_idx_for_warning} in {filename_for_warning} has incorrect shape {edge_index.shape}. Expected (2, num_edges). Treating as no edges.")
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_attr = torch.empty((0,edge_attr.shape[1] if edge_attr.ndim==2 else 7), dtype=torch.float)
        if edge_attr.numel() > 0 and edge_index.shape[1] != edge_attr.shape[0]:
            print(f"Warning: Mismatch between edge_index ({edge_index.shape[1]}) and edge_attr ({edge_attr.shape[0]}) for graph {item_idx_for_warning} in {filename_for_warning}. Edge attributes might be incorrect. ")
            # Attempt to fix or clear edge_attr if dimensions don't match num_edges
            if edge_index.shape[1] == 0 :
                 edge_attr = torch.empty((0,edge_attr.shape[1] if edge_attr.ndim==2 else 7), dtype=torch.float)
            # else:
                # Potentially more complex logic here if trying to salvage, for now, this is a warning.

    # Process y label
    y_val_raw = graph_dict.get('y')
    y_val = 0 # Default dummy label

    if y_val_raw is not None:
        temp_y = y_val_raw
        # Unwrap nested lists like [[[4]]] to 4
        while isinstance(temp_y, list):
            if len(temp_y) == 1:
                temp_y = temp_y[0]
            else: # If it's a list but not a single-element list, it's unexpected for a scalar label
                print(f"Warning: Unexpected y label structure (list with multiple elements or empty list: {temp_y}) for graph {item_idx_for_warning} in {filename_for_warning}. Using default label 0.")
                temp_y = 0 # Fallback
                break 
        
        if isinstance(temp_y, int):
            y_val = temp_y
        else:
            print(f"Warning: y label for graph {item_idx_for_warning} in {filename_for_warning} is not an int after processing: {temp_y} (original: {y_val_raw}). Using default label 0.")
            y_val = 0 # Fallback
    else: # y_val_raw is None
        print(f"Warning: 'y' label missing for graph {item_idx_for_warning} in {filename_for_warning}. Using default label 0. YOU MUST FIX THIS FOR MEANINGFUL TRAINING.")

    y = torch.tensor([y_val], dtype=torch.long)

    data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    # PyG's Data object infers num_nodes from x.shape[0] or edge_index.max()+1
    # Explicitly setting it from your JSON is good if your x/edge_index might be empty for num_nodes > 0
    data_obj.num_nodes = num_nodes

    # --- Calculate RWSE ---
    if data_obj.num_nodes > 0 and data_obj.num_edges > 0: # RWSE only if graph has nodes and edges
        try:
            rwse_pe = get_rw_landing_probs(data_obj.edge_index, data_obj.num_nodes, k_max=RWSE_MAX_K)
            data_obj.rwse_pe = rwse_pe
        except Exception as e:
            print(f"Error calculating RWSE for graph {item_idx_for_warning} in {filename_for_warning}: {e}. Setting RWSE to zeros.")
            data_obj.rwse_pe = torch.zeros((data_obj.num_nodes, RWSE_MAX_K))
    else: # Handle graphs with no nodes or no edges
        data_obj.rwse_pe = torch.zeros((data_obj.num_nodes, RWSE_MAX_K))
        if data_obj.num_nodes > 0 and data_obj.num_edges == 0:
            print(f"Graph {item_idx_for_warning} has {data_obj.num_nodes} nodes but 0 edges. RWSE set to zeros.")


    return data_obj

def get_data_splits(force_reprocess=False):
    """
    Loads train/val/test graph data.
    If pre-processed .pt files exist, loads them. Otherwise, processes from JSONs and saves .pt files.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_path, 'processed_data_ppa')
    os.makedirs(processed_dir, exist_ok=True)

    train_pt_path = os.path.join(processed_dir, 'train_graphs.pt')
    val_pt_path = os.path.join(processed_dir, 'val_graphs.pt')
    test_pt_path = os.path.join(processed_dir, 'test_graphs.pt')

    if not force_reprocess and \
       os.path.exists(train_pt_path) and \
       os.path.exists(val_pt_path) and \
       os.path.exists(test_pt_path):
        print("Loading pre-processed data from .pt files...")
        train_pyg_graphs = torch.load(train_pt_path)
        val_pyg_graphs = torch.load(val_pt_path)
        test_pyg_graphs = torch.load(test_pt_path)
        print("Loaded pre-processed data successfully.")
    else:
        print("Processing data from JSON files (this might take a while)...")
        train_json_path = os.path.join(base_path, 'train.json')
        test_json_path = os.path.join(base_path, 'test.json')

        all_train_pyg_graphs = []
        print("Processing training JSON...")
        try:
            with open(train_json_path, 'r') as f:
                train_data_list = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: {train_json_path} not found!")
            return [], [], []
        except json.JSONDecodeError:
            print(f"ERROR: Could not decode JSON from {train_json_path}. Is it a valid JSON list of graph objects?")
            return [], [], []

        for i, g_data in tqdm(enumerate(train_data_list), total=len(train_data_list), desc="Train Data"):
            if not isinstance(g_data, dict):
                print(f"Warning: Item {i} in train.json is not a dictionary. Skipping.")
                continue
            all_train_pyg_graphs.append(process_graph_data(g_data, "train.json", i))

        # Filter out None if process_graph_data can return None on error
        all_train_pyg_graphs = [g for g in all_train_pyg_graphs if g is not None]

        if not all_train_pyg_graphs:
            print("ERROR: No graphs processed from train.json.")
            return [], [], []

        train_labels_for_split = [g.y.item() for g in all_train_pyg_graphs] # y is tensor([val])
        
        # Stratification check needs at least 2 samples per class for sklearn
        unique_labels, counts = np.unique(train_labels_for_split, return_counts=True)
        can_stratify = (len(all_train_pyg_graphs) > 1 and 
                        len(unique_labels) > 1 and 
                        all(counts >= 2)) # sklearn needs at least 2 instances of each class for stratification in train/test

        if can_stratify:
            try:
                train_pyg_graphs, val_pyg_graphs = train_test_split(
                    all_train_pyg_graphs, test_size=0.1, random_state=42, stratify=train_labels_for_split)
            except ValueError as e: # Catch error if stratification fails (e.g. not enough samples for a class in test set)
                print(f"Warning: Stratification failed ({e}). Using random split for validation.")
                train_pyg_graphs, val_pyg_graphs = train_test_split(
                    all_train_pyg_graphs, test_size=0.1, random_state=42)
        else:
            print("Warning: Conditions for stratification not met (e.g. too few samples, <2 classes, or classes with <2 samples). Using random split for validation.")
            train_pyg_graphs, val_pyg_graphs = train_test_split(
                all_train_pyg_graphs, test_size=0.1, random_state=42)


        test_pyg_graphs = []
        print("Processing test JSON...")
        try:
            with open(test_json_path, 'r') as f:
                test_data_list = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: {test_json_path} not found!")
            # If train was processed, save it at least
            if train_pyg_graphs: torch.save(train_pyg_graphs, train_pt_path)
            if val_pyg_graphs: torch.save(val_pyg_graphs, val_pt_path)
            return train_pyg_graphs, val_pyg_graphs, [] # Return what we have
        except json.JSONDecodeError:
            print(f"ERROR: Could not decode JSON from {test_json_path}.")
            if train_pyg_graphs: torch.save(train_pyg_graphs, train_pt_path)
            if val_pyg_graphs: torch.save(val_pyg_graphs, val_pt_path)
            return train_pyg_graphs, val_pyg_graphs, []


        for i, g_data in tqdm(enumerate(test_data_list), total=len(test_data_list), desc="Test Data"):
            if not isinstance(g_data, dict):
                print(f"Warning: Item {i} in test.json is not a dictionary. Skipping.")
                continue
            test_pyg_graphs.append(process_graph_data(g_data, "test.json", i))
        
        test_pyg_graphs = [g for g in test_pyg_graphs if g is not None]


        print(f"Saving processed data to .pt files in {processed_dir}...")
        if train_pyg_graphs: torch.save(train_pyg_graphs, train_pt_path)
        if val_pyg_graphs: torch.save(val_pyg_graphs, val_pt_path)
        if test_pyg_graphs: torch.save(test_pyg_graphs, test_pt_path)
        print("Saved processed data successfully.")

    print(f"Loaded {len(train_pyg_graphs)} training graphs.")
    print(f"Loaded {len(val_pyg_graphs)} validation graphs.")
    print(f"Loaded {len(test_pyg_graphs)} test graphs.")
    return train_pyg_graphs, val_pyg_graphs, test_pyg_graphs


if __name__ == '__main__':
    # To force re-processing and re-saving (e.g., after code changes or fixing labels):
    # train_graphs, val_graphs, test_graphs = get_data_splits(force_reprocess=True)
    
    # To load existing or process if not found:
    train_graphs, val_graphs, test_graphs = get_data_splits()

    if train_graphs:
        print("\nFirst training graph sample:")
        sample_graph = train_graphs[0]
        print(sample_graph)
        print(f"  x shape: {sample_graph.x.shape}, dtype: {sample_graph.x.dtype}")
        if hasattr(sample_graph, 'rwse_pe'):
            print(f"  rwse_pe shape: {sample_graph.rwse_pe.shape}, dtype: {sample_graph.rwse_pe.dtype}")
        else:
            print("  rwse_pe: Not found")
        print(f"  edge_index shape: {sample_graph.edge_index.shape}")
        print(f"  edge_attr shape: {sample_graph.edge_attr.shape}, dtype: {sample_graph.edge_attr.dtype}")
        print(f"  y: {sample_graph.y}, dtype: {sample_graph.y.dtype}")
        print(f"  num_nodes: {sample_graph.num_nodes}")
        if sample_graph.edge_index.numel() > 0:
            print(f"  max node index in edge_index: {sample_graph.edge_index.max().item()}")

    if val_graphs:
        print(f"\nFirst validation graph y: {val_graphs[0].y}")
    if test_graphs:
        print(f"\nFirst test graph y: {test_graphs[0].y}")