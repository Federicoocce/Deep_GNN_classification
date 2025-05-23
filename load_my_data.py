# load_my_data.py
import json
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import os

def process_graph_data(graph_dict, filename_for_warning="unknown.json", item_idx_for_warning=0):
    """Processes a single graph dictionary into a PyG Data object."""
    num_nodes = graph_dict['num_nodes']

    # 1. Node features (x) - ogbg-ppa has no initial node features.
    # We use zeros. Positional Encodings (like RWSE) will be added by GNN+
    # and the LinearNode encoder will process this.
    x = torch.zeros(num_nodes, 1, dtype=torch.long)  # (num_nodes, 1) placeholder

    # 2. Edge index
    edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)

    # 3. Edge attributes - your edge_attr is already float and 7-dim, which is correct for ppa
    edge_attr = torch.tensor(graph_dict['edge_attr'], dtype=torch.float)

    # 4. Labels (y) - THIS IS CRUCIAL for ogbg-ppa (integer 0-36)
    # MODIFY THIS to correctly get your labels.
    if 'y' not in graph_dict or graph_dict['y'] is None:
        # print(f"WARNING: Graph {item_idx_for_warning} in {filename_for_warning} has no 'y' label. Assigning DUMMY label 0.")
        # print("         YOU MUST FIX THIS FOR MEANINGFUL TRAINING ON OGBG-PPA!")
        y_node = 0 # DUMMY LABEL - REPLACE
    else:
        y_node = graph_dict['y'] # Assuming 'y' field contains the integer label

    y = torch.tensor([y_node], dtype=torch.long) # Graph-level label for classification

    # Ensure num_nodes is present in the Data object if not automatically inferred
    data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data_obj.num_nodes = num_nodes # Makes sure num_nodes is explicitly set

    return data_obj

def get_data_splits():
    """Loads train/val/test graph data from local JSON files."""
    base_path = os.path.dirname(os.path.abspath(__file__)) # Assumes JSONs are in the same dir
    train_json_path = os.path.join(base_path, 'train.json')
    test_json_path = os.path.join(base_path, 'test.json')

    # Load training graphs
    all_train_pyg_graphs = []
    with open(train_json_path, 'r') as f:
        train_data_list = json.load(f) # Assuming train.json contains a list of graph dicts
    for i, g_data in enumerate(train_data_list):
        all_train_pyg_graphs.append(process_graph_data(g_data, "train.json", i))

    # Create validation split from training data
    train_labels_for_split = [g.y.item() for g in all_train_pyg_graphs]
    if len(all_train_pyg_graphs) > 1 and len(set(train_labels_for_split)) > 1 and min(torch.bincount(torch.tensor(train_labels_for_split))) > 1 :
        train_pyg_graphs, val_pyg_graphs = train_test_split(
            all_train_pyg_graphs,
            test_size=0.1,  # 10% for validation, adjust as needed
            random_state=42,
            stratify=train_labels_for_split
        )
    else: # Fallback if stratification is not possible (e.g. too few samples or classes)
        print("Warning: Could not stratify validation split. Using random split.")
        train_pyg_graphs, val_pyg_graphs = train_test_split(
            all_train_pyg_graphs,
            test_size=0.1,
            random_state=42
        )

    # Load test graphs
    test_pyg_graphs = []
    with open(test_json_path, 'r') as f:
        test_data_list = json.load(f) # Assuming test.json contains a list of graph dicts
    for i, g_data in enumerate(test_data_list):
        test_pyg_graphs.append(process_graph_data(g_data, "test.json", i))
    
    print(f"Loaded {len(train_pyg_graphs)} training graphs.")
    print(f"Loaded {len(val_pyg_graphs)} validation graphs.")
    print(f"Loaded {len(test_pyg_graphs)} test graphs.")
    
    return train_pyg_graphs, val_pyg_graphs, test_pyg_graphs

if __name__ == '__main__':
    # This part is for testing the loader script itself
    # You would typically import get_data_splits from another script
    train_graphs, val_graphs, test_graphs = get_data_splits()
    
    # Example: print info about the first training graph
    if train_graphs:
        print("\nFirst training graph sample:")
        print(train_graphs[0])
        print(f"  x shape: {train_graphs[0].x.shape}, dtype: {train_graphs[0].x.dtype}")
        print(f"  edge_index shape: {train_graphs[0].edge_index.shape}")
        print(f"  edge_attr shape: {train_graphs[0].edge_attr.shape}, dtype: {train_graphs[0].edge_attr.dtype}")
        print(f"  y: {train_graphs[0].y}, dtype: {train_graphs[0].y.dtype}")
        print(f"  num_nodes: {train_graphs[0].num_nodes}")

    # You can save them as .pt files if loading is slow, but for smaller subsets it's often fine to load on the fly.
    # torch.save(train_graphs, 'train_graphs.pt')
    # torch.save(val_graphs, 'val_graphs.pt')
    # torch.save(test_graphs, 'test_graphs.pt')