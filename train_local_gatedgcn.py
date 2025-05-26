# train_local_gatedgcn.py (Simplified for full_data_loader)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, Linear
import torch_geometric.utils
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Batch # Added Dataset, Batch
from torch_geometric.transforms import BaseTransform # Added BaseTransform
from sklearn.metrics import accuracy_score
import time
import numpy as np
import pandas as pd
import argparse
import os

# Import from your new data loader file
from full_data_loader import get_data_splits, RWSE_MAX_K

# --- Hyperparameters (reflecting your last run's output where possible) ---
NUM_CLASSES = 6 # Adjust if your combined training data has a different number of classes
GNN_LAYERS = 2
GNN_HIDDEN_DIM = 256
GNN_DROPOUT = 0.3
NODE_EMBEDDING_DIM = 256
EDGE_EMBEDDING_DIM = 256

NODE_CATEGORY_COUNT = 2
EDGE_FEATURE_DIM = 7

USE_RWSE_PE = False
PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0

USE_RESIDUAL = True
USE_FFN = False
USE_BATCHNORM = True

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1.0e-5 # Already present and used correctly
EPOCHS = 300
BATCH_SIZE = 32
NUM_WARMUP_EPOCHS = 10

# --- NEW Hyperparameters for Noise Handling ---
LABEL_SMOOTHING_FACTOR = 0.1 # Example value, tune this
EDGE_DROP_PROBABILITY = 0.1 # Example value, tune this (0 means no edge dropping)
# --- END NEW Hyperparameters ---


# --- NEW: Edge Dropping Transform ---
class DropEdges(BaseTransform):
    def __init__(self, p=0.1, training_only=True):
        self.p = p
        self.training_only = training_only # To control if it applies outside of model.train()

    def __call__(self, data):
        # This check is somewhat redundant if applied only to training loader where model.train() is true
        # but good for general use of the transform.
        # However, the model.training status is not directly available here.
        # We rely on this transform being applied *only* to the training dataset.
        if self.p == 0: # No dropping if probability is 0
            return data
            
        if not hasattr(data, 'edge_index') or data.edge_index is None or data.edge_index.size(1) == 0:
            return data

        num_edges = data.edge_index.size(1)
        mask = torch.rand(num_edges, device=data.edge_index.device) > self.p

        # It's crucial to clone the data object if you are modifying it
        # and the original data object is part of a dataset that might be reused.
        new_data = data.clone()
        new_data.edge_index = data.edge_index[:, mask]
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            new_data.edge_attr = data.edge_attr[mask]
        return new_data

# --- NEW: Custom Dataset to wrap a list of Data objects and apply transforms ---
class ListDataset(Dataset):
    def __init__(self, data_list, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        # Filter out None or problematic graphs if any (though full_data_loader should handle this)
        self.data_list = [g for g in data_list if g is not None]
        if len(self.data_list) != len(data_list):
            print(f"Warning: Filtered out {len(data_list) - len(self.data_list)} None graphs in ListDataset.")


    def len(self):
        return len(self.data_list)

    def get(self, idx):
        # The transform is applied here by the parent Dataset class when get is called
        return self.data_list[idx]


# --- StandaloneGatedGCNLayer (Same as your provided version) ---
class StandaloneGatedGCNLayer(torch.nn.Module):
    def __init__(self, in_dim_node, in_dim_edge, out_dim, dropout, residual, ffn_enabled,
                 batchnorm_enabled, act_fn_constructor, aggr='add', **kwargs):
        super().__init__(**kwargs)
        self.in_dim_node, self.in_dim_edge, self.out_dim = in_dim_node, in_dim_edge, out_dim
        self.activation = act_fn_constructor()
        self.A, self.B = Linear(in_dim_node, out_dim, bias=True), Linear(in_dim_node, out_dim, bias=True)
        self.C = Linear(in_dim_edge, out_dim, bias=True)
        self.D, self.E = Linear(in_dim_node, out_dim, bias=True), Linear(in_dim_node, out_dim, bias=True)

        self.act_fn_x, self.act_fn_e = self.activation, self.activation
        self.dropout_rate, self.residual_enabled, self.e_prop = dropout, residual, None
        self.batchnorm_enabled, self.ffn_enabled, self.aggr = batchnorm_enabled, ffn_enabled, aggr

        if self.batchnorm_enabled:
            self.bn_node_x, self.bn_edge_e = nn.BatchNorm1d(out_dim), nn.BatchNorm1d(out_dim)
        
        self.residual_proj_node = Linear(in_dim_node, out_dim, bias=False) if residual and in_dim_node != out_dim else nn.Identity()
        self.residual_proj_edge = Linear(in_dim_edge, out_dim, bias=False) if residual and in_dim_edge != out_dim else nn.Identity()

        if self.ffn_enabled:
            if self.batchnorm_enabled: self.norm1_ffn = nn.BatchNorm1d(out_dim)
            self.ff_linear1, self.ff_linear2 = Linear(out_dim, out_dim * 2), Linear(out_dim * 2, out_dim)
            self.act_fn_ff = act_fn_constructor()
            if self.batchnorm_enabled: self.norm2_ffn = nn.BatchNorm1d(out_dim)
            self.ff_dropout1, self.ff_dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, x_in_node, edge_idx, edge_in_attr):
        x_ident, e_ident = x_in_node, edge_in_attr
        Ax, Bx, Ce, Dx, Ex = self.A(x_in_node), self.B(x_in_node), self.C(edge_in_attr), self.D(x_in_node), self.E(x_in_node)

        if edge_idx.numel() > 0:
            row, col = edge_idx
            e_ij = Dx[row] + Ex[col] + Ce
            self.e_prop = e_ij
            aggr_out = torch_geometric.utils.scatter(torch.sigmoid(e_ij) * Bx[col], row, 0, dim_size=x_in_node.size(0), reduce=self.aggr)
            x_trans, e_trans = Ax + aggr_out, self.e_prop
        else:
            x_trans, e_trans = Ax, torch.zeros((0, self.out_dim), device=x_in_node.device, dtype=x_in_node.dtype)

        if self.batchnorm_enabled:
            x_trans = self.bn_node_x(x_trans)
            if e_trans.numel() > 0: e_trans = self.bn_edge_e(e_trans)
        
        x_trans = self.act_fn_x(x_trans)
        if e_trans.numel() > 0: e_trans = self.act_fn_e(e_trans)
        
        x_trans = F.dropout(x_trans, self.dropout_rate, training=self.training)
        if e_trans.numel() > 0: e_trans = F.dropout(e_trans, self.dropout_rate, training=self.training)

        x_final = self.residual_proj_node(x_ident) + x_trans if self.residual_enabled else x_trans
        e_final = (self.residual_proj_edge(e_ident) + e_trans) if self.residual_enabled and e_trans.numel() > 0 else e_trans
        
        if self.ffn_enabled:
            x_ffn_ident = x_final
            # Ensure batchnorm isn't applied to empty tensors
            x_ffn_proc = self.norm1_ffn(x_ffn_ident) if self.batchnorm_enabled and x_ffn_ident.numel() > 0 else x_ffn_ident
            if x_ffn_proc.numel() > 0 :
                 x_ffn_proc = x_ffn_ident + self._ff_block(x_ffn_proc)
                 x_final = self.norm2_ffn(x_ffn_proc) if self.batchnorm_enabled else x_ffn_proc
            else:
                 x_final = x_ffn_proc
        return x_final, e_final

# --- Model Definition (Simplified __init__) ---
class MyLocalGatedGCN(torch.nn.Module):
    def __init__(self, current_use_rwse_pe, current_pe_dim):
        super().__init__()
        self.use_rwse_pe = current_use_rwse_pe
        
        self.node_encoder = nn.Embedding(num_embeddings=NODE_CATEGORY_COUNT, embedding_dim=NODE_EMBEDDING_DIM)
        self.edge_encoder = Linear(EDGE_FEATURE_DIM, EDGE_EMBEDDING_DIM)

        current_node_dim = NODE_EMBEDDING_DIM
        if self.use_rwse_pe:
            current_node_dim += current_pe_dim
        
        current_edge_dim = EDGE_EMBEDDING_DIM

        self.gnn_layers = nn.ModuleList()
        for i in range(GNN_LAYERS):
            in_node = current_node_dim if i == 0 else GNN_HIDDEN_DIM
            in_edge = current_edge_dim if i == 0 else GNN_HIDDEN_DIM
            self.gnn_layers.append(
                StandaloneGatedGCNLayer(in_node, in_edge, GNN_HIDDEN_DIM, GNN_DROPOUT,
                                        USE_RESIDUAL, USE_FFN, USE_BATCHNORM, lambda: nn.ReLU())
            )
        self.pool = global_mean_pool
        self.head = Linear(GNN_HIDDEN_DIM, NUM_CLASSES)

    def forward(self, data):
        x, edge_idx, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        if x.dtype == torch.long:
            x_base = self.node_encoder(x.squeeze(-1))
        else:
            print(f"Warning: Unexpected node feature type {x.dtype} in model. Attempting to cast to long for nn.Embedding.")
            x_base = self.node_encoder(x.long().squeeze(-1))
            
        e_attr_enc = torch.empty((0, EDGE_EMBEDDING_DIM), device=x.device, dtype=x_base.dtype)
        if hasattr(edge_attr, 'numel') and edge_attr.numel() > 0 :
             if edge_attr.size(0) > 0 :
                expected_edge_dim_from_encoder = self.edge_encoder.weight.shape[1]
                
                if edge_attr.shape[1] == expected_edge_dim_from_encoder:
                    e_attr_enc = self.edge_encoder(edge_attr)
                else:
                    print(f"Warning: Edge feature dim mismatch in model. Expected {expected_edge_dim_from_encoder}, got {edge_attr.shape[1]}. Creating zero edge features.")
                    if edge_idx.numel() > 0:
                        num_edges = edge_idx.shape[1]
                        e_attr_enc = torch.zeros((num_edges, EDGE_EMBEDDING_DIM), device=x.device, dtype=x_base.dtype)

        current_x = x_base
        if self.use_rwse_pe and hasattr(data, 'rwse_pe') and data.rwse_pe is not None and data.rwse_pe.numel() > 0:
            pe = data.rwse_pe.float().to(x_base.device) # Ensure float and correct device
            if x_base.size(0) == pe.size(0): # Check for node count consistency
                current_x = torch.cat([x_base, pe], dim=-1)
            elif x_base.size(0) > 0 and pe.size(0) > 0: # If both have nodes but mismatch
                print(f"Warning: RWSE PE node count ({pe.size(0)}) mismatch with x_base ({x_base.size(0)}). RWSE PE not used for this batch.")
        
        current_e = e_attr_enc

        for layer in self.gnn_layers:
            current_x, current_e = layer(current_x, edge_idx, current_e)
        
        graph_x = self.pool(current_x, batch)
        return self.head(graph_x)

# --- Training and Evaluation Functions ---
def train_epoch(model, loader, optimizer, criterion, device): # Removed edge_drop_transform as arg
    model.train()
    total_loss, processed_graphs = 0, 0
    for data in loader: # Data from loader will now be pre-transformed by DropEdges
        data = data.to(device)
        # Edge dropping is now handled by the ListDataset's transform for training data

        optimizer.zero_grad()
        out = model(data)
        target_y = data.y.squeeze()
        if target_y.ndim == 0: target_y = target_y.unsqueeze(0)
        loss = criterion(out, target_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        processed_graphs += data.num_graphs
    return total_loss / processed_graphs if processed_graphs else 0

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, is_test_set_preds_only=False):
    model.eval()
    total_loss, processed_graphs = 0, 0
    all_preds_list, all_labels_list = [], []

    for data in loader:
        data = data.to(device)
        out = model(data)
        preds = out.argmax(dim=1)
        all_preds_list.append(preds.cpu())

        if not is_test_set_preds_only:
            target_y = data.y.squeeze()
            if target_y.ndim == 0: target_y = target_y.unsqueeze(0)
            valid_targets = target_y != -1
            if valid_targets.any(): # Check if there's at least one valid target
                loss = criterion(out[valid_targets], target_y[valid_targets])
                total_loss += loss.item() * torch.sum(valid_targets).item() # Use sum of valid targets for count
            all_labels_list.append(target_y.cpu()) # Append all, filtering happens later
        processed_graphs += data.num_graphs
    
    if is_test_set_preds_only:
        return torch.cat(all_preds_list).numpy() if all_preds_list else np.array([])

    if not all_labels_list and not is_test_set_preds_only : return 0, 0 # Guard clause
    
    all_preds_np = torch.cat(all_preds_list).numpy() if all_preds_list else np.array([])
    all_labels_np = torch.cat(all_labels_list).numpy() if all_labels_list else np.array([])
    
    valid_indices = all_labels_np != -1
    accuracy = 0
    num_valid_samples = np.sum(valid_indices)
    if num_valid_samples > 0:
        accuracy = accuracy_score(all_labels_np[valid_indices], all_preds_np[valid_indices])
        
    effective_loss = total_loss / num_valid_samples if num_valid_samples > 0 else 0
    return effective_loss, accuracy

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GatedGCN Training with A,B,C,D datasets')
    parser.add_argument('--force_reprocess_data', action='store_true', help="Force re-processing of data")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument('--no_rwse', action='store_false', dest='use_rwse_pe_cli', help="Disable RWSE Positional Encoding.")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="Learning rate.")
    # --- NEW CLI Arguments ---
    parser.add_argument('--label_smoothing', type=float, default=LABEL_SMOOTHING_FACTOR, help="Label smoothing factor (0.0 for no smoothing).")
    parser.add_argument('--edge_drop_prob', type=float, default=EDGE_DROP_PROBABILITY, help="Probability of dropping an edge during training (0.0 for no dropping).")
    # --- END NEW CLI Arguments ---
    parser.set_defaults(use_rwse_pe_cli=USE_RWSE_PE)

    cli_args = parser.parse_args()

    EPOCHS = cli_args.epochs
    LEARNING_RATE = cli_args.lr
    USE_RWSE_PE = cli_args.use_rwse_pe_cli
    PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0
    # --- Apply CLI args to new hyperparameters ---
    LABEL_SMOOTHING_FACTOR = cli_args.label_smoothing
    EDGE_DROP_PROBABILITY = cli_args.edge_drop_prob
    # --- END ---

    print(f"--- Configuration ---")
    print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}")
    print(f"Model: Layers={GNN_LAYERS}, HiddenDim={GNN_HIDDEN_DIM}, NodeEmb={NODE_EMBEDDING_DIM}, EdgeEmb={EDGE_EMBEDDING_DIM}")
    print(f"Node Category Count: {NODE_CATEGORY_COUNT}, Edge Feature Dim: {EDGE_FEATURE_DIM}")
    print(f"Dropout: {GNN_DROPOUT}, Residual: {USE_RESIDUAL}, FFN: {USE_FFN}, BatchNorm: {USE_BATCHNORM}")
    print(f"RWSE Used: {USE_RWSE_PE}, PE Dim (if used): {PE_DIM}")
    # --- Print new hyperparams ---
    print(f"Label Smoothing: {LABEL_SMOOTHING_FACTOR}")
    print(f"Edge Drop Probability (Train): {EDGE_DROP_PROBABILITY}")
    # --- END ---
    print(f"--------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data from full_data_loader...")
    all_loaded_splits = get_data_splits(force_reprocess=cli_args.force_reprocess_data)
    
    train_graphs = all_loaded_splits.get('train', [])
    val_graphs = all_loaded_splits.get('val', [])
    
    test_datasets = {}
    for ds_name_iter in ['A', 'B', 'C', 'D']:
        test_datasets[ds_name_iter] = all_loaded_splits.get(f'test_{ds_name_iter}', [])

    if not train_graphs:
        print("No training data loaded. Exiting.")
        exit()
    
    if train_graphs and hasattr(train_graphs[0], 'edge_attr') and train_graphs[0].edge_attr is not None and train_graphs[0].edge_attr.numel() > 0:
        actual_edge_dim = train_graphs[0].edge_attr.shape[1]
        if actual_edge_dim != EDGE_FEATURE_DIM:
            print(f"WARNING: Loaded training data has edge_attr dim {actual_edge_dim}, but model expects {EDGE_FEATURE_DIM}. Ensure consistency.")
    elif EDGE_FEATURE_DIM > 0 :
        print(f"WARNING: Model expects edge_attr dim {EDGE_FEATURE_DIM}, but first training graph has no/empty edge_attr.")

    # --- MODIFIED: Apply EdgeDropping Transform to training data ---
    train_transform = None
    if EDGE_DROP_PROBABILITY > 0:
        train_transform = DropEdges(p=EDGE_DROP_PROBABILITY)
        print(f"Applying EdgeDropping (p={EDGE_DROP_PROBABILITY}) to training data.")
    
    train_dataset = ListDataset(train_graphs, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # --- END MODIFICATION ---

    # Validation and Test loaders do not get augmentation
    val_dataset = ListDataset(val_graphs) # Wrap in ListDataset for consistency, no transform
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if val_graphs else None


    model = MyLocalGatedGCN(
        current_use_rwse_pe=USE_RWSE_PE,
        current_pe_dim=PE_DIM
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # --- MODIFIED: Add Label Smoothing to Criterion ---
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=LABEL_SMOOTHING_FACTOR)
    # --- END MODIFICATION ---
    
    scheduler = None
    if EPOCHS > NUM_WARMUP_EPOCHS :
        def lr_lambda_fn(current_epoch_internal):
            if current_epoch_internal < NUM_WARMUP_EPOCHS:
                return float(current_epoch_internal + 1) / float(NUM_WARMUP_EPOCHS + 1)
            else:
                progress = float(current_epoch_internal - NUM_WARMUP_EPOCHS) / float(max(1, EPOCHS - NUM_WARMUP_EPOCHS))
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_fn)

    print("\nStarting training...")
    best_val_acc = 0.0
    
    model_save_dir = 'models' # Ensure this directory exists or is created
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'best_gatedgcn_multids.pth')

    for epoch_iter in range(1, EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        val_loss, val_acc = (0,0)
        if val_loader and val_graphs:
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        if val_loader and val_graphs and val_acc > best_val_acc :
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"*** Best val_acc: {best_val_acc:.4f} (Epoch {epoch_iter}). Model saved to {model_save_path} ***")

        current_lr_val = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch_iter:02d}/{EPOCHS:02d} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.4f} | LR: {current_lr_val:.1e} | Time: {(time.time()-start_time):.2f}s")
        if scheduler: scheduler.step()
    
    print("\nTraining finished.")
    if os.path.exists(model_save_path):
        print(f"Loading best model from {model_save_path} for test predictions...")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print("Warning: No best model saved. Using the model from the last epoch for testing.")
    
    print("\n--- Generating Test Predictions ---")
    for ds_name, current_test_graphs in test_datasets.items():
        if current_test_graphs:
            print(f"Generating predictions for testset_{ds_name}...")
            # Wrap test data in ListDataset (no transform) for consistency with DataLoader
            current_test_dataset = ListDataset(current_test_graphs)
            current_test_loader = DataLoader(current_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            
            test_predictions_array = eval_epoch(model, current_test_loader, criterion, device, is_test_set_preds_only=True)
            
            num_test_samples = len(test_predictions_array)
            if num_test_samples > 0:
                ids = np.arange(1, num_test_samples + 1) # Changed from arange(num_test_samples) for 1-based IDs
                predictions_df = pd.DataFrame({'id': ids, 'pred': test_predictions_array})
                
                output_predictions_filename = f'testset_{ds_name}.csv'
                predictions_df.to_csv(output_predictions_filename, index=False)
                print(f"  Test predictions for {ds_name} saved to {output_predictions_filename}")
            else:
                print(f"  No test predictions generated for {ds_name} (dataset might be empty or predictions array is empty).")
        else:
            print(f"No test data found for dataset {ds_name}. Skipping predictions.")
    print("---------------------------------")