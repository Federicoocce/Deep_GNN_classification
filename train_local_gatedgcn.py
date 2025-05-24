# train_local_gatedgcn.py (Modified for full_data_loader)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, Linear
import torch_geometric.utils 
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
import time
import numpy as np
import pandas as pd # For CSV output
import argparse
import os

# Import from the new data loader file
from full_data_loader import get_data_splits, RWSE_MAX_K # MODIFIED IMPORT

# --- VERY SMALL Hyperparameters for VERY FAST CPU TESTING ---
NUM_CLASSES = 6 # Adjust if your datasets A,B,C,D have different num_classes for training
GNN_LAYERS = 2
GNN_HIDDEN_DIM = 512     
GNN_DROPOUT = 0.3
NODE_EMBEDDING_DIM = 256  
EDGE_EMBEDDING_DIM = 256  
USE_RWSE_PE = False         
PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0

# GNN "Plus" Features
USE_RESIDUAL = True
USE_FFN = False
USE_BATCHNORM = True

# Training
LEARNING_RATE = 0.0003     
WEIGHT_DECAY = 1.0e-5
EPOCHS = 300               
BATCH_SIZE = 32
NUM_WARMUP_EPOCHS = 10     

# --- StandaloneGatedGCNLayer (Same as previous) ---
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
            x_ffn_proc = self.norm1_ffn(x_ffn_ident) if self.batchnorm_enabled and x_ffn_ident.numel() > 0 else x_ffn_ident
            if x_ffn_proc.numel() > 0:
                 x_ffn_proc = x_ffn_ident + self._ff_block(x_ffn_proc) 
                 x_final = self.norm2_ffn(x_ffn_proc) if self.batchnorm_enabled else x_ffn_proc
            else: 
                 x_final = x_ffn_proc 
        return x_final, e_final

# --- Model Definition ---
class MyLocalGatedGCN(torch.nn.Module):
    def __init__(self, current_use_rwse_pe, current_pe_dim, node_feat_dim=1, edge_feat_dim=7): # Allow dynamic dims
        super().__init__()
        self.use_rwse_pe = current_use_rwse_pe 
        self.node_encoder = nn.Embedding(node_feat_dim + 1, NODE_EMBEDDING_DIM) # +1 if node features are 0-indexed categories
                                                                            # If node features are already dense, use Linear
        self.edge_encoder = Linear(edge_feat_dim, EDGE_EMBEDDING_DIM) 

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
        self.head = Linear(GNN_HIDDEN_DIM, NUM_CLASSES) # NUM_CLASSES should be consistent with training labels

    def forward(self, data):
        x, edge_idx, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Node encoding
        if x.dtype == torch.long: # Categorical features (like node type)
            x_base = self.node_encoder(x.squeeze(-1))
        else: # Dense features (already floats)
            # If your node features are already NxD floats, you might want a Linear layer instead of Embedding
            # For this example, assuming they are categorical and need embedding
            # If x is already [num_nodes, NODE_EMBEDDING_DIM], then x_base = x, or pass through a Linear
            print(f"Warning: Unexpected node feature type {x.dtype}. Assuming categorical for nn.Embedding.")
            x_base = self.node_encoder(x.long().squeeze(-1)) # Attempt to cast, might fail or be wrong
            
        e_attr_enc = torch.empty((0,EDGE_EMBEDDING_DIM), device=x.device, dtype=x_base.dtype)
        if hasattr(edge_attr, 'numel') and edge_attr.numel() > 0 : 
             if edge_attr.size(0) > 0 : 
                if edge_attr.shape[1] == self.edge_encoder.in_features: # Check consistency
                    e_attr_enc = self.edge_encoder(edge_attr)
                else:
                    print(f"Warning: Edge feature dim mismatch. Expected {self.edge_encoder.in_features}, got {edge_attr.shape[1]}. Skipping edge encoding for this batch.")
                    # Fallback: create zero edge features of correct dimension if edges exist
                    if edge_idx.numel() > 0:
                        num_edges = edge_idx.shape[1]
                        e_attr_enc = torch.zeros((num_edges, EDGE_EMBEDDING_DIM), device=x.device, dtype=x_base.dtype)

        current_x = x_base
        if self.use_rwse_pe and hasattr(data, 'rwse_pe') and data.rwse_pe is not None and data.rwse_pe.numel() > 0:
            pe = data.rwse_pe.float().to(x_base.device)
            if x_base.size(0) == pe.size(0):
                current_x = torch.cat([x_base, pe], dim=-1)
        
        current_e = e_attr_enc

        for layer in self.gnn_layers:
            current_x, current_e = layer(current_x, edge_idx, current_e)
        
        graph_x = self.pool(current_x, batch)
        return self.head(graph_x)

# --- Training and Evaluation Functions (Mostly same) ---
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, processed_graphs = 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        target_y = data.y.squeeze()
        if target_y.ndim == 0: target_y = target_y.unsqueeze(0) # Ensure 1D
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
            if target_y.ndim == 0: target_y = target_y.unsqueeze(0) # Ensure 1D
            valid_targets = target_y != -1
            if valid_targets.any():
                loss = criterion(out[valid_targets], target_y[valid_targets])
                total_loss += loss.item() * torch.sum(valid_targets).item()
            all_labels_list.append(target_y.cpu())
        processed_graphs += data.num_graphs
    
    if is_test_set_preds_only:
        return torch.cat(all_preds_list).numpy() if all_preds_list else np.array([])

    if not all_labels_list and not is_test_set_preds_only : return 0, 0
    
    all_preds_np = torch.cat(all_preds_list).numpy()
    all_labels_np = torch.cat(all_labels_list).numpy()
    
    valid_indices = all_labels_np != -1
    accuracy = 0
    if np.sum(valid_indices) > 0:
        accuracy = accuracy_score(all_labels_np[valid_indices], all_preds_np[valid_indices])
        
    effective_loss = total_loss / np.sum(valid_indices) if np.sum(valid_indices) > 0 else 0
    return effective_loss, accuracy

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GatedGCN Training with A,B,C,D datasets')
    parser.add_argument('--force_reprocess_data', action='store_true', help="Force re-processing of data")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument('--no_rwse', action='store_true', help="Disable RWSE Positional Encoding.")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="Learning rate.")
    # Potentially add args for node/edge feature dims if they vary and aren't auto-detected well
    # parser.add_argument('--node_feat_dim', type=int, default=1, help="Max value for node features if categorical, or dim if dense.")
    # parser.add_argument('--edge_feat_dim', type=int, default=7, help="Dimension of edge features.")


    cli_args = parser.parse_args()

    EPOCHS = cli_args.epochs
    LEARNING_RATE = cli_args.lr
    USE_RWSE_PE = not cli_args.no_rwse 
    PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0
    # NODE_FEAT_INPUT_DIM = cli_args.node_feat_dim # If using args
    # EDGE_FEAT_INPUT_DIM = cli_args.edge_feat_dim # If using args
    NODE_FEAT_INPUT_DIM = 1 # Assuming node features are single integer type (0 for all nodes in example)
    EDGE_FEAT_INPUT_DIM = 7 # Default for ppa, might need to be dynamic if datasets vary


    print(f"--- Configuration ---")
    print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}")
    print(f"Model: Layers={GNN_LAYERS}, HiddenDim={GNN_HIDDEN_DIM}, NodeEmb={NODE_EMBEDDING_DIM}, EdgeEmb={EDGE_EMBEDDING_DIM}")
    print(f"Node Input Dim (for Embedding): {NODE_FEAT_INPUT_DIM}, Edge Input Dim: {EDGE_FEAT_INPUT_DIM}")
    print(f"Dropout: {GNN_DROPOUT}, Residual: {USE_RESIDUAL}, FFN: {USE_FFN}, BatchNorm: {USE_BATCHNORM}")
    print(f"RWSE Used: {USE_RWSE_PE}, PE Dim (if used): {PE_DIM}")
    print(f"--------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data from full_data_loader...")
    all_loaded_splits = get_data_splits(force_reprocess=cli_args.force_reprocess_data)
    
    train_graphs = all_loaded_splits.get('train', [])
    val_graphs = all_loaded_splits.get('val', [])
    
    # Store test graphs in a dictionary for easier iteration
    test_datasets = {}
    for ds_name in ['A', 'B', 'C', 'D']:
        test_datasets[ds_name] = all_loaded_splits.get(f'test_{ds_name}', [])

    if not train_graphs: 
        print("No training data loaded. Exiting.")
        exit()

    # Dynamically determine edge feature dimension from the first training graph if possible
    if train_graphs and hasattr(train_graphs[0], 'edge_attr') and train_graphs[0].edge_attr is not None and train_graphs[0].edge_attr.numel() > 0:
        EDGE_FEAT_INPUT_DIM = train_graphs[0].edge_attr.shape[1]
        print(f"Auto-detected edge_feat_dim from data: {EDGE_FEAT_INPUT_DIM}")
    else: # Fallback if no edge_attr in first graph or no train_graphs
        print(f"Warning: Could not auto-detect edge_feat_dim. Using default: {EDGE_FEAT_INPUT_DIM}. Ensure this is correct.")

    # Determine max node feature value for nn.Embedding size
    # Assumes node features are categorical integers starting from 0.
    # If x is a single integer (e.g., all nodes are type 0), then max_node_val = 0.
    # nn.Embedding needs num_embeddings = max_node_val + 1.
    # For the example data where x is torch.zeros(num_nodes, 1), max_node_val is 0. So num_embeddings = 1.
    # The model's nn.Embedding(NODE_FEAT_INPUT_DIM (+1 if 0-indexed), ...) handles this.
    # If NODE_FEAT_INPUT_DIM is set to the number of categories (e.g., 1 if only '0' exists)
    # nn.Embedding(1, DIM) is fine.
    # If node features can be 0, 1, 2, then NODE_FEAT_INPUT_DIM should be 3.
    # For simplicity, we'll stick to the hardcoded NODE_FEAT_INPUT_DIM = 1 for the x = torch.zeros case.
    # If your 'x' values are different, you'll need to adjust NODE_FEAT_INPUT_DIM.
    # A more robust way:
    # max_node_val = 0
    # if train_graphs and train_graphs[0].x.numel() > 0:
    #     max_node_val = int(train_graphs[0].x.max().item())
    # NODE_FEAT_INPUT_DIM_FOR_EMBEDDING = max_node_val + 1
    # print(f"Auto-detected NODE_FEAT_INPUT_DIM_FOR_EMBEDDING: {NODE_FEAT_INPUT_DIM_FOR_EMBEDDING}")
    # Then pass this to the model. For now, keeping it simpler.
    NODE_FEAT_INPUT_DIM_FOR_EMBEDDING = NODE_FEAT_INPUT_DIM # Simplification: assuming it's #categories

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if val_graphs else None

    model = MyLocalGatedGCN(
        current_use_rwse_pe=USE_RWSE_PE, 
        current_pe_dim=PE_DIM,
        node_feat_dim=NODE_FEAT_INPUT_DIM_FOR_EMBEDDING, # Number of node categories
        edge_feat_dim=EDGE_FEAT_INPUT_DIM
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1) # Ignores y = -1 for test set
    
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
    
    model_save_dir = 'models'
    model_save_path = os.path.join(model_save_dir, 'best_gatedgcn_multids.pth') # Unique name

    for epoch_iter in range(1, EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        val_loss, val_acc = (0,0)
        if val_loader and val_graphs: # Check if val_graphs is not empty
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        if val_loader and val_graphs and val_acc > best_val_acc :
            best_val_acc = val_acc
            os.makedirs(model_save_dir, exist_ok=True) 
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
    
    # --- Generate predictions for each test dataset ---
    print("\n--- Generating Test Predictions ---")
    for ds_name, current_test_graphs in test_datasets.items():
        if current_test_graphs:
            print(f"Generating predictions for testset_{ds_name}...")
            current_test_loader = DataLoader(current_test_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            
            test_predictions_array = eval_epoch(model, current_test_loader, criterion, device, is_test_set_preds_only=True)
            
            num_test_samples = len(test_predictions_array)
            if num_test_samples > 0:
                ids = np.arange(1, num_test_samples + 1) # 1-based IDs
                predictions_df = pd.DataFrame({'id': ids, 'pred': test_predictions_array})
                
                output_predictions_filename = f'testset_{ds_name}.csv'
                predictions_df.to_csv(output_predictions_filename, index=False)
                print(f"  Test predictions for {ds_name} saved to {output_predictions_filename}")
            else:
                print(f"  No test predictions generated for {ds_name} (dataset might be empty or predictions array is empty).")
        else:
            print(f"No test data found for dataset {ds_name}. Skipping predictions.")
    print("---------------------------------")