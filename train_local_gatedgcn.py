# train_local_gatedgcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, Linear
from torch_geometric.loader import DataLoader
import torch_geometric.utils # For scatter
from sklearn.metrics import accuracy_score
import time
import numpy as np
import argparse # For command-line arguments

# --- Import your data loading function ---
from load_my_data import get_data_splits, RWSE_MAX_K # Import RWSE_MAX_K for PE dim

# --- Hyperparameters (Based on GNN+ Paper Table 13 for GatedGCN+ on OGBG-PPA) ---
# Model Structure
NUM_CLASSES = 6  # !!! IMPORTANT: SET THIS TO YOUR ACTUAL NUMBER OF CLASSES !!!
GNN_LAYERS = 4
GNN_HIDDEN_DIM = 512 # 'dim_inner' in GNN+
GNN_DROPOUT = 0.15
USE_RESIDUAL = True
USE_FFN = True      # Enable Feed-Forward Network block in GatedGCNLayer
USE_BATCHNORM = True # Enable Batch Norm within GatedGCNLayer
NODE_EMBEDDING_DIM = 256 # If using node embeddings for placeholder 'x'
EDGE_EMBEDDING_DIM = 128 # Dimension to project edge features to
USE_RWSE_PE = True # Flag to control PE usage
PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0 # Dimension of our RWSE positional encoding

# Training
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1.0e-5
EPOCHS = 50 # START WITH A SMALL NUMBER FOR TESTING (e.g., 5-10)
BATCH_SIZE = 32
NUM_WARMUP_EPOCHS = 10 # For scheduler

# --- Modified GatedGCNLayer (to remove cfg dependency and handle residual) ---
class StandaloneGatedGCNLayer(torch.nn.Module):
    def __init__(self, in_dim_node, in_dim_edge, out_dim, dropout, residual, ffn_enabled,
                 batchnorm_enabled, act_fn_constructor, aggr='add', **kwargs):
        super().__init__(**kwargs)
        self.in_dim_node = in_dim_node
        self.in_dim_edge = in_dim_edge # Store in_dim_edge
        self.out_dim = out_dim

        self.activation = act_fn_constructor()
        self.A = Linear(in_dim_node, out_dim, bias=True)
        self.B = Linear(in_dim_node, out_dim, bias=True)
        self.C = Linear(in_dim_edge, out_dim, bias=True)
        self.D = Linear(in_dim_node, out_dim, bias=True)
        self.E = Linear(in_dim_node, out_dim, bias=True)

        self.act_fn_x = self.activation
        self.act_fn_e = self.activation
        self.dropout_rate = dropout
        self.residual_enabled = residual
        self.e_prop = None

        self.batchnorm_enabled = batchnorm_enabled
        self.ffn_enabled = ffn_enabled
        self.aggr = aggr

        if self.batchnorm_enabled:
            self.bn_node_x = nn.BatchNorm1d(out_dim)
            self.bn_edge_e = nn.BatchNorm1d(out_dim)

        if self.residual_enabled:
            if self.in_dim_node != self.out_dim:
                self.residual_proj_node = Linear(self.in_dim_node, self.out_dim, bias=False)
            else:
                self.residual_proj_node = nn.Identity()
            
            if self.in_dim_edge != self.out_dim: # Project edge identity if dims change
                self.residual_proj_edge = Linear(self.in_dim_edge, self.out_dim, bias=False)
            else:
                self.residual_proj_edge = nn.Identity()
        else: # Ensure these attributes exist even if residual is False
            self.residual_proj_node = nn.Identity()
            self.residual_proj_edge = nn.Identity()


        if self.ffn_enabled:
            if self.batchnorm_enabled:
                self.norm1_ffn = nn.BatchNorm1d(out_dim)
            self.ff_linear1 = Linear(out_dim, out_dim * 2)
            self.ff_linear2 = Linear(out_dim * 2, out_dim)
            self.act_fn_ff = act_fn_constructor()
            if self.batchnorm_enabled:
                self.norm2_ffn = nn.BatchNorm1d(out_dim)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, x_input_node, edge_index, edge_input_attr):
        x_identity = x_input_node
        e_identity = edge_input_attr

        Ax = self.A(x_input_node)
        Bx = self.B(x_input_node)
        Ce = self.C(edge_input_attr)
        Dx = self.D(x_input_node)
        Ex = self.E(x_input_node)

        row, col = edge_index
        # Check if graph has edges to prevent indexing errors
        if edge_index.numel() > 0:
            dx_i = Dx[row]
            ex_j = Ex[col]
            e_ij = dx_i + ex_j + Ce
            sigma_ij = torch.sigmoid(e_ij)
            self.e_prop = e_ij
            msg_j = sigma_ij * Bx[col]
            aggr_out = torch_geometric.utils.scatter(msg_j, row, dim=0, dim_size=x_input_node.size(0), reduce=self.aggr)
            x_transformed = Ax + aggr_out
            edge_attr_transformed = self.e_prop
        else: # No edges, so no message passing
            x_transformed = Ax # Apply A for consistency in output dim, or just x_input_node if A isn't a simple projection
                               # If A projects, then Ax is right. If first layer and residual, this needs care.
                               # For simplicity, let's assume A is a projection, so Ax is okay.
                               # Or, more safely for no-edge case: x_transformed = self.A(x_input_node) if input and output dims are different
                               # and x_input_node otherwise.
                               # Let's assume A always transforms to out_dim.
            edge_attr_transformed = torch.zeros((0, self.out_dim), device=x_input_node.device) # No edges, so empty edge attrs of out_dim

        if self.batchnorm_enabled:
            x_transformed = self.bn_node_x(x_transformed)
            if edge_attr_transformed.numel() > 0: # Only apply BN if there are edge features
                edge_attr_transformed = self.bn_edge_e(edge_attr_transformed)


        x_transformed = self.act_fn_x(x_transformed)
        if edge_attr_transformed.numel() > 0:
            edge_attr_transformed = self.act_fn_e(edge_attr_transformed)

        x_transformed = F.dropout(x_transformed, self.dropout_rate, training=self.training)
        if edge_attr_transformed.numel() > 0:
            edge_attr_transformed = F.dropout(edge_attr_transformed, self.dropout_rate, training=self.training)

        if self.residual_enabled:
            x_final = self.residual_proj_node(x_identity) + x_transformed
            # For edge_attr_final, ensure e_identity is projected if needed and edge_attr_transformed is not empty
            if edge_attr_transformed.numel() > 0:
                 edge_attr_final = self.residual_proj_edge(e_identity) + edge_attr_transformed
            else: # If no edges, residual on edge features doesn't make sense / output should be empty
                 edge_attr_final = edge_attr_transformed # Which is already empty
        else:
            x_final = x_transformed
            edge_attr_final = edge_attr_transformed

        if self.ffn_enabled:
            x_ffn_identity = x_final
            if self.batchnorm_enabled:
                x_ffn_processed = self.norm1_ffn(x_ffn_identity)
            else:
                x_ffn_processed = x_ffn_identity
            
            x_ffn_processed = self._ff_block(x_ffn_processed)
            x_final = x_ffn_identity + x_ffn_processed
            
            if self.batchnorm_enabled:
                x_final = self.norm2_ffn(x_final)
        
        return x_final, edge_attr_final


# --- Model Definition (Updated for RWSE) ---
class MyLocalGatedGCN(torch.nn.Module):
    def __init__(self, node_input_feat_dim_ignored, edge_input_feat_dim, num_classes,
                 gnn_hidden_dim, gnn_layers, gnn_dropout,
                 use_residual, use_ffn, use_batchnorm,
                 node_embedding_dim, edge_embedding_dim,
                 use_rwse_pe, pe_dim):
        super().__init__()
        self.use_rwse_pe = use_rwse_pe

        self.node_encoder = nn.Embedding(2, node_embedding_dim) # For placeholder x (vocab size 2: 0 and 1)
        self.edge_encoder = Linear(edge_input_feat_dim, edge_embedding_dim)

        first_gnn_node_dim_input = node_embedding_dim
        if self.use_rwse_pe:
            first_gnn_node_dim_input += pe_dim

        self.gnn_layers_modulelist = nn.ModuleList() # Renamed to avoid conflict
        
        # First layer
        self.gnn_layers_modulelist.append(
            StandaloneGatedGCNLayer(
                in_dim_node=first_gnn_node_dim_input,
                in_dim_edge=edge_embedding_dim,
                out_dim=gnn_hidden_dim,
                dropout=gnn_dropout,
                residual=use_residual,
                ffn_enabled=use_ffn,
                batchnorm_enabled=use_batchnorm,
                act_fn_constructor=lambda: nn.ReLU()
            )
        )
        
        # Subsequent layers
        for _ in range(1, gnn_layers):
            self.gnn_layers_modulelist.append(
                StandaloneGatedGCNLayer(
                    in_dim_node=gnn_hidden_dim,
                    in_dim_edge=gnn_hidden_dim, # Edge features are transformed to hidden_dim
                    out_dim=gnn_hidden_dim,
                    dropout=gnn_dropout,
                    residual=use_residual,
                    ffn_enabled=use_ffn,
                    batchnorm_enabled=use_batchnorm,
                    act_fn_constructor=lambda: nn.ReLU()
                )
            )

        self.pool = global_mean_pool
        self.head = Linear(gnn_hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x_base = self.node_encoder(x.squeeze(-1)) # x should be [N] or [N,1] long tensor
        edge_attr_encoded = self.edge_encoder(edge_attr)

        if self.use_rwse_pe and hasattr(data, 'rwse_pe') and data.rwse_pe is not None and data.rwse_pe.numel() > 0:
            pe = data.rwse_pe.float().to(x_base.device) # Ensure PE is float and on correct device
            if x_base.size(0) == pe.size(0):
                x_processed = torch.cat([x_base, pe], dim=-1)
            else:
                # This case should ideally not happen if data loading is correct
                print(f"Warning: Mismatch in node count for PE. x_base: {x_base.shape}, pe: {pe.shape}. Skipping PE for this batch/graph.")
                x_processed = x_base
        else:
            x_processed = x_base
        
        current_x = x_processed
        current_edge_attr = edge_attr_encoded

        for gnn_layer_item in self.gnn_layers_modulelist: # Iterate using the renamed attribute
            current_x, current_edge_attr = gnn_layer_item(current_x, edge_index, current_edge_attr)

        graph_x = self.pool(current_x, batch)
        out = self.head(graph_x)
        return out

# --- Training and Evaluation Functions ---
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    processed_graphs = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # Ensure data.y is 1D for CrossEntropyLoss
        target_y = data.y.squeeze() # Squeeze if y is [B,1] to become [B]
        if target_y.ndim == 0: # If batch size is 1, squeeze might make it scalar
            target_y = target_y.unsqueeze(0)

        loss = criterion(out, target_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        processed_graphs += data.num_graphs
    return total_loss / processed_graphs if processed_graphs > 0 else 0


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    processed_graphs = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        
        target_y = data.y.squeeze()
        if target_y.ndim == 0:
            target_y = target_y.unsqueeze(0)

        loss = criterion(out, target_y)
        total_loss += loss.item() * data.num_graphs
        preds = out.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(target_y.cpu()) # Use squeezed target_y
        processed_graphs += data.num_graphs

    if not all_labels: # Handle empty loader case
        return 0, 0

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy()) # Use .numpy() for sklearn metrics
    return total_loss / processed_graphs if processed_graphs > 0 else 0, accuracy

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local GatedGCN Training')
    parser.add_argument('--force_reprocess_data', action='store_true',
                        help="Force re-processing of data from JSONs even if .pt files exist.")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument('--hidden_dim', type=int, default=GNN_HIDDEN_DIM, help="GNN hidden dimension.")
    parser.add_argument('--num_layers', type=int, default=GNN_LAYERS, help="Number of GNN layers.")
    parser.add_argument('--dropout', type=float, default=GNN_DROPOUT, help="Dropout rate.")
    parser.add_argument('--no_rwse', action='store_true', help="Disable RWSE Positional Encoding.")


    cli_args = parser.parse_args()

    # Update hyperparameters from CLI args
    EPOCHS = cli_args.epochs
    LEARNING_RATE = cli_args.lr
    BATCH_SIZE = cli_args.batch_size
    GNN_HIDDEN_DIM = cli_args.hidden_dim
    GNN_LAYERS = cli_args.num_layers
    GNN_DROPOUT = cli_args.dropout
    USE_RWSE_PE = not cli_args.no_rwse
    PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0


    print(f"Starting local GatedGCN training for {NUM_CLASSES}-class subset...")
    print(f"Using RWSE: {USE_RWSE_PE}, PE_DIM: {PE_DIM}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_graphs, val_graphs, test_graphs = get_data_splits(force_reprocess=cli_args.force_reprocess_data)

    if not train_graphs:
        print("No training data loaded. Exiting.")
        exit()
    
    print(f"Sample y from first training graph: {train_graphs[0].y}")
    if USE_RWSE_PE and hasattr(train_graphs[0], 'rwse_pe') and train_graphs[0].rwse_pe is not None:
         print(f"Sample RWSE PE shape from first training graph: {train_graphs[0].rwse_pe.shape}")
    elif USE_RWSE_PE:
        print("Warning: RWSE PE enabled but not found on sample graph. Ensure it's calculated in load_my_data.py.")


    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # node_input_feat_dim_ignored is not actually used as input to Embedding layer
    # Embedding layer uses vocab size (2 in this case for 0/1 placeholder x)
    edge_input_feat_dim = train_graphs[0].edge_attr.size(1) if train_graphs[0].edge_attr.numel() > 0 else 0


    model = MyLocalGatedGCN(
        node_input_feat_dim_ignored=1, # Not directly used by nn.Embedding like this
        edge_input_feat_dim=edge_input_feat_dim,
        num_classes=NUM_CLASSES,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        gnn_layers=GNN_LAYERS,
        gnn_dropout=GNN_DROPOUT,
        use_residual=USE_RESIDUAL,
        use_ffn=USE_FFN,
        use_batchnorm=USE_BATCHNORM,
        node_embedding_dim=NODE_EMBEDDING_DIM,
        edge_embedding_dim=EDGE_EMBEDDING_DIM,
        use_rwse_pe=USE_RWSE_PE,
        pe_dim=PE_DIM
    ).to(device)

    print("\nModel Architecture:")
    # print(model) # Can be very verbose
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = None
    if EPOCHS > NUM_WARMUP_EPOCHS :
        def lr_lambda(current_epoch_internal): # Renamed to avoid conflict
            if current_epoch_internal < NUM_WARMUP_EPOCHS:
                return float(current_epoch_internal + 1) / float(NUM_WARMUP_EPOCHS + 1)
            else:
                progress = float(current_epoch_internal - NUM_WARMUP_EPOCHS) / float(max(1, EPOCHS - NUM_WARMUP_EPOCHS))
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("\nStarting training...")
    best_val_acc = 0.0
    for epoch_iter in range(1, EPOCHS + 1): # Renamed to avoid conflict
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # It's good practice to evaluate validation set less frequently for large datasets/long epochs
        # For now, evaluating every epoch.
        if val_loader.dataset: # Ensure val_loader is not empty
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = 0, 0
            print("Warning: Validation set is empty. Skipping validation.")

        
        if val_acc > best_val_acc and val_loader.dataset:
            best_val_acc = val_acc
            # torch.save(model.state_dict(), 'best_gatedgcn_local_ppa.pth')
            print(f"*** New best validation accuracy: {best_val_acc:.4f} (Epoch {epoch_iter}) ***")

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch_iter:03d}/{EPOCHS:03d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.1e} | Time: {epoch_time:.2f}s")

        if scheduler:
            scheduler.step()

    print("\nEvaluating on Test Set (using model from last epoch)...")
    if test_loader.dataset: # Ensure test_loader is not empty
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    else:
        print("Warning: Test set is empty. Skipping test evaluation.")
        
    print("\nTraining finished.")