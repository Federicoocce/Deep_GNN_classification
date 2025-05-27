# train_local_gatedgcn.py (Modified to use GATv2Model)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, Linear, GATv2Conv # Added GATv2Conv
import torch_geometric.utils
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.transforms import BaseTransform
from sklearn.metrics import accuracy_score
import time
import numpy as np
import pandas as pd
import argparse
import os

# Import from your new data loader file
from data_loader import get_data_splits, RWSE_MAX_K
from losses import GCELoss, SCELoss, BootstrappingLoss, FocalLoss, MAELoss

# --- General Hyperparameters ---
NUM_CLASSES = 6
GNN_LAYERS = 2          # Number of GNN layers (now for GAT)
GNN_HIDDEN_DIM = 512    # Output dimension of GAT layers (after head concat)
NODE_EMBEDDING_DIM = 256
EDGE_EMBEDDING_DIM = 256 # Dimension of encoded edge features for GAT
NODE_CATEGORY_COUNT = 1
EDGE_FEATURE_DIM = 7    # Raw edge feature dimension
USE_RWSE_PE = False
PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0
USE_BATCHNORM = True
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1.0e-5
EPOCHS = 300
BATCH_SIZE = 32
NUM_WARMUP_EPOCHS = 10

DEFAULT_LABEL_SMOOTHING_FACTOR = 0.2
DEFAULT_EDGE_DROP_PROBABILITY = 0.2

# --- GAT Specific Hyperparameters (can be overridden by CLI) ---
GAT_HEADS = 4           # Number of attention heads for GAT layers
GAT_DROPOUT = 0.5       # Dropout rate for GAT layers (attention and post-activation)


# --- Edge Dropping Transform ---
class DropEdges(BaseTransform):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, data):
        if self.p == 0:
            return data
        if not hasattr(data, 'edge_index') or data.edge_index is None or data.edge_index.size(1) == 0:
            return data
        new_data = data.clone()
        num_edges = data.edge_index.size(1)
        mask = torch.rand(num_edges, device=data.edge_index.device) > self.p
        new_data.edge_index = data.edge_index[:, mask]
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            new_data.edge_attr = data.edge_attr[mask]
        return new_data

# --- Custom Dataset Wrapper ---
class ListDataset(Dataset):
    def __init__(self, data_list, transform=None):
        super().__init__(transform=transform)
        self.data_list = [g for g in data_list if g is not None]
        if len(self.data_list) != len(data_list) and len(data_list) > 0 :
            print(f"Warning: Filtered out {len(data_list) - len(self.data_list)} None graphs in ListDataset.")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# --- Model Definition (MyRobustGATModel) ---
class MyRobustGATModel(torch.nn.Module):
    def __init__(self, current_use_rwse_pe, current_pe_dim,
                 node_emb_dim, edge_emb_dim,
                 gnn_hidden_dim, num_gnn_layers, num_gat_heads,
                 gat_dropout_rate, num_classes, edge_feat_dim,
                 node_category_count, use_batchnorm=True):
        super().__init__()
        self.use_rwse_pe = current_use_rwse_pe
        self.node_encoder = nn.Embedding(num_embeddings=node_category_count, embedding_dim=node_emb_dim)

        # Edge encoder: projects raw edge features to edge_emb_dim for GAT
        # self.edge_encoder is an instance of torch_geometric.nn.Linear
        self.edge_encoder = Linear(edge_feat_dim, edge_emb_dim) if edge_feat_dim > 0 and edge_emb_dim > 0 else None

        current_node_input_dim = node_emb_dim
        if self.use_rwse_pe:
            current_node_input_dim += current_pe_dim

        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batchnorm else None
        self.activations = nn.ModuleList()

        if gnn_hidden_dim % num_gat_heads != 0:
            raise ValueError(f"GNN hidden dimension ({gnn_hidden_dim}) must be divisible by the number of GAT heads ({num_gat_heads}).")
        out_channels_per_head = gnn_hidden_dim // num_gat_heads

        for i in range(num_gnn_layers):
            in_channels = current_node_input_dim if i == 0 else gnn_hidden_dim
            self.gat_layers.append(
                GATv2Conv(in_channels=in_channels,
                          out_channels=out_channels_per_head, # This is per head
                          heads=num_gat_heads,
                          concat=True, # Output dim is heads * out_channels_per_head = gnn_hidden_dim
                          dropout=gat_dropout_rate, # Dropout for attention scores
                          edge_dim=edge_emb_dim if self.edge_encoder else None,
                          add_self_loops=True)
            )
            self.activations.append(nn.ELU()) # ELU or ReLU
            if use_batchnorm:
                self.batch_norms.append(nn.BatchNorm1d(gnn_hidden_dim))

        self.pool = global_mean_pool
        self.head = Linear(gnn_hidden_dim, num_classes)
        self.gat_dropout_rate = gat_dropout_rate # For dropout after activation

    def forward(self, data):
        x, edge_idx, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if x.dtype == torch.long:
            x_base = self.node_encoder(x.squeeze(-1))
        else:
            x_base = self.node_encoder(x.long().squeeze(-1))

        encoded_edge_attr = None
        if self.edge_encoder and hasattr(edge_attr, 'numel') and edge_attr.numel() > 0:
            # Ensure edge_attr has the correct dimension before passing to encoder
            # CORRECTED LINE:
            if edge_attr.shape[1] == self.edge_encoder.in_channels:
                 encoded_edge_attr = self.edge_encoder(edge_attr)
            else:
                # Fallback: if dimensions don't match but edges exist, create zero tensor
                if edge_idx.numel() > 0:
                    # print(f"Warning: Edge feature dim mismatch. Expected {self.edge_encoder.in_channels}, got {edge_attr.shape[1]}. Using zeros for GAT edge_dim if applicable.")
                    encoded_edge_attr = torch.zeros((edge_attr.shape[0], self.edge_encoder.out_channels), device=x.device, dtype=x_base.dtype)


        current_x = x_base
        if self.use_rwse_pe and hasattr(data, 'rwse_pe') and data.rwse_pe is not None and data.rwse_pe.numel() > 0:
            pe = data.rwse_pe.float().to(x_base.device)
            if x_base.size(0) == pe.size(0):
                current_x = torch.cat([x_base, pe], dim=-1)

        for i, layer in enumerate(self.gat_layers):
            current_x = layer(current_x, edge_idx, edge_attr=encoded_edge_attr)
            if self.batch_norms and current_x.numel() > 0: # Check for empty tensor
                current_x = self.batch_norms[i](current_x)
            current_x = self.activations[i](current_x)
            current_x = F.dropout(current_x, p=self.gat_dropout_rate, training=self.training) # Dropout after activation

        graph_x = self.pool(current_x, batch)
        out = self.head(graph_x)
        return out

# --- Training and Evaluation Functions (largely unchanged) ---
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, processed_graphs_count = 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        target_y = data.y.squeeze()
        if target_y.ndim == 0: target_y = target_y.unsqueeze(0) # Ensure target_y is at least 1D
        valid_targets_mask = target_y != -1
        if not valid_targets_mask.any(): continue
        loss = criterion(out[valid_targets_mask], target_y[valid_targets_mask])
        loss.backward()
        optimizer.step()
        num_valid_in_batch = torch.sum(valid_targets_mask).item()
        total_loss += loss.item() * num_valid_in_batch
        processed_graphs_count += num_valid_in_batch
    return total_loss / processed_graphs_count if processed_graphs_count > 0 else 0

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, is_test_set_preds_only=False):
    model.eval()
    total_loss, processed_graphs_count = 0, 0
    all_preds_list, all_labels_list = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        preds = out.argmax(dim=1)
        all_preds_list.append(preds.cpu())
        if not is_test_set_preds_only:
            target_y = data.y.squeeze()
            if target_y.ndim == 0: target_y = target_y.unsqueeze(0) # Ensure target_y is at least 1D
            valid_targets_mask = target_y != -1
            num_valid_in_batch = torch.sum(valid_targets_mask).item()
            if num_valid_in_batch > 0:
                loss = criterion(out[valid_targets_mask], target_y[valid_targets_mask]) # Pass only valid targets to loss
                total_loss += loss.item() * num_valid_in_batch
            all_labels_list.append(target_y.cpu()) # Store all labels (including -1 for later full alignment if needed)
            processed_graphs_count += num_valid_in_batch

    if is_test_set_preds_only:
        return torch.cat(all_preds_list).numpy() if all_preds_list else np.array([])

    if not all_labels_list: return 0, 0
    all_preds_np = torch.cat(all_preds_list).numpy()
    all_labels_np = torch.cat(all_labels_list).numpy()

    # Filter out -1 labels for accuracy calculation
    valid_indices_overall = all_labels_np != -1
    accuracy = 0
    num_valid_samples_overall = np.sum(valid_indices_overall)

    if num_valid_samples_overall > 0:
        accuracy = accuracy_score(all_labels_np[valid_indices_overall], all_preds_np[valid_indices_overall])

    effective_loss = total_loss / num_valid_samples_overall if num_valid_samples_overall > 0 else 0
    return effective_loss, accuracy


# --- Loss Function Helper (unchanged) ---
def _get_loss(loss_name: str, label_smoothing_factor: float = 0.0, num_classes: int = NUM_CLASSES):
    loss_name = loss_name.lower()
    if loss_name == 'ce':
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing_factor if label_smoothing_factor > 0 else 0.0,
                                   ignore_index=-1)
    elif loss_name == 'gce': return GCELoss(q=0.7) # Note: Original GCELoss needs update for ignore_index or ensure masking
    elif loss_name == 'sce': return SCELoss(alpha=1.0, beta=1.0) # Note: Original SCELoss needs update for ignore_index
    elif loss_name == 'bootstrapping': return BootstrappingLoss(beta=0.95)
    elif loss_name == 'focal': return FocalLoss(gamma=2.0) # Note: Original FocalLoss needs update for ignore_index
    elif loss_name == 'mae': return MAELoss()
    else: raise ValueError(f"Unknown loss function: {loss_name}")
    # IMPORTANT: For custom losses (GCE, SCE, Bootstrapping, Focal, MAE), ensure they handle
    # ignore_index=-1 or that the masking (target_y != -1) is correctly applied *before*
    # passing targets to these loss functions if they don't support ignore_index.
    # The current train_epoch function does this masking before calling criterion.

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GATv2 Training with A,B,C,D datasets')
    parser.add_argument('--force_reprocess_data', action='store_true', help="Force re-processing of data")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument('--no_rwse', action='store_false', dest='use_rwse_pe_cli', help="Disable RWSE Positional Encoding.")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument('--ds', type=str, default='A', choices=['A', 'B', 'C', 'D'], help="Dataset SUFFIX to use for training (default: A).")
    parser.add_argument('--loss_fn', type=str, default='ce', choices=['ce', 'gce', 'sce', 'bootstrapping', 'focal', 'mae'], help="Loss function to use (default: ce).")
    parser.add_argument('--label_smoothing', type=float, default=DEFAULT_LABEL_SMOOTHING_FACTOR, help="Label smoothing factor (0.0 for no smoothing). Only for 'ce' loss.")
    parser.add_argument('--edge_drop_prob', type=float, default=DEFAULT_EDGE_DROP_PROBABILITY, help="Probability of dropping an edge during training (0.0 for no dropping).")
    parser.add_argument('--predict_on_test', action='store_true', help="Generate predictions on the test set of the trained dataset suffix.")

    # GAT specific arguments
    parser.add_argument('--gat_heads', type=int, default=GAT_HEADS, help="Number of GAT attention heads.")
    parser.add_argument('--gat_dropout', type=float, default=GAT_DROPOUT, help="Dropout for GAT layers.")
    parser.add_argument('--gnn_hidden_dim', type=int, default=GNN_HIDDEN_DIM, help="Hidden dimension for GNN layers.")
    parser.add_argument('--gnn_layers', type=int, default=GNN_LAYERS, help="Number of GNN layers.")
    parser.add_argument('--node_emb_dim', type=int, default=NODE_EMBEDDING_DIM, help="Node embedding dimension.")
    parser.add_argument('--edge_emb_dim', type=int, default=EDGE_EMBEDDING_DIM, help="Edge embedding dimension (for GAT edge_dim).")


    parser.set_defaults(use_rwse_pe_cli=USE_RWSE_PE) # Default for USE_RWSE_PE
    cli_args = parser.parse_args()

    # Update hyperparameters from CLI
    LOSS_NAME = cli_args.loss_fn
    EPOCHS = cli_args.epochs
    LEARNING_RATE = cli_args.lr
    USE_RWSE_PE = cli_args.use_rwse_pe_cli
    DATASET_SUFFIX_TO_TRAIN_ON = cli_args.ds
    LABEL_SMOOTHING_FACTOR = cli_args.label_smoothing
    EDGE_DROP_PROBABILITY = cli_args.edge_drop_prob
    PREDICT_ON_TEST_SET = cli_args.predict_on_test

    GAT_HEADS = cli_args.gat_heads
    GAT_DROPOUT = cli_args.gat_dropout
    GNN_HIDDEN_DIM = cli_args.gnn_hidden_dim
    GNN_LAYERS = cli_args.gnn_layers
    NODE_EMBEDDING_DIM = cli_args.node_emb_dim
    EDGE_EMBEDDING_DIM = cli_args.edge_emb_dim

    PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0


    print(f"--- Configuration ---")
    print(f"Model Type: GATv2")
    print(f"Training on Dataset: {DATASET_SUFFIX_TO_TRAIN_ON}")
    print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}")
    print(f"GNN: Layers={GNN_LAYERS}, HiddenDim={GNN_HIDDEN_DIM}")
    print(f"GAT: Heads={GAT_HEADS}, Dropout={GAT_DROPOUT}")
    print(f"Embeddings: NodeEmb={NODE_EMBEDDING_DIM}, EdgeEmb={EDGE_EMBEDDING_DIM} (for GAT edge_attr)")
    print(f"NodeCatCount: {NODE_CATEGORY_COUNT}, RawEdgeFeatDim: {EDGE_FEATURE_DIM}")
    print(f"BN: {USE_BATCHNORM}, RWSE: {USE_RWSE_PE}, PE_Dim: {PE_DIM}")
    criterion = _get_loss(LOSS_NAME, label_smoothing_factor=LABEL_SMOOTHING_FACTOR if LOSS_NAME == 'ce' else 0.0)
    print(f"Loss Func: {LOSS_NAME}" + (f" (LS: {LABEL_SMOOTHING_FACTOR})" if LOSS_NAME == 'ce' and LABEL_SMOOTHING_FACTOR > 0 else ""))
    print(f"EdgeDrop (Train): {EDGE_DROP_PROBABILITY}")
    print(f"Predict on Test Set for DS {DATASET_SUFFIX_TO_TRAIN_ON}: {PREDICT_ON_TEST_SET}")
    print(f"--------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data splits from data_loader...")
    all_dataset_splits = get_data_splits(force_reprocess=cli_args.force_reprocess_data)

    if DATASET_SUFFIX_TO_TRAIN_ON not in all_dataset_splits:
        print(f"ERROR: Dataset suffix {DATASET_SUFFIX_TO_TRAIN_ON} not found in loaded splits. Available: {list(all_dataset_splits.keys())}")
        exit()

    current_ds_splits = all_dataset_splits[DATASET_SUFFIX_TO_TRAIN_ON]
    train_graphs = current_ds_splits.get('train', [])
    val_graphs = current_ds_splits.get('val', [])

    if not train_graphs:
        print(f"No training data found for dataset {DATASET_SUFFIX_TO_TRAIN_ON}. Exiting.")
        exit()

    # Check edge feature dimension consistency with EDGE_FEATURE_DIM
    if train_graphs and hasattr(train_graphs[0], 'edge_attr') and train_graphs[0].edge_attr is not None and train_graphs[0].edge_attr.numel() > 0:
        actual_edge_dim = train_graphs[0].edge_attr.shape[1]
        if actual_edge_dim != EDGE_FEATURE_DIM:
            print(f"WARNING: Loaded training data (DS: {DATASET_SUFFIX_TO_TRAIN_ON}) has raw edge_attr_dim {actual_edge_dim}, model expects {EDGE_FEATURE_DIM} for its edge_encoder input.")
    elif EDGE_FEATURE_DIM > 0 and train_graphs: # If model expects edge features but data doesn't have them
        print(f"WARNING: Model expects raw edge_attr_dim {EDGE_FEATURE_DIM}, but first graph in training data (DS: {DATASET_SUFFIX_TO_TRAIN_ON}) has no/empty edge_attr.")


    train_transform = None
    if EDGE_DROP_PROBABILITY > 0:
        train_transform = DropEdges(p=EDGE_DROP_PROBABILITY)
        print(f"Applying EdgeDropping (p={EDGE_DROP_PROBABILITY}) to training data of Dataset {DATASET_SUFFIX_TO_TRAIN_ON}.")

    train_dataset = ListDataset(train_graphs, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=len(train_dataset)>BATCH_SIZE) # drop_last if more than one batch

    val_dataset = ListDataset(val_graphs)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if val_graphs else None

    # Instantiate the GATv2 model
    try:
        model = MyRobustGATModel(
            current_use_rwse_pe=USE_RWSE_PE,
            current_pe_dim=PE_DIM,
            node_emb_dim=NODE_EMBEDDING_DIM,
            edge_emb_dim=EDGE_EMBEDDING_DIM,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            num_gnn_layers=GNN_LAYERS,
            num_gat_heads=GAT_HEADS,
            gat_dropout_rate=GAT_DROPOUT,
            num_classes=NUM_CLASSES,
            edge_feat_dim=EDGE_FEATURE_DIM,
            node_category_count=NODE_CATEGORY_COUNT,
            use_batchnorm=USE_BATCHNORM
        ).to(device)
    except ValueError as e:
        print(f"Error initializing model: {e}")
        exit()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if EPOCHS > NUM_WARMUP_EPOCHS:
        def lr_lambda_fn(epoch):
            if epoch < NUM_WARMUP_EPOCHS: return float(epoch + 1) / float(max(1,NUM_WARMUP_EPOCHS)) # Avoid division by zero if NUM_WARMUP_EPOCHS is 0
            progress = float(epoch - NUM_WARMUP_EPOCHS) / float(max(1, EPOCHS - NUM_WARMUP_EPOCHS))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_fn)

    print(f"\nStarting training for Dataset {DATASET_SUFFIX_TO_TRAIN_ON} with GATv2 model...")
    best_val_acc = 0.0
    model_save_dir = 'models'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'best_gatv2_trained_on_DS_{DATASET_SUFFIX_TO_TRAIN_ON}.pth')

    for epoch_iter in range(1, EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = (0, 0) # Default if no val_loader
        if val_loader and len(val_loader.dataset) > 0: # Check if val_loader has data
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        if val_loader and len(val_loader.dataset) > 0 and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"*** DS {DATASET_SUFFIX_TO_TRAIN_ON} - Best val_acc: {best_val_acc:.4f} (Epoch {epoch_iter}). Model saved to {model_save_path} ***")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"DS {DATASET_SUFFIX_TO_TRAIN_ON} - Epoch {epoch_iter:03d}/{EPOCHS:03d} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.4f} | LR: {current_lr:.1e} | Time: {(time.time()-start_time):.2f}s")
        if scheduler: scheduler.step()

    print(f"\nTraining for Dataset {DATASET_SUFFIX_TO_TRAIN_ON} finished.")

    if PREDICT_ON_TEST_SET:
        if os.path.exists(model_save_path):
            print(f"Loading best model (trained on DS {DATASET_SUFFIX_TO_TRAIN_ON}) from {model_save_path} for test predictions...")
            model.load_state_dict(torch.load(model_save_path, map_location=device))
        else:
            print(f"Warning: No best model saved for DS {DATASET_SUFFIX_TO_TRAIN_ON} at {model_save_path}. Using model from last epoch for testing.")

        print(f"\n--- Generating Test Predictions for Test Set of DS {DATASET_SUFFIX_TO_TRAIN_ON} ONLY ---")
        test_ds_suffix_to_predict = DATASET_SUFFIX_TO_TRAIN_ON

        if test_ds_suffix_to_predict in all_dataset_splits and \
           all_dataset_splits[test_ds_suffix_to_predict].get('test'):

            current_test_graphs = all_dataset_splits[test_ds_suffix_to_predict]['test']
            if not current_test_graphs:
                print(f"  No test graphs found for dataset {test_ds_suffix_to_predict}. Skipping predictions.")
            else:
                print(f"Generating predictions for testset_{test_ds_suffix_to_predict} ({len(current_test_graphs)} graphs)...")
                current_test_dataset = ListDataset(current_test_graphs)
                current_test_loader = DataLoader(current_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

                test_predictions_array = eval_epoch(model, current_test_loader, criterion, device, is_test_set_preds_only=True)
                num_test_samples = len(test_predictions_array)

                if num_test_samples > 0:
                    # Ensure 'id' column matches the number of predictions.
                    # The original data doesn't have explicit graph IDs for test, so simple range is used.
                    ids = np.arange(num_test_samples)
                    predictions_df = pd.DataFrame({'id': ids, 'pred': test_predictions_array})
                    output_filename = f'testset_{test_ds_suffix_to_predict}.csv'
                    predictions_df.to_csv(output_filename, index=False)
                    print(f"  Test predictions for {test_ds_suffix_to_predict} saved to {output_filename}")
                elif len(current_test_graphs) > 0 : # If there were graphs but no predictions (e.g. all filtered)
                     print(f"  No test predictions generated for {test_ds_suffix_to_predict} despite having test graphs. Check model output or filtering.")
                else: # Should be caught by "No test graphs found"
                    print(f"  No test predictions generated for {test_ds_suffix_to_predict} (dataset might be empty).")
        else:
            print(f"No test data found for dataset {test_ds_suffix_to_predict}. Skipping predictions.")
        print("---------------------------------")
    else:
        print("\nSkipping test set predictions as --predict_on_test was not specified.")