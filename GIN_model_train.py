# train_local_gin.py (Modified to use GINModel)
import torch
import torch.nn as nn
import torch.nn.functional as F
# Make sure to import GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, Linear, GINConv
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
GNN_LAYERS = 3          # Number of GIN layers
GNN_HIDDEN_DIM = 256    # Hidden dimension for GIN layers and MLPs within GIN
NODE_EMBEDDING_DIM = 128
# EDGE_EMBEDDING_DIM is not directly used by standard GINConv layers
# EDGE_EMBEDDING_DIM = 128
NODE_CATEGORY_COUNT = 1
EDGE_FEATURE_DIM = 7    # Raw edge feature dimension (available in data, but not used by GINConv)
USE_RWSE_PE = False
PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0
USE_BATCHNORM = True
LEARNING_RATE = 0.001 # GIN often trains well with slightly higher LRs initially
WEIGHT_DECAY = 1.0e-5 # Can be 0 or 1e-5 for GIN
EPOCHS = 300
BATCH_SIZE = 32
NUM_WARMUP_EPOCHS = 10 # Or set to 0 if not using LR decay initially

DEFAULT_LABEL_SMOOTHING_FACTOR = 0.1 # Can be helpful
DEFAULT_EDGE_DROP_PROBABILITY = 0.1 # GIN can be sensitive to edge noise

# --- GIN Specific Hyperparameters (can be overridden by CLI) ---
GIN_MLP_HIDDEN_DIM = 256 # Hidden dimension for the MLPs inside GINConv layers
GIN_EPS = 0.0            # Epsilon for GINConv (0 means sum aggregator, can be learnable)
GIN_TRAIN_EPS = False    # Whether epsilon is learnable
GIN_AGGREGATOR = 'sum'   # 'sum', 'mean', 'max' for global pooling


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
        # Edge attributes are kept if they exist, even if not used by GINConv directly
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


# --- Model Definition (MyRobustGINModel) ---
class MyRobustGINModel(torch.nn.Module):
    def __init__(self, current_use_rwse_pe, current_pe_dim,
                 node_emb_dim, gnn_hidden_dim, num_gnn_layers,
                 gin_mlp_hidden_dim, num_classes, node_category_count,
                 gin_eps, gin_train_eps, use_batchnorm=True, gin_aggregator='sum'):
        super().__init__()
        self.use_rwse_pe = current_use_rwse_pe
        self.node_encoder = nn.Embedding(num_embeddings=node_category_count, embedding_dim=node_emb_dim)

        current_node_input_dim = node_emb_dim
        if self.use_rwse_pe:
            current_node_input_dim += current_pe_dim

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batchnorm else None

        for i in range(num_gnn_layers):
            in_channels = current_node_input_dim if i == 0 else gnn_hidden_dim
            # MLP for GINConv: Linear -> BN -> ReLU -> Linear
            # Output of MLP should be gnn_hidden_dim
            mlp = nn.Sequential(
                Linear(in_channels, gin_mlp_hidden_dim),
                nn.BatchNorm1d(gin_mlp_hidden_dim) if use_batchnorm else nn.Identity(),
                nn.ReLU(),
                Linear(gin_mlp_hidden_dim, gnn_hidden_dim),
            )
            self.convs.append(GINConv(nn=mlp, eps=gin_eps, train_eps=gin_train_eps))
            if use_batchnorm and i < num_gnn_layers -1: # BN after GINConv (except last layer before pool)
                 self.batch_norms.append(nn.BatchNorm1d(gnn_hidden_dim))


        if gin_aggregator == 'sum':
            self.pool = global_add_pool
        elif gin_aggregator == 'mean':
            self.pool = global_mean_pool
        else:
            raise ValueError(f"Unsupported GIN aggregator: {gin_aggregator}")

        # Final MLP classifier
        self.head = nn.Sequential(
            Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.5), # Add dropout before final layer
            Linear(gnn_hidden_dim // 2, num_classes)
        )


    def forward(self, data):
        x, edge_idx, batch = data.x, data.edge_index, data.batch
        # Note: data.edge_attr is not used by GINConv directly

        if x.dtype == torch.long:
            current_x = self.node_encoder(x.squeeze(-1))
        else:
            current_x = self.node_encoder(x.long().squeeze(-1))

        if self.use_rwse_pe and hasattr(data, 'rwse_pe') and data.rwse_pe is not None and data.rwse_pe.numel() > 0:
            pe = data.rwse_pe.float().to(current_x.device)
            if current_x.size(0) == pe.size(0):
                current_x = torch.cat([current_x, pe], dim=-1)

        for i, conv_layer in enumerate(self.convs):
            current_x = conv_layer(current_x, edge_idx)
            current_x = F.relu(current_x) # Activation after GINConv
            if self.batch_norms and i < len(self.convs) - 1 : # Apply BN if not the last conv layer
                 if current_x.numel() > 0: # Check for empty tensor
                    current_x = self.batch_norms[i](current_x)
            # No dropout between GIN layers by default, but can be added

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
        if target_y.ndim == 0: target_y = target_y.unsqueeze(0)
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
            if target_y.ndim == 0: target_y = target_y.unsqueeze(0)
            valid_targets_mask = target_y != -1
            num_valid_in_batch = torch.sum(valid_targets_mask).item()
            if num_valid_in_batch > 0:
                loss = criterion(out[valid_targets_mask], target_y[valid_targets_mask])
                total_loss += loss.item() * num_valid_in_batch
            all_labels_list.append(target_y.cpu())
            processed_graphs_count += num_valid_in_batch

    if is_test_set_preds_only:
        return torch.cat(all_preds_list).numpy() if all_preds_list else np.array([])

    if not all_labels_list: return 0, 0
    all_preds_np = torch.cat(all_preds_list).numpy()
    all_labels_np = torch.cat(all_labels_list).numpy()
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
    elif loss_name == 'gce': return GCELoss(q=0.7)
    elif loss_name == 'sce': return SCELoss(alpha=1.0, beta=1.0)
    elif loss_name == 'bootstrapping': return BootstrappingLoss(beta=0.95)
    elif loss_name == 'focal': return FocalLoss(gamma=2.0)
    elif loss_name == 'mae': return MAELoss()
    else: raise ValueError(f"Unknown loss function: {loss_name}")

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GIN Training with A,B,C,D datasets')
    parser.add_argument('--force_reprocess_data', action='store_true', help="Force re-processing of data")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument('--no_rwse', action='store_false', dest='use_rwse_pe_cli', help="Disable RWSE Positional Encoding.")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument('--ds', type=str, default='A', choices=['A', 'B', 'C', 'D'], help="Dataset SUFFIX to use for training (default: A).")
    parser.add_argument('--loss_fn', type=str, default='ce', choices=['ce', 'gce', 'sce', 'bootstrapping', 'focal', 'mae'], help="Loss function to use (default: ce).")
    parser.add_argument('--label_smoothing', type=float, default=DEFAULT_LABEL_SMOOTHING_FACTOR, help="Label smoothing factor (0.0 for no smoothing). Only for 'ce' loss.")
    parser.add_argument('--edge_drop_prob', type=float, default=DEFAULT_EDGE_DROP_PROBABILITY, help="Probability of dropping an edge during training (0.0 for no dropping).")
    parser.add_argument('--predict_on_test', action='store_true', help="Generate predictions on the test set of the trained dataset suffix.")

    # GIN specific arguments
    parser.add_argument('--gnn_layers', type=int, default=GNN_LAYERS, help="Number of GIN layers.")
    parser.add_argument('--gnn_hidden_dim', type=int, default=GNN_HIDDEN_DIM, help="Hidden dimension for GIN layers (output of GINConv).")
    parser.add_argument('--node_emb_dim', type=int, default=NODE_EMBEDDING_DIM, help="Node embedding dimension.")
    parser.add_argument('--gin_mlp_hidden_dim', type=int, default=GIN_MLP_HIDDEN_DIM, help="Hidden dimension for MLPs within GINConv layers.")
    parser.add_argument('--gin_eps', type=float, default=GIN_EPS, help="Epsilon for GINConv.")
    parser.add_argument('--gin_train_eps', action='store_true', default=GIN_TRAIN_EPS, help="Make GINConv epsilon learnable.")
    parser.add_argument('--gin_aggregator', type=str, default=GIN_AGGREGATOR, choices=['sum', 'mean'], help="Global pooling aggregator for GIN.")


    parser.set_defaults(use_rwse_pe_cli=USE_RWSE_PE)
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

    GNN_LAYERS = cli_args.gnn_layers
    GNN_HIDDEN_DIM = cli_args.gnn_hidden_dim
    NODE_EMBEDDING_DIM = cli_args.node_emb_dim
    GIN_MLP_HIDDEN_DIM = cli_args.gin_mlp_hidden_dim
    GIN_EPS = cli_args.gin_eps
    GIN_TRAIN_EPS = cli_args.gin_train_eps
    GIN_AGGREGATOR = cli_args.gin_aggregator

    PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0


    print(f"--- Configuration ---")
    print(f"Model Type: GIN")
    print(f"Training on Dataset: {DATASET_SUFFIX_TO_TRAIN_ON}")
    print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}")
    print(f"GNN: Layers={GNN_LAYERS}, HiddenDim={GNN_HIDDEN_DIM}")
    print(f"GIN: MLP_Hidden={GIN_MLP_HIDDEN_DIM}, EPS={GIN_EPS}, TrainEPS={GIN_TRAIN_EPS}, Aggregator={GIN_AGGREGATOR}")
    print(f"Embeddings: NodeEmb={NODE_EMBEDDING_DIM}")
    print(f"NodeCatCount: {NODE_CATEGORY_COUNT}, RawEdgeFeatDim: {EDGE_FEATURE_DIM} (available in data, not used by GINConv)")
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

    # Note: EDGE_FEATURE_DIM check might be less critical if GINConv doesn't use edge_attr
    # but data_loader still prepares it.

    train_transform = None
    if EDGE_DROP_PROBABILITY > 0:
        train_transform = DropEdges(p=EDGE_DROP_PROBABILITY)
        print(f"Applying EdgeDropping (p={EDGE_DROP_PROBABILITY}) to training data of Dataset {DATASET_SUFFIX_TO_TRAIN_ON}.")

    train_dataset = ListDataset(train_graphs, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=len(train_dataset)>BATCH_SIZE)

    val_dataset = ListDataset(val_graphs)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if val_graphs else None

    model = MyRobustGINModel(
        current_use_rwse_pe=USE_RWSE_PE,
        current_pe_dim=PE_DIM,
        node_emb_dim=NODE_EMBEDDING_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        num_gnn_layers=GNN_LAYERS,
        gin_mlp_hidden_dim=GIN_MLP_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        node_category_count=NODE_CATEGORY_COUNT,
        gin_eps=GIN_EPS,
        gin_train_eps=GIN_TRAIN_EPS,
        use_batchnorm=USE_BATCHNORM,
        gin_aggregator=GIN_AGGREGATOR
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if EPOCHS > NUM_WARMUP_EPOCHS and NUM_WARMUP_EPOCHS > 0 : # Ensure NUM_WARMUP_EPOCHS is positive
        def lr_lambda_fn(epoch):
            if epoch < NUM_WARMUP_EPOCHS: return float(epoch + 1) / float(NUM_WARMUP_EPOCHS)
            progress = float(epoch - NUM_WARMUP_EPOCHS) / float(max(1, EPOCHS - NUM_WARMUP_EPOCHS))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_fn)
    elif EPOCHS > 0 : # Cosine decay without warmup if NUM_WARMUP_EPOCHS is 0
        def lr_lambda_fn_no_warmup(epoch):
            progress = float(epoch) / float(max(1, EPOCHS))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_fn_no_warmup)


    print(f"\nStarting training for Dataset {DATASET_SUFFIX_TO_TRAIN_ON} with GIN model...")
    best_val_acc = 0.0
    model_save_dir = 'models'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'best_gin_trained_on_DS_{DATASET_SUFFIX_TO_TRAIN_ON}.pth')

    for epoch_iter in range(1, EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = (0, 0)
        if val_loader and len(val_loader.dataset) > 0:
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
                    ids = np.arange(num_test_samples)
                    predictions_df = pd.DataFrame({'id': ids, 'pred': test_predictions_array})
                    output_filename = f'testset_{test_ds_suffix_to_predict}.csv'
                    predictions_df.to_csv(output_filename, index=False)
                    print(f"  Test predictions for {test_ds_suffix_to_predict} saved to {output_filename}")
                elif len(current_test_graphs) > 0 :
                     print(f"  No test predictions generated for {test_ds_suffix_to_predict} despite having test graphs. Check model output or filtering.")
                else:
                    print(f"  No test predictions generated for {test_ds_suffix_to_predict} (dataset might be empty).")
        else:
            print(f"No test data found for dataset {test_ds_suffix_to_predict}. Skipping predictions.")
        print("---------------------------------")
    else:
        print("\nSkipping test set predictions as --predict_on_test was not specified.")