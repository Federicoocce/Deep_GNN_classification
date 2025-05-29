# train_local_gatedgcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, Linear
import torch_geometric.utils
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset # Added Dataset
from torch_geometric.transforms import BaseTransform # Added BaseTransform
from sklearn.metrics import accuracy_score
import time
import numpy as np
import pandas as pd
import argparse
import os

# Import from your new data loader file
from data_loader import get_data_splits, RWSE_MAX_K # Assuming this is the correct filename
from losses import GCELoss, SCELoss, BootstrappingLoss, FocalLoss, MAELoss

# --- Hyperparameters ---
NUM_CLASSES = 6
GNN_LAYERS = 3
GNN_HIDDEN_DIM = 128
GNN_DROPOUT = 0.5
NODE_EMBEDDING_DIM = 128
EDGE_EMBEDDING_DIM = 128
NODE_CATEGORY_COUNT = 1
EDGE_FEATURE_DIM = 7
USE_RWSE_PE = False
PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0
USE_RESIDUAL = True
USE_FFN = False
USE_BATCHNORM = True
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1.0e-7
EPOCHS = 300
BATCH_SIZE = 32
NUM_WARMUP_EPOCHS = 10

DEFAULT_LABEL_SMOOTHING_FACTOR = 0.20
DEFAULT_EDGE_DROP_PROBABILITY = 0.20


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
        if len(self.data_list) != len(data_list) and len(data_list) > 0 : # only print if original list was not empty
            print(f"Warning: Filtered out {len(data_list) - len(self.data_list)} None graphs in ListDataset.")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# --- StandaloneGatedGCNLayer ---
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
            if x_trans.numel() > 0: x_trans = self.bn_node_x(x_trans)
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
            else: x_final = x_ffn_proc
        return x_final, e_final

# --- Model Definition ---
class MyLocalGatedGCN(torch.nn.Module):
    def __init__(self, current_use_rwse_pe, current_pe_dim):
        super().__init__()
        self.use_rwse_pe = current_use_rwse_pe
        self.node_encoder = nn.Embedding(num_embeddings=NODE_CATEGORY_COUNT, embedding_dim=NODE_EMBEDDING_DIM)
        self.edge_encoder = Linear(EDGE_FEATURE_DIM, EDGE_EMBEDDING_DIM)
        current_node_dim = NODE_EMBEDDING_DIM
        if self.use_rwse_pe: current_node_dim += current_pe_dim
        current_edge_dim = EDGE_EMBEDDING_DIM
        self.gnn_layers = nn.ModuleList()
        for i in range(GNN_LAYERS):
            in_node = current_node_dim if i == 0 else GNN_HIDDEN_DIM
            in_edge = current_edge_dim if i == 0 else GNN_HIDDEN_DIM
            self.gnn_layers.append(StandaloneGatedGCNLayer(in_node, in_edge, GNN_HIDDEN_DIM, GNN_DROPOUT, USE_RESIDUAL, USE_FFN, USE_BATCHNORM, lambda: nn.ReLU()))
        self.pool = global_mean_pool
        self.head = Linear(GNN_HIDDEN_DIM, NUM_CLASSES)

    def forward(self, data):
        x, edge_idx, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if x.dtype == torch.long: x_base = self.node_encoder(x.squeeze(-1))
        else:
            # print(f"Warning: Unexpected node feature type {x.dtype}. Casting to long.") # Reduced verbosity
            x_base = self.node_encoder(x.long().squeeze(-1))
        e_attr_enc = torch.empty((0, EDGE_EMBEDDING_DIM), device=x.device, dtype=x_base.dtype)
        if hasattr(edge_attr, 'numel') and edge_attr.numel() > 0 and edge_attr.size(0) > 0:
            expected_edge_dim = self.edge_encoder.weight.shape[1]
            if edge_attr.shape[1] == expected_edge_dim: e_attr_enc = self.edge_encoder(edge_attr)
            else:
                # print(f"Warning: Edge feature dim mismatch. Expected {expected_edge_dim}, got {edge_attr.shape[1]}. Using zeros.") # Reduced verbosity
                if edge_idx.numel() > 0: e_attr_enc = torch.zeros((edge_idx.shape[1], EDGE_EMBEDDING_DIM), device=x.device, dtype=x_base.dtype)
        current_x = x_base
        if self.use_rwse_pe and hasattr(data, 'rwse_pe') and data.rwse_pe is not None and data.rwse_pe.numel() > 0:
            pe = data.rwse_pe.float().to(x_base.device)
            if x_base.size(0) == pe.size(0): current_x = torch.cat([x_base, pe], dim=-1)
            elif x_base.size(0) > 0 and pe.size(0) > 0: 
                pass # print(f"Warning: RWSE PE node count mismatch. RWSE PE not used.") # Reduced verbosity
        current_e = e_attr_enc
        for layer in self.gnn_layers:
            current_x, current_e = layer(current_x, edge_idx, current_e)
        graph_x = self.pool(current_x, batch)
        return self.head(graph_x)

# --- Training and Evaluation Functions ---
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
        # This part is now only called if you explicitly pass is_test_set_preds_only=True
        # For validation, it will not be True.
        return torch.cat(all_preds_list).numpy() if all_preds_list else np.array([])
        
    if not all_labels_list : return 0,0 # if no labels were processed during validation
    all_preds_np = torch.cat(all_preds_list).numpy() if all_preds_list else np.array([])
    all_labels_np = torch.cat(all_labels_list).numpy() if all_labels_list else np.array([])
    valid_indices_overall = all_labels_np != -1
    accuracy = 0
    num_valid_samples_overall = np.sum(valid_indices_overall)
    if num_valid_samples_overall > 0:
        accuracy = accuracy_score(all_labels_np[valid_indices_overall], all_preds_np[valid_indices_overall])
    effective_loss = total_loss / num_valid_samples_overall if num_valid_samples_overall > 0 else 0
    return effective_loss, accuracy

# --- Loss Function Helper ---
def _get_loss(loss_name: str, label_smoothing_factor: float = 0.0, num_classes: int = NUM_CLASSES):
    loss_name = loss_name.lower()
    if loss_name == 'ce':
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing_factor if label_smoothing_factor > 0 else 0.0,
                                   ignore_index=-1) # CE handles ignore_index internally
    # For other losses, ensure they handle ignore_index or that masking is done before calling them
    elif loss_name == 'gce': return GCELoss(q=0.5)
    elif loss_name == 'sce': return SCELoss(alpha=1.0, beta=1.0)
    elif loss_name == 'bootstrapping': return BootstrappingLoss(beta=0.95, num_classes=num_classes) # Masking done in train/eval
    elif loss_name == 'focal': return FocalLoss(gamma=2.0, num_classes=num_classes, ignore_index=-1)
    elif loss_name == 'mae': return MAELoss() # Masking done in train/eval
    else: raise ValueError(f"Unknown loss function: {loss_name}")

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GatedGCN Training with A,B,C,D datasets')
    parser.add_argument('--force_reprocess_data', action='store_true', help="Force re-processing of data")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument('--no_rwse', action='store_false', dest='use_rwse_pe_cli', help="Disable RWSE Positional Encoding.")
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument('--ds', type=str, default='A', choices=['A', 'B', 'C', 'D'], help="Dataset SUFFIX to use for training (default: A).")
    parser.add_argument('--loss_fn', type=str, default='ce', choices=['ce', 'gce', 'sce', 'bootstrapping', 'focal', 'mae'], help="Loss function to use (default: ce).")
    parser.add_argument('--label_smoothing', type=float, default=DEFAULT_LABEL_SMOOTHING_FACTOR, help="Label smoothing factor (0.0 for no smoothing). Only for 'ce' loss.")
    parser.add_argument('--edge_drop_prob', type=float, default=DEFAULT_EDGE_DROP_PROBABILITY, help="Probability of dropping an edge during training (0.0 for no dropping).")
    # --- NEW: Argument to control test set prediction ---
    parser.add_argument('--predict_on_test', action='store_true', help="Generate predictions on the test set of the trained dataset suffix.")
    parser.set_defaults(use_rwse_pe_cli=USE_RWSE_PE)
    cli_args = parser.parse_args()

    LOSS_NAME = cli_args.loss_fn
    EPOCHS = cli_args.epochs
    LEARNING_RATE = cli_args.lr
    USE_RWSE_PE = cli_args.use_rwse_pe_cli
    PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0
    DATASET_SUFFIX_TO_TRAIN_ON = cli_args.ds
    LABEL_SMOOTHING_FACTOR = cli_args.label_smoothing
    EDGE_DROP_PROBABILITY = cli_args.edge_drop_prob
    PREDICT_ON_TEST_SET = cli_args.predict_on_test # Store the new argument

    print(f"--- Configuration ---")
    print(f"Training on Dataset: {DATASET_SUFFIX_TO_TRAIN_ON}")
    print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}")
    print(f"Model: Layers={GNN_LAYERS}, Hidden={GNN_HIDDEN_DIM}, NodeEmb={NODE_EMBEDDING_DIM}, EdgeEmb={EDGE_EMBEDDING_DIM}")
    print(f"NodeCatCount: {NODE_CATEGORY_COUNT}, EdgeFeatDim: {EDGE_FEATURE_DIM}")
    print(f"Dropout: {GNN_DROPOUT}, Residual: {USE_RESIDUAL}, FFN: {USE_FFN}, BN: {USE_BATCHNORM}")
    print(f"RWSE: {USE_RWSE_PE}, PE_Dim: {PE_DIM}")
    criterion = _get_loss(LOSS_NAME, label_smoothing_factor=LABEL_SMOOTHING_FACTOR if LOSS_NAME == 'ce' else 0.0)
    print(f"Loss Func: {LOSS_NAME}" + (f" (LS: {LABEL_SMOOTHING_FACTOR})" if LOSS_NAME == 'ce' and LABEL_SMOOTHING_FACTOR > 0 else ""))
    print(f"EdgeDrop (Train): {EDGE_DROP_PROBABILITY}")
    print(f"Predict on Test Set for DS {DATASET_SUFFIX_TO_TRAIN_ON}: {PREDICT_ON_TEST_SET}") # Print new arg status
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

    if train_graphs and hasattr(train_graphs[0], 'edge_attr') and train_graphs[0].edge_attr is not None and train_graphs[0].edge_attr.numel() > 0:
        actual_edge_dim = train_graphs[0].edge_attr.shape[1]
        if actual_edge_dim != EDGE_FEATURE_DIM:
            print(f"WARNING: Loaded training data (DS: {DATASET_SUFFIX_TO_TRAIN_ON}) has edge_attr_dim {actual_edge_dim}, model expects {EDGE_FEATURE_DIM}.")
    elif EDGE_FEATURE_DIM > 0 and train_graphs:
        print(f"WARNING: Model expects edge_attr_dim {EDGE_FEATURE_DIM}, but first graph in training data (DS: {DATASET_SUFFIX_TO_TRAIN_ON}) has no/empty edge_attr.")

    train_transform = None
    if EDGE_DROP_PROBABILITY > 0:
        train_transform = DropEdges(p=EDGE_DROP_PROBABILITY)
        print(f"Applying EdgeDropping (p={EDGE_DROP_PROBABILITY}) to training data of Dataset {DATASET_SUFFIX_TO_TRAIN_ON}.")
    
    train_dataset = ListDataset(train_graphs, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_dataset = ListDataset(val_graphs) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if val_graphs else None

    model = MyLocalGatedGCN(current_use_rwse_pe=USE_RWSE_PE, current_pe_dim=PE_DIM).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if EPOCHS > NUM_WARMUP_EPOCHS:
        def lr_lambda_fn(epoch):
            if epoch < NUM_WARMUP_EPOCHS: return float(epoch + 1) / float(NUM_WARMUP_EPOCHS + 1)
            progress = float(epoch - NUM_WARMUP_EPOCHS) / float(max(1, EPOCHS - NUM_WARMUP_EPOCHS))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_fn)

    print(f"\nStarting training for Dataset {DATASET_SUFFIX_TO_TRAIN_ON}...")
    best_val_acc = 0.0
    model_save_dir = 'models'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'best_gatedgcn_trained_on_DS_{DATASET_SUFFIX_TO_TRAIN_ON}.pth')

    for epoch_iter in range(1, EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = (0, 0)
        if val_loader and val_graphs: 
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        if val_loader and val_graphs and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"*** DS {DATASET_SUFFIX_TO_TRAIN_ON} - Best val_acc: {best_val_acc:.4f} (Epoch {epoch_iter}). Model saved to {model_save_path} ***")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"DS {DATASET_SUFFIX_TO_TRAIN_ON} - Epoch {epoch_iter:02d}/{EPOCHS:02d} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.4f} | LR: {current_lr:.1e} | Time: {(time.time()-start_time):.2f}s")
        if scheduler: scheduler.step()

    print(f"\nTraining for Dataset {DATASET_SUFFIX_TO_TRAIN_ON} finished.")

    # --- MODIFIED: Conditional Test Prediction ---
    if PREDICT_ON_TEST_SET:
        if os.path.exists(model_save_path):
            print(f"Loading best model (trained on DS {DATASET_SUFFIX_TO_TRAIN_ON}) from {model_save_path} for test predictions...")
            model.load_state_dict(torch.load(model_save_path, map_location=device))
        else:
            print(f"Warning: No best model saved for DS {DATASET_SUFFIX_TO_TRAIN_ON}. Using model from last epoch for testing.")

        print(f"\n--- Generating Test Predictions for Test Set of DS {DATASET_SUFFIX_TO_TRAIN_ON} ONLY ---")
        
        test_ds_suffix_to_predict = DATASET_SUFFIX_TO_TRAIN_ON # Only predict on the corresponding test set

        if test_ds_suffix_to_predict in all_dataset_splits and \
           all_dataset_splits[test_ds_suffix_to_predict].get('test'):
            
            current_test_graphs = all_dataset_splits[test_ds_suffix_to_predict]['test']
            print(f"Generating predictions for testset_{test_ds_suffix_to_predict}...")
            
            current_test_dataset = ListDataset(current_test_graphs) 
            current_test_loader = DataLoader(current_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            
            # Pass is_test_set_preds_only=True to get only predictions
            test_predictions_array = eval_epoch(model, current_test_loader, criterion, device, is_test_set_preds_only=True)
            
            num_test_samples = len(test_predictions_array)
            if num_test_samples > 0:
                ids = np.arange(num_test_samples) 
                predictions_df = pd.DataFrame({'id': ids, 'pred': test_predictions_array})
                
                # Output filename will be simple: testset_A.csv, testset_B.csv etc.
                output_filename = f'testset_{test_ds_suffix_to_predict}.csv'
                predictions_df.to_csv(output_filename, index=False)
                print(f"  Test predictions for {test_ds_suffix_to_predict} saved to {output_filename}")
            else:
                print(f"  No test predictions generated for {test_ds_suffix_to_predict} (dataset might be empty or predictions array is empty).")
        else:
            print(f"No test data found for dataset {test_ds_suffix_to_predict}. Skipping predictions.")
        print("---------------------------------")
    else:
        print("\nSkipping test set predictions as --predict_on_test was not specified.")