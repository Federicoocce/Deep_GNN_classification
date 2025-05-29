# train_local_gatedgcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, Linear
import torch_geometric.utils as pyg_utils
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.transforms import BaseTransform
from sklearn.metrics import accuracy_score
import time
import numpy as np
import pandas as pd
import argparse
import os

# Assuming data_loader.py is in the same directory or accessible via PYTHONPATH
from data_loader import get_data_splits, RWSE_MAX_K # Your existing data_loader
from rsgnn_graphclass_components import LinkPredictorMLP, ReconstructionLoss

# --- Hyperparameters ---
NUM_CLASSES = 6 # For your 6-class single-label subset
GNN_LAYERS = 3
GNN_HIDDEN_DIM = 256
GNN_DROPOUT = 0.5
NODE_EMBEDDING_DIM = 128
EDGE_EMBEDDING_DIM = 128
NODE_CATEGORY_COUNT = 1
EDGE_FEATURE_DIM = 7
USE_RWSE_PE = True
PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0
USE_RESIDUAL = True
USE_FFN = False
USE_BATCHNORM = True
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 200
BATCH_SIZE = 32
NUM_WARMUP_EPOCHS = 10
DEFAULT_EDGE_DROP_PROBABILITY = 0.1

# --- RS-GNN Hyperparameters ---
RSGNN_LP_MLP_HIDDEN_DIM = 64
RSGNN_LP_T_SMALL = 0.05
RSGNN_REC_LOSS_SIGMA = 1.0
RSGNN_REC_LOSS_NEG_SAMPLES_RATIO = 1.0
RSGNN_ALPHA_REC_LOSS = 0.1
RSGNN_LR_LP = 0.001
RSGNN_LP_TRAIN_STEPS = 1
RSGNN_GNN_TRAIN_STEPS = 1
RSGNN_INTERMEDIATE_GNN_LAYER_IDX = 0

# --- Edge Dropping Transform ---
class DropEdges(BaseTransform): # Your existing class
    def __init__(self, p=0.1): self.p = p
    def __call__(self, data):
        if self.p == 0 or not hasattr(data, 'edge_index') or data.edge_index is None or data.edge_index.size(1) == 0: return data
        new_data = data.clone(); num_edges = data.edge_index.size(1)
        mask = torch.rand(num_edges, device=data.edge_index.device) > self.p
        new_data.edge_index = data.edge_index[:, mask]
        if hasattr(data, 'edge_attr') and data.edge_attr is not None: new_data.edge_attr = data.edge_attr[mask]
        return new_data

# --- Custom Dataset Wrapper ---
class ListDataset(Dataset): # Your existing class, with y check
    def __init__(self, data_list, transform=None):
        super().__init__(transform=transform)
        valid_data = []
        if data_list:
            for g in data_list:
                if g is not None and hasattr(g, 'y') and g.y is not None:
                    # CrossEntropyLoss can use ignore_index=-1 for labels to skip
                    # So, graphs with y.item() == -1 can be kept if loss is configured
                    valid_data.append(g) 
        self.data_list = valid_data
        original_non_none_count = len([g for g in data_list if g is not None]) if data_list else 0
        if len(self.data_list) != original_non_none_count:
            print(f"Warning: Filtered out {original_non_none_count - len(self.data_list)} graphs due to missing 'y' or other issues in ListDataset.")
    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]


# --- StandaloneGatedGCNLayer --- (Your existing class, assumed to work with scalar weights)
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
        self.dropout_rate, self.residual_enabled = dropout, residual
        self.batchnorm_enabled, self.ffn_enabled, self.aggr = batchnorm_enabled, ffn_enabled, aggr
        if self.batchnorm_enabled:
            self.bn_node_x, self.bn_edge_e = nn.BatchNorm1d(out_dim), nn.BatchNorm1d(out_dim)
        self.residual_proj_node = Linear(in_dim_node, out_dim, bias=False) if residual and in_dim_node != out_dim else nn.Identity()
        self.residual_proj_edge = Linear(in_dim_edge, out_dim, bias=False) if residual and in_dim_edge != out_dim else nn.Identity()
        if self.ffn_enabled:
            self.ff_linear1, self.ff_linear2 = Linear(out_dim, out_dim * 2), Linear(out_dim * 2, out_dim)
            self.act_fn_ff = act_fn_constructor()
            self.ff_dropout1, self.ff_dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
            if self.batchnorm_enabled: self.norm1_ffn, self.norm2_ffn = nn.BatchNorm1d(out_dim), nn.BatchNorm1d(out_dim)

    def _ff_block(self, x):
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, x_in_node, edge_idx, edge_in_attr, edge_scalar_weights):
        x_ident, e_ident = x_in_node, edge_in_attr
        Ax, Bx, Ce, Dx, Ex = self.A(x_in_node), self.B(x_in_node), self.C(edge_in_attr), self.D(x_in_node), self.E(x_in_node)
        x_trans = Ax 
        e_trans = torch.zeros((0, self.out_dim), device=x_in_node.device, dtype=x_in_node.dtype)
        if edge_idx.numel() > 0:
            row, col = edge_idx
            e_ij = Dx[row] + Ex[col] + Ce
            if edge_scalar_weights.ndim == 1: edge_scalar_weights = edge_scalar_weights.unsqueeze(-1)
            aggr_messages = torch.sigmoid(e_ij) * Bx[col]
            weighted_aggr_messages = edge_scalar_weights * aggr_messages
            aggr_out = pyg_utils.scatter(weighted_aggr_messages, row, 0, dim_size=x_in_node.size(0), reduce=self.aggr)
            x_trans = Ax + aggr_out
            e_trans = e_ij 
        if self.batchnorm_enabled:
            if x_trans.numel() > 0: x_trans = self.bn_node_x(x_trans)
            if e_trans.numel() > 0 and hasattr(self.bn_edge_e, 'num_features') and self.bn_edge_e.num_features == e_trans.shape[1]:
                 e_trans = self.bn_edge_e(e_trans)
        x_trans = self.act_fn_x(x_trans)
        if e_trans.numel() > 0: e_trans = self.act_fn_e(e_trans)
        x_trans = F.dropout(x_trans, self.dropout_rate, training=self.training)
        if e_trans.numel() > 0: e_trans = F.dropout(e_trans, self.dropout_rate, training=self.training)
        x_final = self.residual_proj_node(x_ident) + x_trans if self.residual_enabled else x_trans
        e_final = (self.residual_proj_edge(e_ident) + e_trans) if self.residual_enabled and e_trans.numel() > 0 else e_trans
        if self.ffn_enabled and x_final.numel() > 0 :
            x_ffn_ident = x_final
            x_ffn_proc = self.norm1_ffn(x_ffn_ident) if self.batchnorm_enabled else x_ffn_ident
            x_ffn_proc = x_ffn_ident + self._ff_block(x_ffn_proc)
            x_final = self.norm2_ffn(x_ffn_proc) if self.batchnorm_enabled else x_ffn_proc
        return x_final, e_final

# --- Model Definition (MyLocalGatedGCN) --- (Your existing model)
class MyLocalGatedGCN(torch.nn.Module):
    def __init__(self, current_use_rwse_pe, current_pe_dim, num_graph_classes,
                 intermediate_layer_idx_for_lp=0):
        super().__init__()
        self.use_rwse_pe = current_use_rwse_pe
        self.node_encoder = nn.Embedding(num_embeddings=NODE_CATEGORY_COUNT, embedding_dim=NODE_EMBEDDING_DIM)
        self.edge_encoder = Linear(EDGE_FEATURE_DIM, EDGE_EMBEDDING_DIM)
        self.intermediate_layer_idx_for_lp = intermediate_layer_idx_for_lp
        self.intermediate_node_embeddings_for_lp = None
        current_node_dim_gnn_input = NODE_EMBEDDING_DIM
        if self.use_rwse_pe: current_node_dim_gnn_input += current_pe_dim
        self.gnn_layers = nn.ModuleList()
        for i in range(GNN_LAYERS):
            in_node = current_node_dim_gnn_input if i == 0 else GNN_HIDDEN_DIM
            in_edge_for_layer = EDGE_EMBEDDING_DIM
            self.gnn_layers.append(StandaloneGatedGCNLayer(
                in_node, in_edge_for_layer, GNN_HIDDEN_DIM, GNN_DROPOUT,
                USE_RESIDUAL, USE_FFN, USE_BATCHNORM, lambda: nn.ReLU()
            ))
        self.pool = global_mean_pool
        self.head = Linear(GNN_HIDDEN_DIM, num_graph_classes)

    def get_initial_node_features(self, data_x, data_rwse_pe):
        if data_x.dtype == torch.long: x_base = self.node_encoder(data_x.squeeze(-1))
        else: x_base = self.node_encoder(data_x.long().squeeze(-1))
        current_x = x_base
        if self.use_rwse_pe and data_rwse_pe is not None and data_rwse_pe.numel() > 0:
            pe = data_rwse_pe.float().to(x_base.device)
            if x_base.size(0) == pe.size(0): current_x = torch.cat([x_base, pe], dim=-1)
            elif x_base.size(0)>0 and pe.size(0) > 0:
                 print(f"Warning: RWSE PE node count mismatch ({x_base.size(0)} vs {pe.size(0)}). RWSE PE not used.")
        return current_x

    def forward(self, data, edge_scalar_weights_override=None):
        initial_node_feats = self.get_initial_node_features(data.x, getattr(data, 'rwse_pe', None))
        edge_idx, edge_attr_raw, batch = data.edge_index, data.edge_attr, data.batch
        encoded_edge_attrs = torch.empty((0, EDGE_EMBEDDING_DIM), device=initial_node_feats.device, dtype=initial_node_feats.dtype)
        if edge_idx is not None and edge_idx.numel() > 0 :
            if hasattr(edge_attr_raw, 'numel') and edge_attr_raw.numel() > 0 and edge_attr_raw.size(0) > 0:
                if edge_attr_raw.shape[1] == EDGE_FEATURE_DIM:
                    encoded_edge_attrs = self.edge_encoder(edge_attr_raw)
                else: 
                    encoded_edge_attrs = torch.zeros((edge_idx.shape[1], EDGE_EMBEDDING_DIM), device=initial_node_feats.device, dtype=initial_node_feats.dtype)
            elif edge_idx.numel() > 0:
                encoded_edge_attrs = torch.zeros((edge_idx.shape[1], EDGE_EMBEDDING_DIM), device=initial_node_feats.device, dtype=initial_node_feats.dtype)
        current_x_gnn = initial_node_feats
        current_e_features_for_gnn_layer = encoded_edge_attrs
        if edge_scalar_weights_override is None:
            if edge_idx is not None and edge_idx.numel() > 0:
                 edge_scalar_weights = torch.ones(edge_idx.size(1), device=current_x_gnn.device, dtype=current_x_gnn.dtype)
            else: 
                 edge_scalar_weights = torch.empty(0, device=current_x_gnn.device, dtype=current_x_gnn.dtype)
        else:
            edge_scalar_weights = edge_scalar_weights_override
            if edge_idx is not None and edge_idx.numel() > 0 and edge_scalar_weights.shape[0] != edge_idx.shape[1]:
                raise ValueError(f"edge_scalar_weights_override size ({edge_scalar_weights.shape[0]}) must match number of edges ({edge_idx.shape[1]})")
        for i, layer in enumerate(self.gnn_layers):
            current_x_gnn, _ = layer(current_x_gnn, edge_idx, current_e_features_for_gnn_layer, edge_scalar_weights)
            if i == self.intermediate_layer_idx_for_lp:
                self.intermediate_node_embeddings_for_lp = current_x_gnn
        graph_emb = self.pool(current_x_gnn, batch)
        out_graph = self.head(graph_emb)
        return out_graph

# --- Training and Evaluation Functions (Corrected for CrossEntropyLoss) ---
def train_epoch(gnn_model, link_predictor, recon_loss_module, loader, 
                optimizer_gnn, optimizer_lp, criterion_graph_classif, device, epoch_num):
    gnn_model.train()
    link_predictor.train()
    total_loss_gnn_main_agg = 0
    total_loss_rec_agg = 0
    processed_graphs_count = 0

    for data in loader:
        data = data.to(device)
        target_y = data.y.view(-1) # Ensure [batch_size] for CrossEntropyLoss

        with torch.no_grad():
            initial_node_features_for_recon_sim = gnn_model.get_initial_node_features(data.x, getattr(data, 'rwse_pe', None))

        for _ in range(RSGNN_LP_TRAIN_STEPS):
            optimizer_lp.zero_grad()
            _ = gnn_model(data) 
            current_intermediate_node_embs_for_lp = gnn_model.intermediate_node_embeddings_for_lp
            # Detach embeddings when input to LP to stabilize LP training from rapid GNN changes
            edge_weights_for_gnn_loss_lp_step = link_predictor(current_intermediate_node_embs_for_lp.detach(), data.edge_index)
            graph_preds_for_lp = gnn_model(data, edge_scalar_weights_override=edge_weights_for_gnn_loss_lp_step)
            loss_gnn_for_lp = criterion_graph_classif(graph_preds_for_lp, target_y)
            
            pred_weights_pos = link_predictor(current_intermediate_node_embs_for_lp.detach(), data.edge_index)
            loss_rec_pos = recon_loss_module.calculate_weighted_mse(
                pred_weights_pos, torch.ones_like(pred_weights_pos),
                data.edge_index, initial_node_features_for_recon_sim
            )
            loss_rec_neg = torch.tensor(0.0, device=device)
            if RSGNN_REC_LOSS_NEG_SAMPLES_RATIO > 0 and data.num_nodes > 1 and data.edge_index.numel() > 0:
                num_neg_edges = int(RSGNN_REC_LOSS_NEG_SAMPLES_RATIO * data.edge_index.size(1))
                if num_neg_edges == 0 and data.edge_index.size(1) > 0 : num_neg_edges = data.edge_index.size(1)
                if num_neg_edges > 0:
                    neg_edge_index = pyg_utils.negative_sampling(
                        edge_index=data.edge_index, num_nodes=data.num_nodes,
                        num_neg_samples=num_neg_edges, method='sparse' 
                    ).to(device)
                    if neg_edge_index.numel() > 0:
                         pred_weights_neg = link_predictor(current_intermediate_node_embs_for_lp.detach(), neg_edge_index)
                         loss_rec_neg = recon_loss_module.calculate_weighted_mse(
                            pred_weights_neg, torch.zeros_like(pred_weights_neg),
                            neg_edge_index, initial_node_features_for_recon_sim
                         )
            current_total_rec_loss = loss_rec_pos + loss_rec_neg
            total_loss_lp = loss_gnn_for_lp + RSGNN_ALPHA_REC_LOSS * current_total_rec_loss
            total_loss_lp.backward()
            optimizer_lp.step()

        for _ in range(RSGNN_GNN_TRAIN_STEPS):
            optimizer_gnn.zero_grad()
            with torch.no_grad():
                _ = gnn_model(data) 
                intermediate_embs_for_gnn_step = gnn_model.intermediate_node_embeddings_for_lp
                current_edge_weights_for_gnn = link_predictor(intermediate_embs_for_gnn_step, data.edge_index)
            graph_preds_for_gnn = gnn_model(data, edge_scalar_weights_override=current_edge_weights_for_gnn)
            loss_gnn_main = criterion_graph_classif(graph_preds_for_gnn, target_y)
            loss_gnn_main.backward()
            optimizer_gnn.step()

        total_loss_gnn_main_agg += loss_gnn_main.item() * target_y.size(0)
        total_loss_rec_agg += current_total_rec_loss.item() * target_y.size(0)
        processed_graphs_count += target_y.size(0)
        
    avg_loss_gnn = total_loss_gnn_main_agg / processed_graphs_count if processed_graphs_count > 0 else 0
    avg_loss_rec = total_loss_rec_agg / processed_graphs_count if processed_graphs_count > 0 else 0
    return {"gnn_loss": avg_loss_gnn, "rec_loss": avg_loss_rec}

@torch.no_grad()
def eval_epoch(gnn_model, link_predictor, loader, criterion_graph_classif, device, is_test_set_preds_only=False):
    gnn_model.eval()
    link_predictor.eval()
    total_loss_eval = 0
    all_graph_preds_list, all_graph_labels_list = [], []
    processed_graphs_count_eval = 0

    for data in loader:
        data = data.to(device)
        target_y = data.y.view(-1) # Ensure 1D for CrossEntropyLoss and accuracy_score

        _ = gnn_model(data)
        intermediate_embs_for_eval = gnn_model.intermediate_node_embeddings_for_lp
        edge_weights_eval = link_predictor(intermediate_embs_for_eval, data.edge_index)
        graph_logits = gnn_model(data, edge_scalar_weights_override=edge_weights_eval)
        
        if not is_test_set_preds_only:
            # Only compute loss if target_y is not -1 (or whatever ignore_index is)
            # This needs to be handled carefully if criterion_graph_classif has ignore_index set.
            # For now, assume all labels in val loader are valid after ListDataset filtering.
            loss = criterion_graph_classif(graph_logits, target_y) 
            total_loss_eval += loss.item() * target_y.size(0)
            all_graph_labels_list.append(target_y.cpu())
            processed_graphs_count_eval += target_y.size(0)
        
        graph_preds_classes = graph_logits.argmax(dim=1)
        all_graph_preds_list.append(graph_preds_classes.cpu())

    if is_test_set_preds_only:
        if not all_graph_preds_list: return np.array([])
        return torch.cat(all_graph_preds_list, dim=0).numpy()
    if not all_graph_labels_list : return 0, 0 
    avg_loss_eval = total_loss_eval / processed_graphs_count_eval if processed_graphs_count_eval > 0 else 0
    
    final_preds_np = torch.cat(all_graph_preds_list, dim=0).numpy()
    final_labels_np = torch.cat(all_graph_labels_list, dim=0).numpy()
    
    acc_metric = accuracy_score(final_labels_np, final_preds_np)
    return avg_loss_eval, acc_metric

# --- Loss Function Helper ---
def _get_graph_classif_loss(loss_name: str):
    loss_name = loss_name.lower()
    if loss_name == 'ce':
        print("Using CrossEntropyLoss for graph classification.")
        return nn.CrossEntropyLoss(ignore_index=-1) # Handles -1 labels if they appear
    elif loss_name == 'bce': # Kept for flexibility, but 'ce' is primary for this setup
        print("Using BCEWithLogitsLoss for multi-label graph classification.")
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown graph classification loss: {loss_name}. Use 'ce' or 'bce'.")

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GatedGCN Training with RS-GNN components')
    parser.add_argument('--force_reprocess_data', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--no_rwse', action='store_false', dest='use_rwse_pe_cli', default=USE_RWSE_PE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--lr_lp', type=float, default=RSGNN_LR_LP)
    parser.add_argument('--ds', type=str, default='A', choices=['A', 'B', 'C', 'D'])
    parser.add_argument('--graph_loss_fn', type=str, default='ce', choices=['ce', 'bce']) # DEFAULT TO 'ce'
    parser.add_argument('--edge_drop_prob', type=float, default=DEFAULT_EDGE_DROP_PROBABILITY)
    parser.add_argument('--predict_on_test', action='store_true', default=False)
    parser.add_argument('--alpha_rec', type=float, default=RSGNN_ALPHA_REC_LOSS)
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES) # Default is 6

    cli_args = parser.parse_args()

    EPOCHS = cli_args.epochs
    LEARNING_RATE = cli_args.lr
    RSGNN_LR_LP = cli_args.lr_lp
    USE_RWSE_PE = cli_args.use_rwse_pe_cli
    PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0
    DATASET_SUFFIX_TO_TRAIN_ON = cli_args.ds
    EDGE_DROP_PROBABILITY = cli_args.edge_drop_prob
    PREDICT_ON_TEST_SET = cli_args.predict_on_test
    RSGNN_ALPHA_REC_LOSS = cli_args.alpha_rec
    GRAPH_CLASSIF_LOSS_NAME = cli_args.graph_loss_fn
    NUM_CLASSES = cli_args.num_classes # This will be 6 based on your setup

    print(f"--- Configuration ---")
    print(f"Target NUM_CLASSES: {NUM_CLASSES}")
    print(f"Graph Classif Loss: {GRAPH_CLASSIF_LOSS_NAME}")
    # ... (rest of print statements for config) ...
    print(f"Training on Dataset: {DATASET_SUFFIX_TO_TRAIN_ON}")
    print(f"Epochs: {EPOCHS}, GNN_LR: {LEARNING_RATE}, LP_LR: {RSGNN_LR_LP}, Batch: {BATCH_SIZE}")
    print(f"Model: GNN_Layers={GNN_LAYERS}, GNN_Hidden={GNN_HIDDEN_DIM}, NodeEmb={NODE_EMBEDDING_DIM}, EdgeEmb={EDGE_EMBEDDING_DIM}")
    print(f"RWSE: {USE_RWSE_PE}, PE_Dim: {PE_DIM}")
    print(f"EdgeDrop (Train): {EDGE_DROP_PROBABILITY}")
    print(f"RS-GNN: AlphaRec={RSGNN_ALPHA_REC_LOSS}, LP_Hidden={RSGNN_LP_MLP_HIDDEN_DIM}, IntermediateGNNLayerForLP={RSGNN_INTERMEDIATE_GNN_LAYER_IDX}")
    print(f"Predict on Test Set for DS {DATASET_SUFFIX_TO_TRAIN_ON}: {PREDICT_ON_TEST_SET}")
    print(f"--------------------")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    criterion_graph_classif = _get_graph_classif_loss(GRAPH_CLASSIF_LOSS_NAME)
    
    print("Loading data splits from data_loader...")
    all_dataset_splits = get_data_splits(force_reprocess=cli_args.force_reprocess_data)

    if DATASET_SUFFIX_TO_TRAIN_ON not in all_dataset_splits:
        print(f"ERROR: Dataset suffix {DATASET_SUFFIX_TO_TRAIN_ON} not found. Available: {list(all_dataset_splits.keys())}")
        exit()

    current_ds_splits = all_dataset_splits[DATASET_SUFFIX_TO_TRAIN_ON]
    train_graphs = current_ds_splits.get('train', [])
    val_graphs = current_ds_splits.get('val', [])

    if not train_graphs: print(f"No training data for {DATASET_SUFFIX_TO_TRAIN_ON}. Exiting."); exit()

    train_transform = DropEdges(p=EDGE_DROP_PROBABILITY) if EDGE_DROP_PROBABILITY > 0 else None
    train_dataset = ListDataset(train_graphs, transform=train_transform)
    if not train_dataset.data_list: print("Training dataset is empty after filtering. Exiting."); exit()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=len(train_dataset.data_list)>BATCH_SIZE) 
    
    val_loader = None
    if val_graphs:
        val_dataset = ListDataset(val_graphs) 
        if val_dataset.data_list:
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    gnn_model = MyLocalGatedGCN(
        current_use_rwse_pe=USE_RWSE_PE, 
        current_pe_dim=PE_DIM,
        num_graph_classes=NUM_CLASSES,
        intermediate_layer_idx_for_lp=RSGNN_INTERMEDIATE_GNN_LAYER_IDX
    ).to(device)
    
    lp_input_node_emb_dim = GNN_HIDDEN_DIM
    if RSGNN_INTERMEDIATE_GNN_LAYER_IDX < 0 :
        lp_input_node_emb_dim = NODE_EMBEDDING_DIM + (PE_DIM if USE_RWSE_PE else 0)
        
    link_predictor = LinkPredictorMLP(
        node_emb_dim=lp_input_node_emb_dim, 
        mlp_hidden_dim=RSGNN_LP_MLP_HIDDEN_DIM,
        t_small=RSGNN_LP_T_SMALL
    ).to(device)
    recon_loss_module = ReconstructionLoss(sigma=RSGNN_REC_LOSS_SIGMA)

    optimizer_gnn = torch.optim.AdamW(gnn_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer_lp = torch.optim.AdamW(link_predictor.parameters(), lr=RSGNN_LR_LP, weight_decay=WEIGHT_DECAY)
    
    scheduler_gnn = None
    if EPOCHS > NUM_WARMUP_EPOCHS and NUM_WARMUP_EPOCHS > 0 :
        def lr_lambda_fn(epoch):
            if epoch < NUM_WARMUP_EPOCHS: return float(epoch + 1) / float(NUM_WARMUP_EPOCHS + 1)
            progress = float(epoch - NUM_WARMUP_EPOCHS) / float(max(1, EPOCHS - NUM_WARMUP_EPOCHS))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler_gnn = torch.optim.lr_scheduler.LambdaLR(optimizer_gnn, lr_lambda_fn)

    best_val_metric = 0.0
    model_save_dir = 'models_rsgnn_graphclass'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'best_rsgnn_gatedgcn_DS_{DATASET_SUFFIX_TO_TRAIN_ON}_cl{NUM_CLASSES}.pth')

    print(f"\nStarting training for Dataset {DATASET_SUFFIX_TO_TRAIN_ON} ({GRAPH_CLASSIF_LOSS_NAME} loss)...")
    for epoch_iter in range(1, EPOCHS + 1):
        start_time_epoch = time.time()
        train_losses = train_epoch(
            gnn_model, link_predictor, recon_loss_module, train_loader, 
            optimizer_gnn, optimizer_lp, criterion_graph_classif, device, epoch_iter
        )
        val_loss, val_metric = (0,0)
        if val_loader: 
            val_loss, val_metric = eval_epoch(
                gnn_model, link_predictor, val_loader, criterion_graph_classif, device
            )
        if val_loader and val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save({
                'gnn_model_state_dict': gnn_model.state_dict(),
                'link_predictor_state_dict': link_predictor.state_dict(),
                'epoch': epoch_iter, 'best_val_metric': best_val_metric
            }, model_save_path)
            print(f"*** DS {DATASET_SUFFIX_TO_TRAIN_ON} - Best val_acc: {best_val_metric:.4f} (Epoch {epoch_iter}). Model saved. ***")
        
        current_lr_gnn = optimizer_gnn.param_groups[0]['lr']
        current_lr_lp = optimizer_lp.param_groups[0]['lr']
        print(f"DS {DATASET_SUFFIX_TO_TRAIN_ON} - Ep {epoch_iter:02d}/{EPOCHS:02d} | "
              f"TrLossGNN: {train_losses['gnn_loss']:.4f} | TrLossRec: {train_losses['rec_loss']:.4f} | "
              f"ValLoss: {val_loss:.4f} | ValAcc: {val_metric:.4f} | "
              f"LR_GNN: {current_lr_gnn:.1e} | LR_LP: {current_lr_lp:.1e} | Time: {(time.time()-start_time_epoch):.2f}s")
        if scheduler_gnn: scheduler_gnn.step()

    print(f"\nTraining for Dataset {DATASET_SUFFIX_TO_TRAIN_ON} finished.")

    if PREDICT_ON_TEST_SET:
        if os.path.exists(model_save_path):
            print(f"Loading best model from {model_save_path} for test predictions...")
            checkpoint = torch.load(model_save_path, map_location=device)
            gnn_model.load_state_dict(checkpoint['gnn_model_state_dict'])
            link_predictor.load_state_dict(checkpoint['link_predictor_state_dict'])
        else:
            print(f"Warning: No best model saved. Using model from last epoch for testing.")

        print(f"\n--- Generating Test Predictions for Test Set of DS {DATASET_SUFFIX_TO_TRAIN_ON} ---")
        test_ds_suffix = DATASET_SUFFIX_TO_TRAIN_ON
        test_graphs_list = current_ds_splits.get('test', [])
        current_test_loader = None
        if test_graphs_list:
            # For test prediction, allow graphs even if y is -1 or missing
            test_dataset_unfiltered = Dataset() 
            test_dataset_unfiltered.data_list = [g for g in test_graphs_list if g is not None]
            test_dataset_unfiltered.len = lambda: len(test_dataset_unfiltered.data_list)
            test_dataset_unfiltered.get = lambda idx: test_dataset_unfiltered.data_list[idx]
            if test_dataset_unfiltered.data_list:
                current_test_loader = DataLoader(test_dataset_unfiltered, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        if current_test_loader:
            test_predictions_classes_np = eval_epoch(
                gnn_model, link_predictor, current_test_loader, criterion_graph_classif, device, is_test_set_preds_only=True
            )
            if test_predictions_classes_np.size > 0:
                ids = np.arange(len(test_predictions_classes_np))
                predictions_df = pd.DataFrame({'id': ids, 'pred': test_predictions_classes_np}) # pred contains class index
                output_filename = f'testset_rsgnn_{test_ds_suffix}_cl{NUM_CLASSES}_preds.csv'
                predictions_df.to_csv(output_filename, index=False)
                print(f"  Test predictions (classes) for {test_ds_suffix} saved to {output_filename}")
            else: print(f"  No test predictions generated for {test_ds_suffix}.")
        else:
            print(f"No test data/loader for {test_ds_suffix}. Skipping predictions.")
        print("---------------------------------")
    else:
        print("\nSkipping test set predictions.")