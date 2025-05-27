# train_local_gatedgcn_organized.py
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, Linear
import torch_geometric.utils
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
import time
import numpy as np
import pandas as pd
import argparse
import os

# Import from your new data loader file
from data_loader import get_data_splits, RWSE_MAX_K
from losses import GCELoss, SCELoss, BootstrappingLoss, FocalLoss, MAELoss, ELRLoss


class ModelConfig:
    """Configuration class for model hyperparameters"""
    # Model Architecture
    NUM_CLASSES = 6
    GNN_LAYERS = 4
    GNN_HIDDEN_DIM = 256
    GNN_DROPOUT = 0.3
    NODE_EMBEDDING_DIM = 128
    EDGE_EMBEDDING_DIM = 128

    # Data dimensions
    NODE_CATEGORY_COUNT = 1
    EDGE_FEATURE_DIM = 7

    # Positional Encoding
    USE_RWSE_PE = False
    PE_DIM = RWSE_MAX_K if USE_RWSE_PE else 0

    # GNN Features
    USE_RESIDUAL = True
    USE_FFN = False
    USE_BATCHNORM = True

    # Training
    LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 1.0e-5
    EPOCHS = 300
    BATCH_SIZE = 16
    NUM_WARMUP_EPOCHS = 10

    # Early Learning Regularization
    USE_ELR = False

    # Early Stopping
    EARLY_STOPPING = False
    EARLY_STOPPING_PATIENCE = 8
    EARLY_STOPPING_MIN_DELTA = 0.01

    # Co-Teaching
    CO_TEACHING = False
    MAX_CO_TEACHING_FORGET_RATE = 0.20  # Percentage of samples to forget each epoch
    NUM_GRADUAL = 5  # Epochs before starting to forget samples
    FORGET_RATE_MAX_EPOCH = 30  # Maximum epoch to reach max forget rate
    EXPONENT = 0.25  # Rate of forget rate increase


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving"""

    def __init__(self, patience=50, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            return False

        if self.mode == 'max':
            improved = val_score > self.best_score + self.min_delta
        else:
            improved = val_score < self.best_score - self.min_delta

        if improved:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


class StandaloneGatedGCNLayer(torch.nn.Module):
    """Individual GatedGCN layer implementation"""

    def __init__(self, in_dim_node, in_dim_edge, out_dim, dropout, residual, ffn_enabled,
                 batchnorm_enabled, act_fn_constructor, aggr='add', **kwargs):
        super().__init__(**kwargs)
        self.in_dim_node, self.in_dim_edge, self.out_dim = in_dim_node, in_dim_edge, out_dim
        self.activation = act_fn_constructor()
        self.A = Linear(in_dim_node, out_dim, bias=True)
        self.B = Linear(in_dim_node, out_dim, bias=True)
        self.C = Linear(in_dim_edge, out_dim, bias=True)
        self.D = Linear(in_dim_node, out_dim, bias=True)
        self.E = Linear(in_dim_node, out_dim, bias=True)

        self.act_fn_x, self.act_fn_e = self.activation, self.activation
        self.dropout_rate, self.residual_enabled, self.e_prop = dropout, residual, None
        self.batchnorm_enabled, self.ffn_enabled, self.aggr = batchnorm_enabled, ffn_enabled, aggr

        if self.batchnorm_enabled:
            self.bn_node_x = nn.BatchNorm1d(out_dim)
            self.bn_edge_e = nn.BatchNorm1d(out_dim)

        self.residual_proj_node = Linear(in_dim_node, out_dim,
                                         bias=False) if residual and in_dim_node != out_dim else nn.Identity()
        self.residual_proj_edge = Linear(in_dim_edge, out_dim,
                                         bias=False) if residual and in_dim_edge != out_dim else nn.Identity()

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

    def forward(self, x_in_node, edge_idx, edge_in_attr):
        x_ident, e_ident = x_in_node, edge_in_attr
        Ax, Bx, Ce, Dx, Ex = self.A(x_in_node), self.B(x_in_node), self.C(edge_in_attr), self.D(x_in_node), self.E(
            x_in_node)

        if edge_idx.numel() > 0:
            row, col = edge_idx
            e_ij = Dx[row] + Ex[col] + Ce
            self.e_prop = e_ij
            aggr_out = torch_geometric.utils.scatter(torch.sigmoid(e_ij) * Bx[col], row, 0, dim_size=x_in_node.size(0),
                                                     reduce=self.aggr)
            x_trans, e_trans = Ax + aggr_out, self.e_prop
        else:
            x_trans, e_trans = Ax, torch.zeros((0, self.out_dim), device=x_in_node.device, dtype=x_in_node.dtype)

        if self.batchnorm_enabled:
            x_trans = self.bn_node_x(x_trans)
            if e_trans.numel() > 0:
                e_trans = self.bn_edge_e(e_trans)

        x_trans = self.act_fn_x(x_trans)
        if e_trans.numel() > 0:
            e_trans = self.act_fn_e(e_trans)

        x_trans = F.dropout(x_trans, self.dropout_rate, training=self.training)
        if e_trans.numel() > 0:
            e_trans = F.dropout(e_trans, self.dropout_rate, training=self.training)

        x_final = self.residual_proj_node(x_ident) + x_trans if self.residual_enabled else x_trans
        e_final = (self.residual_proj_edge(
            e_ident) + e_trans) if self.residual_enabled and e_trans.numel() > 0 else e_trans

        if self.ffn_enabled:
            x_ffn_ident = x_final
            x_ffn_proc = self.norm1_ffn(
                x_ffn_ident) if self.batchnorm_enabled and x_ffn_ident.numel() > 0 else x_ffn_ident
            if x_ffn_proc.numel() > 0:
                x_ffn_proc = x_ffn_ident + self._ff_block(x_ffn_proc)
                x_final = self.norm2_ffn(x_ffn_proc) if self.batchnorm_enabled else x_ffn_proc
            else:
                x_final = x_ffn_proc
        return x_final, e_final


class MyLocalGatedGCN(torch.nn.Module):
    """Main GatedGCN model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_rwse_pe = config.USE_RWSE_PE

        # Node encoder
        self.node_encoder = nn.Embedding(num_embeddings=config.NODE_CATEGORY_COUNT,
                                         embedding_dim=config.NODE_EMBEDDING_DIM)

        # Edge encoder
        self.edge_encoder = Linear(config.EDGE_FEATURE_DIM, config.EDGE_EMBEDDING_DIM)

        current_node_dim = config.NODE_EMBEDDING_DIM
        if self.use_rwse_pe:
            current_node_dim += config.PE_DIM

        current_edge_dim = config.EDGE_EMBEDDING_DIM

        self.gnn_layers = nn.ModuleList()
        for i in range(config.GNN_LAYERS):
            in_node = current_node_dim if i == 0 else config.GNN_HIDDEN_DIM
            in_edge = current_edge_dim if i == 0 else config.GNN_HIDDEN_DIM
            self.gnn_layers.append(
                StandaloneGatedGCNLayer(in_node, in_edge, config.GNN_HIDDEN_DIM, config.GNN_DROPOUT,
                                        config.USE_RESIDUAL, config.USE_FFN, config.USE_BATCHNORM, lambda: nn.ReLU())
            )
        self.pool = global_mean_pool
        self.head = Linear(config.GNN_HIDDEN_DIM, config.NUM_CLASSES)

    def forward(self, data):
        x, edge_idx, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if x.dtype == torch.long:
            x_base = self.node_encoder(x.squeeze(-1))
        else:
            x_base = self.node_encoder(x.long().squeeze(-1))

        e_attr_enc = torch.empty((0, self.config.EDGE_EMBEDDING_DIM), device=x.device, dtype=x_base.dtype)
        if hasattr(edge_attr, 'numel') and edge_attr.numel() > 0:
            if edge_attr.size(0) > 0:
                expected_edge_dim_from_encoder = self.edge_encoder.weight.shape[1]
                if edge_attr.shape[1] == expected_edge_dim_from_encoder:
                    e_attr_enc = self.edge_encoder(edge_attr)
                else:
                    if edge_idx.numel() > 0:
                        num_edges = edge_idx.shape[1]
                        e_attr_enc = torch.zeros((num_edges, self.config.EDGE_EMBEDDING_DIM),
                                                 device=x.device, dtype=x_base.dtype)

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


class LossManager:
    """Manages loss functions"""

    @staticmethod
    def get_loss(loss_name: str):
        loss = loss_name.lower()
        if loss == 'ce':
            return nn.CrossEntropyLoss()
        elif loss == 'gce':
            return GCELoss(q=0.7)
        elif loss == 'sce':
            return SCELoss(alpha=1.0, beta=1.0)
        elif loss == 'bootstrapping':
            return BootstrappingLoss(beta=0.95)
        elif loss == 'focal':
            return FocalLoss(gamma=2.0)
        elif loss == 'mae':
            return MAELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss}")


class CoTeachingTrainer:
    """Handles training and evaluation logic for co-teaching"""

    def __init__(self, model1, model2, config, device):
        self.model1 = model1
        self.model2 = model2
        self.config = config
        self.device = device

        # Initialize forget rate schedule
        self.epoch = 0
        self.num_gradual = config.NUM_GRADUAL  # When to start forgetting
        self.exponent = config.EXPONENT  # How fast to increase forget rate
        self.max_forget_rate = config.MAX_CO_TEACHING_FORGET_RATE
        self.forget_rate_max_epoch = config.FORGET_RATE_MAX_EPOCH
    def _get_current_forget_rate(self):
        """Calculate forget rate for current epoch with exponential growth"""
        if self.epoch < self.num_gradual:
            return 0.0
        else:
            progress = (self.epoch - self.num_gradual) / (self.forget_rate_max_epoch - self.num_gradual)
            rate = self.max_forget_rate * (progress ** self.exponent)
            return min(self.max_forget_rate, rate)

    def _select_samples_by_loss(self, data_batch, selecting_model, forget_rate):
        """
        Use selecting_model to choose clean samples based on small loss
        Returns indices of samples to keep (clean samples)
        """
        selecting_model.eval()
        with torch.no_grad():
            outputs = selecting_model(data_batch)
            targets = data_batch.y.squeeze()
            if targets.ndim == 0:
                targets = targets.unsqueeze(0)

            # Calculate loss for each sample
            losses = F.cross_entropy(outputs, targets, reduction='none')

            # Select samples with smallest loss (presumably clean)
            num_remember = max(1, int((1 - forget_rate) * len(losses)))
            _, clean_indices = torch.topk(-losses, num_remember)  # Negative for smallest

        return clean_indices.cpu().numpy()

    def train_epoch(self, loader, optimizer1, optimizer2, criterion):
        """Train one epoch with co-teaching"""
        self.model1.train()
        self.model2.train()

        total_loss1, total_loss2 = 0, 0
        processed_graphs = 0

        current_forget_rate = self._get_current_forget_rate()

        for batch_data in loader:
            batch_data = batch_data.to(self.device)

            # Convert to list for indexing
            data_list = batch_data.to_data_list()
            batch_size = len(data_list)

            if batch_size == 1:
                # If only one sample, both models train on it
                clean_indices_for_model1 = [0]
                clean_indices_for_model2 = [0]
            else:
                # Model2 selects clean samples for Model1 to train on
                clean_indices_for_model1 = self._select_samples_by_loss(
                    batch_data, self.model2, current_forget_rate
                )

                # Model1 selects clean samples for Model2 to train on
                clean_indices_for_model2 = self._select_samples_by_loss(
                    batch_data, self.model1, current_forget_rate
                )

            # Create clean data batches
            clean_data_for_model1 = Batch.from_data_list([data_list[i] for i in clean_indices_for_model1])
            clean_data_for_model2 = Batch.from_data_list([data_list[i] for i in clean_indices_for_model2])

            # Train Model1 on samples selected by Model2
            if len(clean_indices_for_model1) > 0:
                optimizer1.zero_grad()
                outputs1 = self.model1(clean_data_for_model1)
                targets1 = clean_data_for_model1.y.squeeze()
                if targets1.ndim == 0:
                    targets1 = targets1.unsqueeze(0)
                loss1 = criterion(outputs1, targets1)
                loss1.backward()
                optimizer1.step()
                total_loss1 += loss1.item() * len(clean_indices_for_model1)

            # Train Model2 on samples selected by Model1
            if len(clean_indices_for_model2) > 0:
                optimizer2.zero_grad()
                outputs2 = self.model2(clean_data_for_model2)
                targets2 = clean_data_for_model2.y.squeeze()
                if targets2.ndim == 0:
                    targets2 = targets2.unsqueeze(0)
                loss2 = criterion(outputs2, targets2)
                loss2.backward()
                optimizer2.step()
                total_loss2 += loss2.item() * len(clean_indices_for_model2)

            processed_graphs += batch_size

        # Update epoch counter
        self.epoch += 1

        avg_loss1 = total_loss1 / processed_graphs if processed_graphs > 0 else 0
        avg_loss2 = total_loss2 / processed_graphs if processed_graphs > 0 else 0

        return avg_loss1, avg_loss2, current_forget_rate

    @torch.no_grad()
    def eval_epoch(self, loader, criterion, is_test_set_preds_only=False):
        """Evaluate both models"""
        self.model1.eval()
        self.model2.eval()

        total_loss1, total_loss2 = 0, 0
        processed_graphs = 0
        all_preds1, all_preds2 = [], []
        all_labels = []

        for data in loader:
            data = data.to(self.device)

            # Get predictions from both models
            out1 = self.model1(data)
            out2 = self.model2(data)

            preds1 = out1.argmax(dim=1)
            preds2 = out2.argmax(dim=1)
            all_preds1.append(preds1.cpu())
            all_preds2.append(preds2.cpu())

            if not is_test_set_preds_only:
                targets = data.y.squeeze()
                if targets.ndim == 0:
                    targets = targets.unsqueeze(0)

                valid_mask = targets != -1
                if valid_mask.any():
                    loss1 = criterion(out1[valid_mask], targets[valid_mask])
                    loss2 = criterion(out2[valid_mask], targets[valid_mask])
                    total_loss1 += loss1.item() * valid_mask.sum().item()
                    total_loss2 += loss2.item() * valid_mask.sum().item()

                all_labels.append(targets.cpu())

            processed_graphs += data.num_graphs

        # Combine predictions
        all_preds1 = torch.cat(all_preds1).numpy() if all_preds1 else np.array([])
        all_preds2 = torch.cat(all_preds2).numpy() if all_preds2 else np.array([])

        if is_test_set_preds_only:
            return all_preds1, all_preds2

        if not all_labels:
            return (0, 0), (0, 0)

        all_labels = torch.cat(all_labels).numpy()
        valid_mask = all_labels != -1

        acc1 = acc2 = 0
        if valid_mask.sum() > 0:
            from sklearn.metrics import accuracy_score
            acc1 = accuracy_score(all_labels[valid_mask], all_preds1[valid_mask])
            acc2 = accuracy_score(all_labels[valid_mask], all_preds2[valid_mask])

        avg_loss1 = total_loss1 / valid_mask.sum() if valid_mask.sum() > 0 else 0
        avg_loss2 = total_loss2 / valid_mask.sum() if valid_mask.sum() > 0 else 0

        return (avg_loss1, avg_loss2), (acc1, acc2)


class SingleModelTrainer:
    """Handles training and evaluation logic for single model"""

    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

    def train_epoch(self, loader, optimizer, criterion, teacher_model=None):
        self.model.train()
        total_loss, processed_graphs = 0, 0
        for data in loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            out = self.model(data)
            target_y = data.y.squeeze()
            if target_y.ndim == 0:
                target_y = target_y.unsqueeze(0)

            if teacher_model:
                loss = criterion(out, target_y, data)
            else:
                loss = criterion(out, target_y)

            loss.backward()
            optimizer.step()

            if teacher_model:
                teacher_model.load_state_dict(self.model.state_dict())

            total_loss += loss.item() * data.num_graphs
            processed_graphs += data.num_graphs
        return total_loss / processed_graphs if processed_graphs else 0

    @torch.no_grad()
    def eval_epoch(self, loader, criterion, is_test_set_preds_only=False):
        self.model.eval()
        total_loss, processed_graphs = 0, 0
        all_preds_list, all_labels_list = [], []

        for data in loader:
            data = data.to(self.device)
            out = self.model(data)
            preds = out.argmax(dim=1)
            all_preds_list.append(preds.cpu())

            if not is_test_set_preds_only:
                target_y = data.y.squeeze()
                if target_y.ndim == 0:
                    target_y = target_y.unsqueeze(0)
                valid_targets = target_y != -1

                if valid_targets.any():
                    if self.config.USE_ELR:
                        loss = criterion(out[valid_targets], target_y[valid_targets], data)
                    else:
                        loss = criterion(out[valid_targets], target_y[valid_targets])
                    total_loss += loss.item() * torch.sum(valid_targets).item()
                all_labels_list.append(target_y.cpu())
            processed_graphs += data.num_graphs

        if is_test_set_preds_only:
            return torch.cat(all_preds_list).numpy() if all_preds_list else np.array([])

        if not all_labels_list and not is_test_set_preds_only:
            return 0, 0

        all_preds_np = torch.cat(all_preds_list).numpy()
        all_labels_np = torch.cat(all_labels_list).numpy()

        valid_indices = all_labels_np != -1
        accuracy = 0
        if np.sum(valid_indices) > 0:
            accuracy = accuracy_score(all_labels_np[valid_indices], all_preds_np[valid_indices])

        effective_loss = total_loss / np.sum(valid_indices) if np.sum(valid_indices) > 0 else 0
        return effective_loss, accuracy


class GatedGCNTrainingSystem:
    """Main training system that orchestrates everything"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model1 = None
        self.model2 = None
        self.teacher_model1 = None
        self.teacher_model2 = None
        self.trainer = None
        self.data = {}

    def setup_model(self):
        """Initialize models and trainers"""
        if self.config.CO_TEACHING:
            self.model1 = MyLocalGatedGCN(self.config).to(self.device)
            self.model2 = MyLocalGatedGCN(self.config).to(self.device)
            if self.config.USE_ELR:
                self.teacher_model1 = copy.deepcopy(self.model1)
                self.teacher_model2 = copy.deepcopy(self.model2)
            self.trainer = CoTeachingTrainer(self.model1, self.model2, self.config, self.device)

            num_params = sum(p.numel() for p in self.model1.parameters() if p.requires_grad)
            print(f"Number of trainable parameters per model: {num_params:,} (two models in co-teaching)")
        else:
            self.model = MyLocalGatedGCN(self.config).to(self.device)
            if self.config.USE_ELR:
                self.teacher_model = copy.deepcopy(self.model)
            self.trainer = SingleModelTrainer(self.model, self.config, self.device)

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {num_params:,}")

    def load_data(self, dataset_name, force_reprocess=False):
        """Load and prepare data"""
        print("Loading data from full_data_loader...")
        all_loaded_splits = get_data_splits(force_reprocess=force_reprocess)

        self.data['train'] = all_loaded_splits[dataset_name]['train']
        self.data['val'] = all_loaded_splits[dataset_name]['val']

        if not self.data['train'] or not self.data['val']:
            print(f"No training data found for dataset {dataset_name}. Exiting.")
            return False

        # Load test datasets for all datasets
        self.data['test_datasets'] = {}
        for ds_name in ['A', 'B', 'C', 'D']:
            self.data['test_datasets'][ds_name] = all_loaded_splits[ds_name]['test']

        self._validate_data_dimensions()
        return True

    def _validate_data_dimensions(self):
        """Validate data dimensions match model expectations"""
        if (self.data['train'] and hasattr(self.data['train'][0], 'edge_attr')
                and self.data['train'][0].edge_attr is not None
                and self.data['train'][0].edge_attr.numel() > 0):

            actual_edge_dim = self.data['train'][0].edge_attr.shape[1]
            if actual_edge_dim != self.config.EDGE_FEATURE_DIM:
                print(f"WARNING: Loaded training data has edge_attr dim {actual_edge_dim}, "
                      f"but model expects {self.config.EDGE_FEATURE_DIM}.")

    def create_data_loaders(self):
        """Create PyTorch Geometric data loaders"""
        train_loader = DataLoader(self.data['train'], batch_size=self.config.BATCH_SIZE,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(self.data['val'], batch_size=self.config.BATCH_SIZE,
                                shuffle=False, num_workers=0) if self.data['val'] else None
        return train_loader, val_loader

    def create_optimizers_and_schedulers(self):
        """Create optimizers and learning rate schedulers"""
        if self.config.CO_TEACHING:
            optimizer1 = torch.optim.AdamW(self.model1.parameters(),
                                           lr=self.config.LEARNING_RATE,
                                           weight_decay=self.config.WEIGHT_DECAY)
            optimizer2 = torch.optim.AdamW(self.model2.parameters(),
                                           lr=self.config.LEARNING_RATE,
                                           weight_decay=self.config.WEIGHT_DECAY)

            scheduler1 = None
            scheduler2 = None
            if self.config.EPOCHS > self.config.NUM_WARMUP_EPOCHS:
                def lr_lambda_fn(current_epoch_internal):
                    if current_epoch_internal < self.config.NUM_WARMUP_EPOCHS:
                        return float(current_epoch_internal + 1) / float(self.config.NUM_WARMUP_EPOCHS + 1)
                    else:
                        progress = float(current_epoch_internal - self.config.NUM_WARMUP_EPOCHS) / float(
                            max(1, self.config.EPOCHS - self.config.NUM_WARMUP_EPOCHS))
                        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

                scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda_fn)
                scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda_fn)

            return (optimizer1, optimizer2), (scheduler1, scheduler2)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.config.LEARNING_RATE,
                                          weight_decay=self.config.WEIGHT_DECAY)

            scheduler = None
            if self.config.EPOCHS > self.config.NUM_WARMUP_EPOCHS:
                def lr_lambda_fn(current_epoch_internal):
                    if current_epoch_internal < self.config.NUM_WARMUP_EPOCHS:
                        return float(current_epoch_internal + 1) / float(self.config.NUM_WARMUP_EPOCHS + 1)
                    else:
                        progress = float(current_epoch_internal - self.config.NUM_WARMUP_EPOCHS) / float(
                            max(1, self.config.EPOCHS - self.config.NUM_WARMUP_EPOCHS))
                        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_fn)

            return optimizer, scheduler

    def train(self, dataset_name, loss_fn='ce', force_reprocess=False, use_elr=False):
        """Main training loop"""
        if not self.load_data(dataset_name, force_reprocess):
            return

        self.setup_model()
        train_loader, val_loader = self.create_data_loaders()

        if self.config.CO_TEACHING:
            (optimizer1, optimizer2), (scheduler1, scheduler2) = self.create_optimizers_and_schedulers()
            criterion = LossManager.get_loss(loss_fn)
        else:
            optimizer, scheduler = self.create_optimizers_and_schedulers()
            criterion = LossManager.get_loss(loss_fn)
            if self.config.USE_ELR:
                criterion = ELRLoss(self.teacher_model, criterion=criterion)

        early_stopping = None
        if self.config.EARLY_STOPPING:
            early_stopping = EarlyStopping(
                patience=self.config.EARLY_STOPPING_PATIENCE,
                min_delta=self.config.EARLY_STOPPING_MIN_DELTA,
                mode='max'
            )

        self._print_config(dataset_name, loss_fn, criterion, use_elr)

        print("\nStarting training...")
        best_val_acc = 0.0
        model_save_dir = 'models'
        os.makedirs(model_save_dir, exist_ok=True)

        if self.config.CO_TEACHING:
            model1_save_path = os.path.join(model_save_dir, 'best_gatedgcn_multids_model1.pth')
            model2_save_path = os.path.join(model_save_dir, 'best_gatedgcn_multids_model2.pth')
        else:
            model_save_path = os.path.join(model_save_dir, 'best_gatedgcn_multids.pth')

        for epoch in range(1, self.config.EPOCHS + 1):
            start_time = time.time()

            if self.config.CO_TEACHING:
                train_loss1, train_loss2, forget_rate = self.trainer.train_epoch(train_loader, optimizer1, optimizer2,
                                                                                 criterion)
                print(f"Epoch {epoch:02d} | Forget Rate: {forget_rate:.4f}")
                if val_loader and self.data['val']:
                    (val_loss1, val_loss2), (val_acc1, val_acc2) = self.trainer.eval_epoch(val_loader, criterion)
                    val_acc = max(val_acc1, val_acc2)  # Track the better of the two models

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(self.model1.state_dict(), model1_save_path)
                        torch.save(self.model2.state_dict(), model2_save_path)
                        print(f"*** Best val_acc: {best_val_acc:.4f} (Epoch {epoch}). Models saved ***")

                current_lr_val = optimizer1.param_groups[0]['lr']
                print(f"Epoch {epoch:02d}/{self.config.EPOCHS:02d} | "
                      f"TrainLoss1: {train_loss1:.4f} | TrainLoss2: {train_loss2:.4f} | "
                      f"ValLoss1: {val_loss1:.4f} | ValLoss2: {val_loss2:.4f} | "
                      f"ValAcc1: {val_acc1:.4f} | ValAcc2: {val_acc2:.4f} | "
                      f"LR: {current_lr_val:.1e} | Time: {(time.time() - start_time):.2f}s")

                if scheduler1 and scheduler2:
                    scheduler1.step()
                    scheduler2.step()
            else:
                teacher_model = self.teacher_model if self.config.USE_ELR else None
                train_loss = self.trainer.train_epoch(train_loader, optimizer, criterion, teacher_model)

                if val_loader and self.data['val']:
                    val_loss, val_acc = self.trainer.eval_epoch(val_loader, criterion)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(self.model.state_dict(), model_save_path)
                        print(
                            f"*** Best val_acc: {best_val_acc:.4f} (Epoch {epoch}). Model saved to {model_save_path} ***")

                current_lr_val = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:02d}/{self.config.EPOCHS:02d} | TrainLoss: {train_loss:.4f} | "
                      f"ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.4f} | LR: {current_lr_val:.1e} | "
                      f"Time: {(time.time() - start_time):.2f}s")

                if scheduler:
                    scheduler.step()

            # Check early stopping condition
            if early_stopping and val_loader and self.data['val']:
                if early_stopping(val_acc):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best validation accuracy: {early_stopping.best_score:.4f}")
                    break

        print("\nTraining finished.")
        self._generate_test_predictions(criterion)

    def _print_config(self, dataset_name, loss_fn, criterion, use_elr=False):
        """Print training configuration"""
        print(f"\n--- Configuration ---")
        print(f"Training Mode: {'Co-Teaching' if self.config.CO_TEACHING else 'Single Model'}")
        if self.config.CO_TEACHING:
            print(f"Forget Rate: {self.config.MAX_CO_TEACHING_FORGET_RATE}")
        print(f"Epochs: {self.config.EPOCHS}, LR: {self.config.LEARNING_RATE}, Batch Size: {self.config.BATCH_SIZE}")
        print(f"Model: Layers={self.config.GNN_LAYERS}, HiddenDim={self.config.GNN_HIDDEN_DIM}")
        print(f"NodeEmb={self.config.NODE_EMBEDDING_DIM}, EdgeEmb={self.config.EDGE_EMBEDDING_DIM}")
        print(f"Criterion used: {criterion}")
        print(f"Early Learning Regularization: {use_elr}")
        print(f"Early Stopping: {self.config.EARLY_STOPPING}")
        if self.config.EARLY_STOPPING:
            print(f"  - Patience: {self.config.EARLY_STOPPING_PATIENCE}")
            print(f"  - Min Delta: {self.config.EARLY_STOPPING_MIN_DELTA}")
        print(f"Dataset: {dataset_name}")
        print(f"Using device: {self.device}")
        print(f"--------------------\n")

    def _generate_test_predictions(self, criterion):
        """Generate predictions for all test datasets"""
        print("\n--- Generating Test Predictions ---")
        for ds_name, current_test_graphs in self.data['test_datasets'].items():
            if current_test_graphs:
                print(f"Generating predictions for testset_{ds_name}...")
                current_test_loader = DataLoader(current_test_graphs, batch_size=self.config.BATCH_SIZE,
                                                 shuffle=False, num_workers=0)

                if self.config.CO_TEACHING:
                    test_preds1, test_preds2 = self.trainer.eval_epoch(current_test_loader, criterion,
                                                                       is_test_set_preds_only=True)
                    # For co-teaching, we can choose to use either model's predictions
                    # Here we're using model1's predictions as an example
                    test_predictions_array = test_preds1
                else:
                    test_predictions_array = self.trainer.eval_epoch(current_test_loader, criterion,
                                                                     is_test_set_preds_only=True)

                num_test_samples = len(test_predictions_array)
                if num_test_samples > 0:
                    ids = np.arange(1, num_test_samples + 1)
                    predictions_df = pd.DataFrame({'id': ids, 'pred': test_predictions_array})

                    output_predictions_filename = f'testset_{ds_name}.csv'
                    predictions_df.to_csv(output_predictions_filename, index=False)
                    print(f"  Test predictions for {ds_name} saved to {output_predictions_filename}")
            else:
                print(f"No test data found for dataset {ds_name}. Skipping predictions.")
        print("---------------------------------")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='GatedGCN Training with A,B,C,D datasets')
    parser.add_argument('--force_reprocess_data', action='store_true', help="Force re-processing of data")
    parser.add_argument('--epochs', type=int, default=ModelConfig.EPOCHS, help="Number of training epochs.")
    parser.add_argument('--no_rwse', action='store_false', dest='use_rwse_pe_cli',
                        help="Disable RWSE Positional Encoding.")
    parser.add_argument('--lr', type=float, default=ModelConfig.LEARNING_RATE, help="Learning rate.")
    parser.add_argument('--ds', type=str, default='A', choices=['A', 'B', 'C', 'D'],
                        help="Dataset to use for training (default: A).")
    parser.add_argument('--loss_fn', type=str, default='ce',
                        choices=['ce', 'gce', 'sce', 'bootstrapping', 'focal', 'mae'],
                        help="Loss function to use (default: ce).")
    parser.add_argument('--use_elr', action='store_true', default=False,
                        help="Enable Early Learning Regularization (default: False).")

    # Early stopping arguments
    parser.add_argument('--early_stopping', action='store_true', default=ModelConfig.EARLY_STOPPING,
                        help="Enable early stopping (default: False).")
    parser.add_argument('--early_stopping_patience', type=int, default=ModelConfig.EARLY_STOPPING_PATIENCE,
                        help="Number of epochs to wait before early stopping (default: 50).")
    parser.add_argument('--early_stopping_min_delta', type=float, default=ModelConfig.EARLY_STOPPING_MIN_DELTA,
                        help="Minimum change in validation accuracy to qualify as improvement (default: 0.0001).")
    parser.add_argument('--co_teaching', action='store_true', default=ModelConfig.CO_TEACHING,
                        help="Co-Teaching ( default: True )")
    # TODO co-teaching parameters and all the parameters
    # TODO add option to specify name of file
    # TODO use model of th paper ( try )
    parser.set_defaults(use_rwse_pe_cli=ModelConfig.USE_RWSE_PE)

    args = parser.parse_args()

    # Create config with CLI arguments
    config = ModelConfig()
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.USE_RWSE_PE = args.use_rwse_pe_cli
    config.PE_DIM = RWSE_MAX_K if config.USE_RWSE_PE else 0
    config.USE_ELR = args.use_elr
    config.CO_TEACHING = args.co_teaching

    # Early stopping configuration
    config.EARLY_STOPPING = args.early_stopping
    config.EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    config.EARLY_STOPPING_MIN_DELTA = args.early_stopping_min_delta

    # Create and run training system
    training_system = GatedGCNTrainingSystem(config)
    training_system.train(
        dataset_name=args.ds,
        loss_fn=args.loss_fn,
        force_reprocess=args.force_reprocess_data,
        use_elr=args.use_elr
    )


if __name__ == '__main__':
    main()
