# train_local_gatedgcn_organized.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils
from torch.nn import Linear
from torch_geometric.nn import Linear
from torch_geometric.nn import global_mean_pool


# Standard GCN Network

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


##########################################################################
