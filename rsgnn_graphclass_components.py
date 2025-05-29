# rsgnn_graphclass_components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils

class LinkPredictorMLP(nn.Module):
    def __init__(self, node_emb_dim, mlp_hidden_dim, t_small=0.05):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_emb_dim * 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1) 
        )
        self.t_small = t_small

    def forward(self, node_embeddings, edge_index):
        row, col = edge_index
        if edge_index.numel() > 0: 
            if row.max() >= node_embeddings.size(0) or col.max() >= node_embeddings.size(0):
                raise ValueError(f"Edge index out of bounds for node_embeddings. Max index: {max(row.max(), col.max())}, Num nodes: {node_embeddings.size(0)}")

            emb_row = node_embeddings[row]
            emb_col = node_embeddings[col]

            edge_node_pair_embeddings = torch.cat([emb_row, emb_col], dim=-1)
            raw_weights = self.mlp(edge_node_pair_embeddings).squeeze(-1)
            
            # Output of ReLU (this tensor should not be modified inplace if used in graph)
            weights_after_relu = F.relu(raw_weights) 
            
            # Create a new tensor for thresholding if t_small > 0
            if self.t_small > 0:
                # Option 1: Using torch.where (cleaner)
                estimated_weights_processed = torch.where(
                    weights_after_relu < self.t_small, 
                    torch.zeros_like(weights_after_relu), 
                    weights_after_relu
                )
                # Option 2: Cloning then modifying (also works)
                # estimated_weights_processed = weights_after_relu.clone()
                # estimated_weights_processed[estimated_weights_processed < self.t_small] = 0.0
            else:
                estimated_weights_processed = weights_after_relu
        else: 
            estimated_weights_processed = torch.empty(0, dtype=node_embeddings.dtype, device=node_embeddings.device)
        
        # This 'estimated_weights_processed' is what should be used by the GNN if thresholding is applied.
        # The LinkPredictor itself, when its weights are used to compute loss_gnn_for_lp,
        # should provide these processed weights.
        # For the reconstruction loss component that directly uses the LP's output (pred_weights_pos/neg),
        # it should ideally use 'weights_after_relu' *before* thresholding, as the thresholding
        # is a post-processing step for the GNN, not inherent to the LP's direct output target (0 or 1).
        # However, RS-GNN paper implies the GNN uses the thresholded weights.

        # Let's return the processed (thresholded) weights as the primary output
        # The training loop must be careful about which version of weights it uses for reconstruction loss
        # vs. GNN input.
        # For now, LinkPredictor returns the final processed weights.
        return estimated_weights_processed

# ... ReconstructionLoss remains the same ...
class ReconstructionLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.mse_loss_none = nn.MSELoss(reduction='none')

    def calculate_weighted_mse(self, predicted_weights, target_weights, 
                               edge_index_for_similarity, node_features_for_similarity):
        if edge_index_for_similarity.numel() == 0 or predicted_weights.numel() == 0:
            return torch.tensor(0.0, device=predicted_weights.device)
        row, col = edge_index_for_similarity
        if row.max() >= node_features_for_similarity.size(0) or col.max() >= node_features_for_similarity.size(0):
            raise ValueError(f"Edge index for similarity calculation is out of bounds for node_features_for_similarity. Max index: {max(row.max(), col.max())}, Num node features: {node_features_for_similarity.size(0)}")
        feat_diff = node_features_for_similarity[row] - node_features_for_similarity[col]
        feat_dist_sq = torch.sum(feat_diff**2, dim=1)
        if self.sigma > 1e-7:
            if torch.all(target_weights == 1): 
                 similarity_factor = torch.exp(-feat_dist_sq / (self.sigma**2))
            elif torch.all(target_weights == 0): 
                 similarity_factor = torch.exp(feat_dist_sq / (self.sigma**2)) 
            else: 
                 print("Warning: Mixed target_weights in calculate_weighted_mse. Defaulting similarity_factor to 1.")
                 similarity_factor = torch.ones_like(feat_dist_sq, device=predicted_weights.device)
        else: 
            similarity_factor = torch.ones_like(feat_dist_sq, device=predicted_weights.device)
        loss_values = self.mse_loss_none(predicted_weights, target_weights) 
        weighted_loss = (similarity_factor * loss_values).mean()
        return weighted_loss