import torch
import numpy as np
import shap
from torch_geometric.explain import Explainer, GNNExplainer

class XAIEngine:
    def __init__(self, model, device=torch.device("cpu")):
        self.model = model
        self.device = device

    def explain_local_temporal(self, x_dyn_node, x_stat_node):
        """
        Compute SHAP values for the ClientTemporalModel of a single district.
        x_dyn_node: (1, 4, 9) tensor of scaled dynamic features.
        x_stat_node: (1, 2) tensor of static features.
        """
        self.model.eval()
        client_model = self.model.client
        
        # We want to explain the embedding output or a simplified version.
        # Since SHAP works best with flat inputs, we define a wrapper function.
        def wrapper(x_flat):
            # x_flat is (B, 36) -> reshape to (B, 4, 9)
            x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=self.device)
            B = x_tensor.shape[0]
            x_reshaped = x_tensor.view(B, 4, 9)
            # Replicate static features for batch
            stat_rep = x_stat_node.repeat(B, 1).to(self.device)
            emb = client_model(x_reshaped, stat_rep)
            # Return mean embedding value as target
            return emb.mean(dim=-1).detach().cpu().numpy()

        # Reshape input to flat array for SHAP
        flat_input = x_dyn_node.view(1, -1).cpu().numpy()
        
        # Build background dataset (zeros or slightly perturbed)
        background = np.zeros((5, 36))
        
        explainer = shap.KernelExplainer(wrapper, background)
        shap_values = explainer.shap_values(flat_input)
        
        # Reshape SHAP values back to (4, 9)
        shap_reshaped = shap_values[0].reshape(4, 9)
        return shap_reshaped

    def explain_spatial_gnn(self, x_d, x_s, edge_index, edge_attr, target_node_idx):
        """
        Use PyG GNNExplainer to explain the Spatial GAT's prediction for a target district.
        """
        self.model.eval()
        
        # Define a GNN wrapper model that takes node embeddings and produces classification logits
        class SpatialGNNWrapper(torch.nn.Module):
            def __init__(self, server_model, head_model, edge_index, edge_attr):
                super().__init__()
                self.server = server_model
                self.head = head_model
                self.edge_index = edge_index
                self.edge_attr = edge_attr
                
            def forward(self, local_emb):
                spatial_emb = self.server(local_emb, self.edge_index, self.edge_attr)
                fused = torch.cat([local_emb, spatial_emb], dim=-1)
                _, logit = self.head(fused)
                return logit

        # Get current client embeddings for all nodes
        with torch.no_grad():
            local_emb = self.model.client(x_d.to(self.device), x_s.to(self.device))
            
        wrapper_model = SpatialGNNWrapper(self.model.server, self.model.head, edge_index.to(self.device), edge_attr.to(self.device))
        
        # Configure Explainer
        explainer = Explainer(
            model=wrapper_model,
            algorithm=GNNExplainer(epochs=50), # fast explanation for live demo
            explanation_type='model',
            node_mask_type=None,
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='node',
                return_type='raw'
            )
        )
        
        # Run explanation
        explanation = explainer(
            x=local_emb,
            edge_index=edge_index.to(self.device),
            target=torch.tensor([target_node_idx], device=self.device)
        )
        
        # Extract edge importances
        edge_mask = explanation.edge_mask.detach().cpu().numpy()
        return edge_mask
