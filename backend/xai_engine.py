import torch
import numpy as np
import shap
from torch_geometric.explain import Explainer, GNNExplainer

class XAIEngine:
    def __init__(self, model, device=torch.device("cpu")):
        self.model = model
        self.device = device

    def explain_local_temporal(self, x_dyn_node, x_stat_node, n_idx, windows):
        """
        Compute SHAP values for the ClientTemporalModel of a single district.
        x_dyn_node: (1, 4, N_DYN) tensor of scaled dynamic features.
        x_stat_node: (1, N_STAT) tensor of static features.
        n_idx: index of the target node in the spatial graph.
        windows: global list of windows for background dataset construction.
        
        Returns signed SHAP values (shape: num_weeks x num_features).
        Positive values = feature INCREASES outbreak risk (green).
        Negative values = feature DECREASES outbreak risk (red).
        """
        self.model.eval()
        client_model = self.model.client
        
        num_weeks = x_dyn_node.shape[1]
        num_features = x_dyn_node.shape[2]
        flat_size = num_weeks * num_features

        def wrapper(x_flat):
            """Predict outbreak probability from flat dynamic feature input."""
            x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=self.device)
            B = x_tensor.shape[0]
            x_reshaped = x_tensor.reshape(B, num_weeks, num_features)
            stat_rep = x_stat_node.repeat(B, 1).to(self.device)
            with torch.no_grad():
                emb = client_model(x_reshaped, stat_rep)  # (B, embed_dim)
                # Project through head trunk to get classification logit
                h = self.model.head.trunk(torch.cat([emb, emb], dim=-1))
                logit = self.model.head.clf_head(h).squeeze(-1)  # (B,)
            return logit.detach().cpu().numpy()

        # Reshape input to flat array for SHAP
        flat_input = x_dyn_node.reshape(1, -1).cpu().numpy()
        
        # Build background dataset from actual historical windows of the target node
        num_bg_samples = min(15, len(windows))
        bg_indices = np.linspace(0, len(windows) - 1, num_bg_samples, dtype=int)
        background_list = []
        for idx in bg_indices:
            x_w, _, _, _, _ = windows[idx]
            bg_x = x_w[n_idx].cpu().numpy().reshape(-1)
            background_list.append(bg_x)
        background = np.array(background_list)

        explainer = shap.KernelExplainer(wrapper, background)
        shap_values = explainer.shap_values(flat_input, nsamples=50, silent=True)
        
        # Reshape SHAP values back to (num_weeks, num_features)
        if isinstance(shap_values, list):
            shap_arr = np.array(shap_values[0])
        else:
            shap_arr = np.array(shap_values)
        shap_reshaped = shap_arr.reshape(num_weeks, num_features)
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
                
            def forward(self, local_emb, edge_index=None):
                # Use provided edge_index if available (Explainer passes it)
                curr_edge_index = edge_index if edge_index is not None else self.edge_index
                spatial_emb = self.server(local_emb, curr_edge_index, self.edge_attr)
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
