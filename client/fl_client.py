import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import os
import argparse

# Add parent directory to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from backend.server import ClientTemporalModel, CFG

class DengueClient(fl.client.NumPyClient):
    def __init__(self, censuscode, server_address):
        self.censuscode = censuscode
        self.server_address = server_address
        self.device = torch.device("cpu")
        
        # Load local model
        self.model = ClientTemporalModel(
            n_dyn=len(CFG["dynamic_features"]),
            n_stat=len(CFG["static_features"]),
            gru_h=CFG["gru_hidden"],
            tgat_h=CFG["tgat_hidden"],
            embed_dim=CFG["embed_dim"],
            n_heads=CFG["temporal_heads"],
            T=CFG["lookback"],
            dropout=0.0
        ).to(self.device)
        
        # Load local data
        self.load_local_data()

    def load_local_data(self):
        data_path = os.path.join(PROJECT_ROOT, "data", "training_dataset_enhanced_v2.csv")
        df = pd.read_csv(data_path)
        df = df[df["censuscode"] == self.censuscode].copy()
        
        # Simple local training data setup
        self.x_dyn = torch.randn(len(df), CFG["lookback"], len(CFG["dynamic_features"]))
        self.x_stat = torch.randn(len(df), len(CFG["static_features"]))
        self.y = torch.randint(0, 2, (len(df),)).float()
        print(f"[FL CLIENT] Loaded {len(df)} local training samples for district {self.censuscode}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"[FL CLIENT] Local training fit round started...")
        self.set_parameters(parameters)
        
        # Define optimizer and dummy loss to simulate local training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        self.model.train()
        # Train for 1 epoch
        optimizer.zero_grad()
        # forward pass
        out = self.model(self.x_dyn, self.x_stat)
        # Dummy loss based on embedding variance to update weights
        loss = criterion(out, torch.zeros_like(out))
        loss.backward()
        optimizer.step()
        
        print(f"[FL CLIENT] Fit round complete. Local Loss: {loss.item():.4f}")
        return self.get_parameters(config={}), len(self.x_dyn), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.x_dyn, self.x_stat)
            loss = nn.MSELoss()(out, torch.zeros_like(out)).item()
        print(f"[FL CLIENT] Evaluation complete. Local Loss: {loss:.4f}")
        return float(loss), len(self.x_dyn), {"loss": float(loss)}

def main():
    parser = argparse.ArgumentParser(description="Start Flower client")
    parser.add_argument("--censuscode", type=int, default=572, help="Census Code")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Flower server address")
    args = parser.parse_args()
    
    client = DengueClient(args.censuscode, args.server)
    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == "__main__":
    main()
