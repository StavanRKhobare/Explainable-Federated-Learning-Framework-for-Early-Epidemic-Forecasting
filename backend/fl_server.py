import flwr as fl
import sys
import os

# Add parent directory to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def main():
    print("[FL SERVER] Starting Flower Federated Learning Server...")
    
    # Define FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,          # Train on all available clients
        fraction_evaluate=1.0,     # Evaluate on all clients
        min_fit_clients=2,         # Need at least 2 clients to start a round
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
