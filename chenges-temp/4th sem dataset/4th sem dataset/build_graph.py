import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def build_spatiotemporal_graph(master_csv_path, edges_csv_path, lookback_weeks=4):
    """
    Constructs PyTorch Geometric Data objects from the epidemic dataset.
    Creates a sliding window over the temporal data.
    """
    print("Loading datasets...")
    df = pd.read_csv(master_csv_path)
    edges_df = pd.read_csv(edges_csv_path)

    # 1. Define Nodes (V) mapping based on censuscode
    unique_nodes = df['censuscode'].unique()
    unique_nodes.sort()
    num_nodes = len(unique_nodes)
    
    node_mapping = {code: idx for idx, code in enumerate(unique_nodes)}
    print(f"Total Unique Nodes (Districts): {num_nodes}")

    # 2. Build Adjacency Matrix (Edges & Weights)
    edge_index = []
    edge_attr = []
    
    for _, row in edges_df.iterrows():
        src = row['source_censuscode']
        dst = row['target_censuscode']
        weight = row['shared_border_km']
        
        # Only add edge if both nodes exist in our master dataset
        if src in node_mapping and dst in node_mapping:
            edge_index.append([node_mapping[src], node_mapping[dst]])
            # Add reverse edge to make it undirected
            edge_index.append([node_mapping[dst], node_mapping[src]])
            
            # Use border length as weight
            edge_attr.extend([weight, weight])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).view(-1, 1)
    
    # Normalize Edge Weights
    edge_scaler = StandardScaler()
    edge_attr = torch.tensor(edge_scaler.fit_transform(edge_attr), dtype=torch.float32)
    print(f"Graph constructed with {edge_index.shape[1]} edges.")

    # 3. Sort data chronologically to maintain temporal integrity
    df = df.sort_values(by=['iso_year', 'iso_week'])
    
    # Fill missing values (like LAI or lagged cases for early weeks)
    df.fillna(0, inplace=True)

    # Feature selection
    dynamic_features = ['temp_k', 'preci_mm', 'LAI', 'cases_lag1', 'cases_lag2', 'cases_lag3']
    static_features = ['population_2024', 'pop_density_per_km2_2024']
    all_features = dynamic_features + static_features

    # Normalize node features
    scaler = StandardScaler()
    df[all_features] = scaler.fit_transform(df[all_features])

    # 4. Group data by time step (Year + Week)
    time_steps = df[['iso_year', 'iso_week']].drop_duplicates().values
    
    X_all_time = []
    Y_reg_all_time = []
    Y_clf_all_time = []

    for year, week in time_steps:
        # Get data for this specific week
        week_data = df[(df['iso_year'] == year) & (df['iso_week'] == week)]
        
        # Initialize node matrices for this timestep (zeros if district didn't report)
        x_t = np.zeros((num_nodes, len(all_features)))
        y_reg_t = np.zeros((num_nodes, 1))
        y_clf_t = np.zeros((num_nodes, 1))
        
        for _, row in week_data.iterrows():
            idx = node_mapping[row['censuscode']]
            x_t[idx] = row[all_features].values
            y_reg_t[idx] = row['cases']
            y_clf_t[idx] = row['is_outbreak']
            
        X_all_time.append(x_t)
        Y_reg_all_time.append(y_reg_t)
        Y_clf_all_time.append(y_clf_t)

    X_all_time = torch.tensor(np.array(X_all_time), dtype=torch.float32) # Shape: (Time, Nodes, Features)
    Y_reg_all_time = torch.tensor(np.array(Y_reg_all_time), dtype=torch.float32)
    Y_clf_all_time = torch.tensor(np.array(Y_clf_all_time), dtype=torch.float32)

    # 5. Create Sliding Windows
    num_timesteps = X_all_time.shape[0]
    dataset = []

    print(f"Creating sliding windows with lookback = {lookback_weeks} weeks...")
    for t in range(num_timesteps - lookback_weeks):
        # Input features: [t, t+lookback)
        X_window = X_all_time[t : t + lookback_weeks] # Shape: (Seq_Len, Nodes, Features)
        
        # Reshape to (Nodes, Seq_Len, Features) for standard DGAT input
        X_window = X_window.permute(1, 0, 2)
        
        # Targets: values at t + lookback (the week we are predicting)
        Y_reg_target = Y_reg_all_time[t + lookback_weeks]
        Y_clf_target = Y_clf_all_time[t + lookback_weeks]
        
        data = Data(x=X_window, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr,
                    y_reg=Y_reg_target, 
                    y_clf=Y_clf_target)
        dataset.append(data)

    print(f"Successfully created {len(dataset)} Spatio-Temporal Graph samples!")
    print(f"Example Shape -> X: {dataset[0].x.shape}, Y_reg: {dataset[0].y_reg.shape}, Y_clf: {dataset[0].y_clf.shape}")
    
    return dataset

if __name__ == "__main__":
    MASTER_CSV = "final_datasets/master_dataset.csv"
    EDGES_CSV = "final_datasets/graph_edges.csv"
    
    graphs = build_spatiotemporal_graph(MASTER_CSV, EDGES_CSV, lookback_weeks=4)
