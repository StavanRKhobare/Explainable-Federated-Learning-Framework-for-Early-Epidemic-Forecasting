# Split-Federated DGAT Epidemic Forecasting Implementation

This document outlines the architecture and implementation steps for building the privacy-preserving, federated Spatio-Temporal Graph Attention Network (Split-FedSTGNN) to forecast epidemic case counts and outbreak probabilities.

## Background Context
The system treats each District as a Hospital (Client) in a federated learning network. 
- **Clients** run a local Temporal Graph + GRU on their own sequential weather, demographic, and historical case data to capture the biological incubation delays and long-term seasonal trends of epidemics. They produce a "Local Temporal Embedding" at the end of each week.
- **Central Hub (Server)** receives these embeddings without seeing raw patient counts (preserving privacy). It runs a Spatial DGAT over the country-wide graph (where nodes are districts and edges are shared borders) to allow neighboring districts to influence each other's risk profiles, acting as an "Early Warning Radar".

## User Review Required

> [!IMPORTANT]
> **Dataset vs Current Code Discrepancy:** The current `build_graph.py` relies on columns like `censuscode`, `iso_year`, and `iso_week`. However, the final `master_dataset.csv` uses `state_ut`, `district`, `year`, `mon`, and `week_of_outbreak`. `build_graph.py` will be completely rewritten to align with the final dataset documentation.

> [!NOTE]
> **Spatial Graph Training vs Message Passing:** You mentioned: *"I dont think there will be any training for the spatial graph, just embeddings that will be influenced by neighbouring nodes"*. 
> In a Graph Attention Network (GAT), the *structure* (edges/borders) is static and not trained, but the GAT *weights* (which determine *how much* attention to pay to a specific neighbor's embedding) are typically learned. If we strictly want no learned parameters on the server, we can use a simpler Graph Convolutional Network (GCN) or a static message-passing aggregation (like standard FedAvg spatial smoothing) instead of GAT. *For this plan, I assume the Server learns the Spatial Attention weights to optimally combine neighbor embeddings before returning them to the local node for the final prediction.*

## Open Questions

> [!WARNING]
> Please clarify the following before we begin implementation:
> 1. **Disease Filtering:** The dataset contains a `Disease` column (e.g., Dengue, Malaria). Should the model be trained on a specific disease (e.g., filter only for Dengue), or should we train a unified model where all diseases are aggregated? (Usually, epidemic dynamics differ wildly between diseases).
> 2. **Target Window:** How many weeks ahead are we predicting? (e.g., using data from Weeks 1-4 to predict Week 5). What is our sliding window `T`?
> 3. **Spatial Graph Edges:** Should the edge weights be derived solely from `border_length` (from `graph_edges.csv`), or should we also factor in the geographical distance between the `Latitude`/`Longitude` centroids?

## Feature Selection

Based on the `dataset_documentation.md`, here is the breakdown of what features are useful and what will be discarded:

### Useful Features (To be used for training)
*   **Dynamic / Temporal Features (Input to Local GRU/Temporal Graph):**
    *   `Cases` (Target to predict, but past weeks will be used as Autoregressive Lag features).
    *   `preci` (Precipitation): Crucial for vector-borne / water-borne diseases.
    *   `Temp` (Temperature): Defines the survival rate of vectors (mosquitoes).
    *   `LAI` (Leaf Area Index): Vegetation density heavily impacts epidemics.
*   **Static Features (Node properties):**
    *   `density`: Population density determines transmission speed.
    *   `pop_2024`: Total susceptible population.
*   **Target Variables (Dual-Task):**
    *   `Cases` (Regression task).
    *   `is_outbreak` (Classification task - Epidemic Prediction Score).

### Discarded / Redundant Features
*   `Unnamed: 0`: Artifact of pandas merge.
*   `pop_2025`: Redundant with `pop_2024`.
*   `area`: Redundant, already captured by the relationship between `pop_2024` and `density`.
*   `Deaths`: Lagging indicator. Predicting cases and outbreaks provides earlier warnings.
*   `day`: Irrelevant since aggregation and feature collection happen at the end of the *week*.

## Proposed Changes

We will implement this in distinct, modular stages.

### 1. Data Processing Pipeline
#### [MODIFY] [build_graph.py](file:///home/stavan-khobare/Desktop/4th%20sem%20dataset/build_graph.py)
*   Rewrite to parse `master_dataset.csv` using the correct columns (`district`, `year`, `week_of_outbreak`).
*   Generate District-to-Index mappings using `region_mapping.csv`.
*   Implement a sliding window mechanism that processes $T$ weeks of historical features (Cases, Temp, Preci, LAI) and static features (Density) per district.
*   Construct the static Spatial Adjacency Matrix using `graph_edges.csv`.

### 2. Neural Architecture Core
#### [NEW] `models/client_temporal.py`
*   Contains the **Temporal Attention GRU (TA-GRU)**.
*   **Input**: Week-wise sequence of local features.
*   **Output**: Local Temporal Embedding (e.g., Dim 64).

#### [NEW] `models/server_spatial.py`
*   Contains the **Spatial DGAT** layer.
*   **Input**: All 640+ Client Embeddings + Static Graph Edges.
*   **Output**: Spatially refined embeddings.

#### [NEW] `models/dual_task_head.py`
*   Contains the final prediction MLP.
*   **Input**: Fused (Local + Spatial) Embedding.
*   **Output**: `Predicted_Cases` (Regression) and `Outbreak_Prob` (Classification).

### 3. Federated Learning Simulator
#### [NEW] `federated_training.py`
*   A PyTorch simulation of the Split-Federated process.
*   **Forward Pass:** Clients process sequences -> Server runs Spatial GAT -> Clients run final prediction.
*   **Backward Pass:** Loss is computed locally based on dual-task targets -> gradients flow from Client prediction head to Server Spatial GAT, and back down to Client Temporal GRU.

## Verification Plan

### Automated Tests
*   Run `build_graph.py` to ensure the sliding window outputs the correct tensor dimensions: `[Batch, Time_Steps, Features]`.
*   Test a single forward pass of a mock dataset through `client_temporal.py` -> `server_spatial.py` -> `dual_task_head.py` to verify tensor shape compatibility.
*   Confirm that the Spatial Graph properly masks out non-neighboring districts using the adjacency matrix.

### Manual Verification
*   Train the federated setup for 5 epochs on a subset of the data and verify that both Regression Loss (MSE for cases) and Classification Loss (BCE for `is_outbreak`) are successfully decreasing without the server ever seeing the raw `Cases` input directly.
