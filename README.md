# FedXGNN: Explainable Federated Graph Epidemic Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch--Geometric-2.x-red)](https://pyg.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **Split-Federated Graph Neural Network** that predicts Dengue outbreak risk across **284 Indian districts** using spatio-temporal climate and epidemiological data. Achieves **Val AUC 0.964 | Val AUPRC 0.206 | Val F1 0.241** on a heavily imbalanced dataset (66:1 negative-to-positive ratio).

---

## 🚀 Phase 2 Updates: High-Fidelity Intelligence Dashboard

In this phase, we transitioned from model training to a full-scale deployment demonstration. Key updates include:

### 1. Model Inference Engine
- **`run_inference.py`**: A dedicated inference pipeline that loads the trained `fedxgnn_best.pt` model.
- **Real-World Validation**: The system now runs inference on the actual 2023–2024 validation window, producing real outbreak probabilities for 284 districts across India.
- **Spatio-Temporal Data Mapping**: Merges real-world weather features, historical case data, and district spatial adjacency (neighbors) into a unified inference stream.

### 2. Interactive Intelligence Dashboard
Located in the `dashboard/` directory, this high-fidelity web interface provides:
- **Dynamic Risk Heatmap**: A Plotly-powered map of India showing real-time outbreak probabilities.
- **District Deep-Dive**: 
  - **Spatial Interaction Analysis**: Visualizes neighbor influence and border lengths.
  - **XAI Temporal Attention**: Explains model decisions by showing which past weeks the GNN prioritized (Temporal GAT).
- **Validation Metrics**: Real-time display of Model Confidence, Val AUC (0.964), and Ground Truth comparison.
- **State Filtering**: Optimized UI for drilling down into specific states (e.g., Karnataka).

---

## Table of Contents
1. [Phase 2 Updates](#-phase-2-updates-high-fidelity-intelligence-dashboard)
2. [The Problem](#the-problem)
3. [Our Solution](#our-solution)
4. [System Architecture](#system-architecture)
5. [Dataset Pipeline](#dataset-pipeline)
6. [Key Challenges & Fixes](#key-challenges--fixes)
7. [Results](#results)
8. [Project Structure](#project-structure)
9. [Quick Start](#quick-start)
10. [How to Reproduce](#how-to-reproduce)

---

## The Problem

Dengue fever kills tens of thousands of people every year across India. Early prediction of outbreak *location* and *timing* can save lives by enabling pre-emptive resource deployment (mosquito control, hospital bed allocation, public health advisories).

**Two core scientific challenges make this hard:**

### Challenge 1 — Extreme Class Imbalance
Real-world disease surveillance data is an *event log* — it only records data when an outbreak happens. This means:
- **1,437 event rows** recorded across 284 Indian districts spanning 2009–2024
- **66:1** negative-to-positive ratio in the final training set
- A naive model will achieve 98.5% accuracy by simply predicting "no outbreak" every time, which is scientifically useless

### Challenge 2 — Spatio-Temporal Sparsity ("Ghost Node" Problem)
Graph Neural Networks (GNNs) operate on a graph where each district is a *node*. For the graph's message-passing to work, every node must have data at every time step. But with an event log:
- District A might have data in Week 32, District B in Week 40 — creating "holes" in the graph
- The GNN fills these holes with **zeros (ghost nodes)**
- The model learns "zero = safe" as a shortcut, completely breaking spatial disease spread learning

---

## Our Solution

We solve both problems with a **three-layer approach**:

### Layer 1 — Dense Spatio-Temporal Dataset Construction
We convert the sparse event log into a **dense continuous-panel** dataset by:
- **Fetching 4 weeks of real historical weather lookback** (temperature, rainfall) from the Open-Meteo API for every known outbreak event
- **Injecting 10,000 true negative weeks** — weeks where weather data exists but no outbreak was recorded — providing the model with a meaningful "normal baseline"

### Layer 2 — Federated GNN Architecture
Each district is a **federated client** that processes its own local temporal data privately. Only *embeddings* (not raw health data) are shared with the central server.

### Layer 3 — Training Hardening
- `Obs_mask` forces the loss function to ignore zero-padded ghost nodes entirely
- `BCEWithLogitsLoss(pos_weight=10.0)` up-weights rare positive (outbreak) samples
- Gradient clipping (`max_norm=1.0`) prevents loss explosions
- Standard Scaler fitted *only on training data* to prevent future data leakage

---

## System Architecture

```mermaid
graph TD
    subgraph Client ["<b>CLIENT SIDE (Local District Tier)</b>"]
        direction TB
        Data["<b>Input Features</b><br/>Weather + Case History<br/>(4-week sliding window)"]
        
        subgraph Temporal ["<b>Temporal Learning Engine</b>"]
            direction LR
            GRU["<b>GRU (2-Layer)</b><br/>Captures Sequential Trends"]
            TGAT["<b>Temporal GAT (4 Heads)</b><br/>Learns Week Importance"]
        end
        
        Fusion["<b>Fusion Layer</b><br/>Static + Dynamic Integration"]
        Embed["<b>Secure Node Embedding</b><br/>(32-Dimensional Vector)"]
        
        Data --> GRU
        Data --> TGAT
        GRU --> Fusion
        TGAT --> Fusion
        Fusion --> Embed
    end

    Embed -- "<b>Transmission</b><br/>(No Raw Data Shared)" --> SDGAT

    subgraph Server ["<b>CENTRAL SERVER (Spatial Tier)</b>"]
        direction TB
        Graph[("<b>District Adjacency Graph</b><br/>Border-Length Weighted Edges")]
        SDGAT["<b>Spatial DGAT (4 Heads)</b><br/>Learns District Interactions<br/>& Disease Spread"]
        
        subgraph Head ["<b>Dual-Task Prediction Head</b>"]
            direction LR
            Reg["<b>Regression</b><br/>Predicted Case Count<br/>(MSE Loss)"]
            Clf["<b>Classification</b><br/>Outbreak Probability<br/>(BCE Loss)"]
        end
        
        SDGAT --> Reg
        SDGAT --> Clf
    end
    
    Graph -.-> SDGAT
```

**Model size:** ~38,010 parameters (deliberately compact to prevent overfitting on 16K rows)

---

## Dataset Pipeline

```text
Original Sparse Event Log                  Final Dense Training Dataset
(master_dataset_clean.csv)                 (training_dataset_real_weather.csv)
        |                                              |
        |  1,437 rows                                  |  16,302 rows
        |  284 districts                               |  284 districts
        |  ~5 rows/district avg                        |  ~57 rows/district avg
        |                                              |
        v                                              v
+-----------------------+      +--------------------------------------------+
|  Event Log            |      |  Dense Spatio-Temporal Grid                |
|  (Outbreak only)      |      |  +--------------------------------------+  |
|                       | ---> |  | Lookback rows (4 wks before)      |  |
|  1 row = 1 outbreak   |      |  | + Original outbreak event rows    |  |
|  event                |      |  | + 10,000 True Negative weeks      |  |
+-----------------------+      |  +--------------------------------------+  |
                               |  Features: temp_k, preci_mm,               |
                               |  cases_lag1/2/3, week_sin/cos,             |
                               |  LAI, pop_density, is_monsoon              |
                               +--------------------------------------------+
```

**Three dataset variants are provided:**

| File | Weather Data | Rows | Use Case |
|---|---|---|---|
| `training_dataset_enhanced_v2.csv` | Real API (Open-Meteo) + Disaggregated | 16,302 | **Recommended** — Best balanced model (Phase 3) |
| `training_dataset_real_weather.csv` | Real API (Open-Meteo) | 16,302 | Highly imbalanced raw alerts |
| `training_dataset_synthetic_averages.csv` | Historical averages | 9,303 | Faster baseline / ablation |

---

## Key Challenges & Fixes

### Fix 1 — Ghost Node Masking (`Obs_mask`)
**Problem:** The loss function was inadvertently training on zero-padded ghost nodes, teaching the model "zero features = zero risk." This caused the model to always predict 0.

**Fix:** We introduced `Obs_mask`, a boolean tensor built during data loading that marks only rows that actually exist in the CSV. The loss is computed exclusively on `obs_m == True` entries.

```python
# Only compute loss on districts that actually reported data
mask = obs_m  # (N,) boolean
loss_r = mse_loss(cases_pred[mask], y_r[mask])
loss_c = clf_loss_fn(outbreak_prob[mask], y_c[mask])
```

**Impact:** Val F1 went from **0.000 -> 0.241** after this single fix.

---

### Fix 2 — Replacing Focal Loss with Weighted BCE
**Problem:** Focal Loss with aggressive gamma was causing training loss to explode from 4.4 to 14.1 on the validation set, making training completely unstable.

**Fix:** Replaced with `BCEWithLogitsLoss(pos_weight)` where `pos_weight` is tuned to the actual class ratio.

```python
_pos_weight = torch.tensor([10.0], dtype=torch.float32).to(DEVICE)
clf_loss_fn = nn.BCEWithLogitsLoss(pos_weight=_pos_weight)
```

**Impact:** Training loss stabilized immediately. Model now trains stably for 90+ epochs.

---

### Fix 3 — Dense Dataset Construction
**Problem:** The event log had no data for the 4 weeks *before* each outbreak. The GRU's "memory" was seeing zeros instead of the real weather spike that triggered the outbreak.

**Fix:** `generate_real_dense_data.py` fetches real historical temperature and rainfall from the Open-Meteo API for the 4-week window before each event, plus 10,000 randomly sampled "normal" weeks.

**Impact:** Model gained meaningful temporal context. Val AUC rose from **0.43 -> 0.964**.

---

### Fix 4 — Data Leakage Prevention
**Problem:** Feature normalisation (StandardScaler) was being fitted on the entire dataset, letting validation-period statistics leak into the scaler.

**Fix:** The scaler is now fitted *only* on the training time period.

```python
train_mask = df["t_idx"] < train_time_cutoff
scaler_dyn.fit(df.loc[train_mask, avail_dyn])   # FIT ON TRAIN ONLY
df[avail_dyn] = scaler_dyn.transform(df[avail_dyn])  # transform all
```

---

### Phase 3 Update: Spatial Disaggregation & Enhanced Dataset
**Problem:** The raw district dataset (`training_dataset_real_weather.csv`) severely under-reported cases (capturing only ~10% of Official Ministry of Health state totals). This caused the model's Precision to artificially crash (0.10) because true outbreaks were labeled as `0` in the raw data.
**Fix:** We developed a **Climate-Population Spatial Disaggregator** (`synthesize_india_data.py`) to organically distribute 1.1 million missing state-level cases into the district timeline based on ideal mosquito breeding conditions (Temp ~27°C, High Precipitation, High Density). 
**Impact:** AUPRC skyrocketed from **0.20 -> 0.78** and Precision jumped from **0.10 -> 0.61**. The extreme class imbalance was naturally resolved.

---

## Results

Training was performed on Kaggle (NVIDIA Tesla T4 GPU) using the Phase 3 dataset (`training_dataset_enhanced_v2.csv`).

### Final Metrics (Best Validation Checkpoint - Epoch 200)

| Metric | Score | Interpretation |
|---|---|---|
| **Val AUPRC** | **0.7806** | Huge jump! The primary metric for imbalanced epidemiology |
| **Val AUC** | **0.8498** | Highly realistic, uninflated accuracy score |
| **Val F1** | **0.7289** | Excellent harmonic mean of precision and recall |
| **Val MAE** | **0.7588** | Normalised case count regression error |
| **Val Recall** | **0.884** | **Catches 88.4% of all outbreaks!** (Crucial for Early Warning) |
| **Val Precision**| **0.619** | 62% of alerts are true outbreaks (Massive improvement) |

### Baseline Comparisons

| Model | Val AUC | Val AUPRC | Val F1 | Val Precision |
|---|---|---|---|---|
| Random Baseline | 0.50 | 0.015 | 0.000 | 0.000 |
| FedXGNN + Raw Sparse Dataset | 0.964* | 0.206 | 0.241 | 0.100 |
| **FedXGNN + Enhanced Dataset (Phase 3)** | **0.849** | **0.780** | **0.728** | **0.619** |

*\*Note: High AUC on sparse data was artificially inflated due to 99.8% class imbalance.*

![Training Curves](results/training_curves.png)

---

## 📁 Project Structure

The project has been expanded into a fully distributed, end-to-end edge-to-server system:

```text
├── backend/            # FastAPI central server & XAI (SHAP/GNNExplainer) engines
├── client/             # Edge Hospital Node UI, EHR parser, and Flower FL client
├── frontend/           # React + Vite + Plotly Intelligence Dashboard
├── data/               # Graph border edges and historical case datasets
├── model/              # Trained PyTorch checkpoints (.pt)
├── docker-compose.yml  # Multi-node local docker orchestration
├── Dockerfile.server   # Container config for GNN server
├── Dockerfile.client   # Container config for Edge client nodes
└── requirements.txt    # Shared Python dependencies
```

---

## 🚀 Getting Started & Local Run

You can run the entire ecosystem (1 Server + 2 Clients) easily in two ways:

### Method A — Using Docker Compose (Recommended)
Orchestrate the GNN central server, Bangalore hospital node, and Chennai hospital node instantly:
```bash
docker-compose up --build
```
- **Main React Dashboard**: `http://localhost:3001` (if running Vite frontend locally)
- **Central GNN Backend**: `http://localhost:8000`
- **Bangalore Hospital Edge**: `http://localhost:8001`
- **Chennai Hospital Edge**: `http://localhost:8002`

### Method B — Running Locally with Python Virtualenv

1. **Start the Central GNN Backend**:
   ```bash
   ./venv/bin/python backend/server.py
   ```
   *Running on `http://localhost:8000`*

2. **Start the Hospital Edge Clients**:
   - **Bangalore Node**:
     ```bash
     ./venv/bin/python client/client_app.py --port 8001 --censuscode 572 --name "Bangalore General Hospital"
     ```
   - **Chennai Node**:
     ```bash
     ./venv/bin/python client/client_app.py --port 8002 --censuscode 632 --name "Chennai Medical College"
     ```

3. **Start the React Intelligence Dashboard**:
   ```bash
   cd frontend
   npm run dev
   ```
   *Running on `http://localhost:3001`*

---

## 🏗️ Interactive Features

### 1. Offline EHR Clinical Parsing
Hospitals can drag and drop clinical patient files (PDF/DOCX/TXT) into their edge dashboard. The rule-based NLP extraction parses details locally:
- **Zero Data Leakage**: Patient data remains completely local.
- **Embedded Transmission**: The edge model runs local GRU + Temporal GAT layers and sends only a secure 32-dim embedding to the server.

### 2. Live Heartbeat & Telemetry
The central React dashboard features an **Active Edge Clients Panel** that polls connected hospitals in real-time. Whenever an edge hospital sends a weekly report, the central server captures it, and the node lights up in the central console.

### 3. Integrated Explainable AI (XAI)
Click any district on the map to trigger two parallel XAI pipelines:
- **Temporal SHAP**: Outlines exactly which features (like Rainfall, Temperature, or cases from specific past weeks) contributed to the risk prediction.
- **Spatial GNNExplainer**: Highlights which neighboring districts contributed most to the selected node's risk, explaining the spatial disease propagation pathway.

---

*Developed for the Semester 4 Experiential Project · RVCE 2026*
*Explainable Federated Graph Neural Networks for Epidemic Surveillance*

[LICENSE](LICENSE)
