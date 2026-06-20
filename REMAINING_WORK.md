# EpiGraph AI — Remaining Work Tracker
> Living document. Update checkboxes as items are completed.  
> Based on: Full Repository Audit Report (2026-06-20)  
> Repo: Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting

---

## ✅ Completed

- [x] `/api/timeline` crash bug fixed — `truths` → `truths_c` in `backend/server.py` L471
- [x] `run_inference.py` dataset mismatch fixed — now uses `training_dataset_enhanced_v2.csv`
- [x] `train_fedxgnn_run.py` — `metrics.json` logging added (val AUC, AUPRC, F1, MAE, full epoch history)
- [x] Data pipeline — 16,302 rows, 284 districts, 2009–2022, real district names + lat/lon
- [x] Spatial graph — DataMeet GeoJSON → `graph_edges.csv` with shared border lengths
- [x] GRU + Temporal GAT client encoder — functional, trained, checkpoint saved
- [x] Spatial DGAT server — 4-head GAT over 284-node district graph, functional
- [x] Dual-task head — regression (cases) + classification (outbreak prob), functional
- [x] FastAPI backend — `/api/predict`, `/api/graph`, `/api/federated-demo`, `/api/custom-predict`, `/api/model-info`, `/api/district-node`
- [x] React + Vite + Plotly dashboard — 3 pages (SpatialGraph, LivePredict, FederatedDemo)
- [x] Model checkpoint saved (`fedxgnn_best.pt`, 38,034 params, epoch 40)
- [x] Training curves PNG saved (6-panel: Loss, MAE, RMSE, AUC, AUPRC, F1)

---

## 🔴 Must Fix — Critical (will be caught in demo)

- [ ] **Re-run training on Kaggle** and download `metrics.json` output → commit to `results/metrics.json`
  - Produces verifiable val AUC, AUPRC, F1 numbers for the report
  - Current checkpoint was saved at epoch 40; re-run may improve scores
  - *Effort: 1–2 hrs (mostly Kaggle runtime)*

- [ ] **Temporal attention weight extraction** — explainability claim is currently empty
  - In `TemporalGAT.forward()`: add `return_attention_weights=True` to `self.gat1(...)`
  - Return `(N, T, T)` attention matrix alongside embeddings
  - Add `/api/attention?t=<window>` endpoint in `backend/server.py`
  - Render 4×4 heatmap in `FederatedDemo.jsx` (which week → which week per district)
  - *Effort: 4–6 hrs*

- [ ] **Baseline comparison script** — needed to prove GNN beats trivial lag-1 threshold
  - `scripts/baseline_eval.py`: logistic regression on `cases_lag1` alone → print AUC, F1
  - Add results to `results/metrics.json` under `"baseline"` key
  - *Effort: 1–2 hrs*

- [ ] **Fix report factual errors** (no code change, report text only):
  - NASA POWER → Open-Meteo archive API
  - BCEWithLogits → FocalLoss (α=0.75, γ=2.0) + MSELoss
  - Adam lr=0.001 → AdamW lr=2e-4
  - 80/20 split → 75/25 split
  - 100 epochs → 200 max, early stopped at ~90 epochs
  - Choropleth heatmap → Scatter geo dot map
  - NetworkX → direct pandas → torch edge list pipeline

---

## 🟡 Should Do — Stronger Final Report

- [ ] **Real federated simulation with Flower**
  - Install `flwr`
  - Split 284 districts into 5 client groups by geography/state
  - Implement `DengueClient(fl.client.NumPyClient)` with `get_parameters`, `fit`, `evaluate`
  - Run `fl.server.start_server` with `FedAvg` strategy
  - Log per-round accuracy, show client weight divergence before/after aggregation
  - *Effort: 8–12 hrs*

- [ ] **SHAP-based XAI**
  - `shap.DeepExplainer` on classification head
  - Per-feature importance bars (temp_k, preci_mm, cases_lag1, etc.)
  - Add `/api/shap?censuscode=<code>&t=<window>` endpoint
  - Render bar chart in dashboard
  - *Effort: 6–8 hrs*

- [ ] **Malaria data integration**
  - Pipeline already labels malaria rows in `build_all_datasets.py`
  - Add disease toggle (Dengue / Malaria) to frontend
  - Train separate head or disease-conditional model
  - *Effort: 6–8 hrs*

- [ ] **Choropleth map** (replace scatter geo)
  - Load DataMeet GeoJSON (already in `datasets/shapefiles/`)
  - Replace `type: 'scattergeo'` with `type: 'choropleth'` in `SpatialGraph.jsx` and `LivePredict.jsx`
  - Color district polygons by outbreak probability
  - *Effort: 3–4 hrs*

- [ ] **Trend charts per district**
  - Time-series line chart showing case trajectory + predictions over all weeks for a selected district
  - Add to district detail panel in `LivePredict.jsx`
  - *Effort: 2–3 hrs*

- [ ] **Commit trained `metrics.json` to `results/` folder**
  - After Kaggle re-run, commit `results/metrics.json` + `results/training_curves.png`
  - *Effort: 15 mins after training*

---

## 🟢 Nice to Have — Cut if Short on Time

- [ ] **Differential privacy** — add Opacus to training loop (requires full retraining)
- [ ] **Secure aggregation** — needs real federated setup first
- [ ] **Real-time inference** — WebSocket or polling endpoint for live prediction updates
- [ ] **Cloud deployment** — Railway/Render for FastAPI + Vercel for React frontend
- [ ] **End-to-end reproducibility script** — single `make` or `run_all.sh` from raw data → dashboard

---

## 📊 Current Metrics Status

| Metric | Value | Source | Verified? |
|--------|-------|--------|-----------|
| Val AUC-ROC | ~0.93–0.94 | `training_curves.png` (visual) | ⚠️ Approximate |
| Val AUPRC | ~0.15–0.20 | `training_curves.png` (visual) | ⚠️ Approximate |
| Val F1 | ~0.15–0.17 | `training_curves.png` (visual) | ⚠️ Approximate |
| Best epoch | 40 | `fedxgnn_best.pt` checkpoint | ✅ Exact |
| Total params | 38,034 | checkpoint `model_state` count | ✅ Exact |
| Districts | 284 | `training_dataset_enhanced_v2.csv` | ✅ Exact |
| Training rows | 16,302 | `training_dataset_enhanced_v2.csv` | ✅ Exact |
| Date range | 2009–2022 | dataset `iso_year` column | ✅ Exact |

> ⚠️ = Needs re-run to get exact numbers in `metrics.json`

---

## 🗓️ Update Log

| Date | What Changed |
|------|-------------|
| 2026-06-20 | Created this file from audit report |
| 2026-06-20 | Fixed `/api/timeline` NameError (`server.py`) |
| 2026-06-20 | Fixed `run_inference.py` dataset path |
| 2026-06-20 | Added `metrics.json` logging to `train_fedxgnn_run.py` |
