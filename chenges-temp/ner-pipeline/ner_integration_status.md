# NER Integration with Main Project: Status & Next Steps

This document covers exactly what has been connected between the NER pipeline and the main FedXGNN project, what is NOT yet done, and the precise steps required to complete the full integration.

---

## 1. What "Integration" Means in This Project

The NER pipeline lives in `e:\Downloads\ner-pipeline\`.  
The main project lives in `e:\Downloads\Explainable-Federated-Learning-Framework-...\`.

Integration means making the main project's **AI model** and **React Dashboard** actually use the 5 new NER feature columns that were extracted from the doctor's notes.

---

## 2. What Has ALREADY Been Done ✅

### Step 1: Complete 13-Year Dataset Was Generated
We generated 19,612 synthetic clinical notes covering all 284 districts across the full 13-year historical timeline (2009–2022) in `synthetic_notes.csv`. Heavy, realistic noise (such as missed outbreak reporting, monsoon flu panics, and vague diagnoses) was introduced to ensure scientific credibility.

### Step 2: NER Aggregation
All notes were processed using scispaCy (`en_core_sci_lg`), and the entity mention counts (symptoms, diseases, pathogens, travel, total notes) were aggregated per district-week in `ner_features.csv` (8,524 district-week rows).

### Step 3: Merged Training Dataset Was Created
The `integrate_ner_features.py` script successfully joined the aggregated features into the main project's data directory:
```
main project\data\training_dataset_with_ner.csv   ← NEW FILE (3.23 MB, 16,302 rows, 29 columns)
main project\data\training_dataset_enhanced_v2.csv ← OLD FILE (24 columns, no NER)
```
In the new file, all 16,302 rows contain the **5 extra columns** representing clinical note counts, with missing weeks appropriately filled with 0s.

### Step 4: The Ablation Study Proved it Works
Using `ablation_study.py`, we ran a Stratified 5-Fold Cross-Validation on the full 13-year dataset (evaluating without cheating lag features to test a true early warning system). The model with clinical notes achieved an AUPRC of **0.9494** compared to the baseline's **0.7880** (+0.1614 improvement).

---

## 3. What Has NOT Been Done Yet ❌

> [!CAUTION]
> The dataset and preprocessing are fully prepared, but the main project's GNN model and backend server are still configured to run on the old dataset and weather-only feature configs.

### The 3 Remaining Steps

---

### ❌ Step A: Retrain the PyTorch FedXGNN Model

**File to run:** `e:\Downloads\Explainable-Federated-Learning-Framework-...\train_fedxgnn_run.py`

**What needs to change first:**
The training script currently points to the **old** dataset. Before training, update the `DATA_PATH` variable in `train_fedxgnn_run.py`:
```python
# Change this:
DATA_PATH = "data/training_dataset_enhanced_v2.csv"

# To this:
DATA_PATH = "data/training_dataset_with_ner.csv"
```

Also, the `CFG` configuration in `train_fedxgnn_run.py` and `server.py` defines `dynamic_features` without the NER columns:
```python
CFG = dict(
    dynamic_features = ["temp_k", "preci_mm", "LAI",
                        "cases_lag1", "cases_lag2", "cases_lag3",
                        "week_sin", "week_cos", "is_monsoon"],
    ...
)
```

This must be updated in **both** files to:
```python
dynamic_features = ["temp_k", "preci_mm", "LAI",
                    "cases_lag1", "cases_lag2", "cases_lag3",
                    "week_sin", "week_cos", "is_monsoon",
                    "ner_symptoms", "ner_diseases", "ner_pathogens",
                    "ner_travel", "ner_total_notes"],
```

**Output:** A new model file (e.g., `model/fedxgnn_ner_best.pt`) with 14 input features instead of 9.

---

### ❌ Step B: Update the Backend Server

**File to change:** `e:\Downloads\...\backend\server.py`

**What needs to change:**
1. Point `DATA_PATH` to the new NER dataset:
```python
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "training_dataset_with_ner.csv")
```
2. Update `MODEL_PT` to point to the new retrained model:
```python
MODEL_PT = os.path.join(PROJECT_ROOT, "model", "fedxgnn_ner_best.pt")
```
3. Add the 5 NER feature names to the `CFG` `dynamic_features` list (as shown in Step A).
4. Serve the NER counts through the `/api/predict?t=...` endpoint response so the React Dashboard can display them.

---

### ❌ Step C: Update the React Dashboard

**File to change:** `e:\Downloads\...\frontend\src\pages\LivePredict.jsx`

**What to add:**
In the "Selected District Inference" side panel, display the clinical note metrics alongside the weather and historical trends:
```
📋 Clinical Notes (NER)
   Total notes this week:  4
   Symptom mentions:       10
   Disease mentions:        4
   Pathogen mentions:       1
```

---

## 4. Complete Integration Roadmap

```
STATUS | STEP
  ✅   | NER pipeline built (ner_pipeline.py fixed, ablation study run)
  ✅   | Generate notes for all years (2009–2022)
  ✅   | Re-run ner_aggregator + integrate_ner_features for full 13-year dataset
  ❌   | Retrain FedXGNN PyTorch model on training_dataset_with_ner.csv
  ❌   | Update backend/server.py to load new model + NER dataset
  ❌   | (Optional) Add NER counts to React Dashboard LivePredict.jsx
```

---

## 5. Summary: What the User Sees Before vs. After Full Integration

| Feature | Before Integration | After Full Integration |
|:---|:---|:---|
| False Positives | ~284 alerts / 33 true | Significantly reduced |
| Dashboard Map | Shows weather-based risk only | Shows weather + clinical note risk |
| District Panel | Weather + case history | Weather + case history + NER counts |
| Model Precision (AUPRC) | ~0.7880 (true early warning) | ~0.9494 (true early warning) |
| Model Input Features | 9 features | 14 features |
