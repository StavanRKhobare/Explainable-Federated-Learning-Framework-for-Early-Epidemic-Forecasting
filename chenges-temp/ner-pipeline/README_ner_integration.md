# Dengue Epidemic Forecasting — NER Layer Integration Guide

This directory contains the **Named Entity Recognition (NER) pipeline** designed to process unstructured clinical notes, extract early epidemic signals (symptoms, diseases, pathogens, and travel history), aggregate them per district-week, and integrate them as new dynamic features into the **FedXGNN** model.

---

## 📂 Directory Structure

```
ner-pipeline/
├── requirements.txt             # Dependency requirements
├── generate_synthetic_notes.py  # Generates realistic clinical notes for 2022
├── ner_pipeline.py              # Core extraction engine with biomedical entity classification
├── ner_aggregator.py            # Extracts entities and aggregates them per district-week
├── integrate_ner_features.py    # Merges NER features into the project training dataset
├── ablation_study.py            # Evaluates predictive lift using Stratified 5-Fold CV
└── README_ner_integration.md    # This guide
```

---

## ⚙️ Step 1: Installation & Setup

1. **Activate your environment** (virtualenv/conda) and install basic dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download and install the large scispaCy model** (`en_core_sci_lg`):
   ```bash
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
   ```

---

## 🏃 Step 2: Running the Pipeline

To run the pipeline locally and merge features into the project training dataset:

1. **Generate Synthetic Notes** (simulates patient files for 2022 matching real district census codes):
   ```bash
   python generate_synthetic_notes.py
   # Generates: synthetic_notes.csv (censuscode, iso_year, iso_week, note)
   ```

2. **Process and Aggregate Entities**:
   ```bash
   python ner_aggregator.py
   # Runs NER, classifies entities, and sums them per district-week.
   # Generates: ner_features.csv (censuscode, iso_year, iso_week, ner_symptoms, ner_diseases, ner_pathogens, ner_travel, ner_total_notes)
   ```

3. **Merge Features into Primary Dataset**:
   ```bash
   python integrate_ner_features.py
   # Reads 'data/training_dataset_enhanced_v2.csv' in the project directory,
   # left-joins 'ner_features.csv', fills missing/prior years with 0, and saves:
   # Output: data/training_dataset_with_ner.csv (in the project data directory)
   ```

4. **Verify Predictive Performance (Ablation Study)**:
   ```bash
   python ablation_study.py
   # Runs a Stratified 5-Fold Cross-Validation on the 2022 subset to measure
   # AUPRC and ROC AUC improvement with the added NER features.
   ```

---

## 🚀 Step 3: Kaggle Retraining & Local Integration

Since adding 5 new features changes the input dimensions from **9 to 14**, the existing checkpoint `model/fedxgnn_best.pt` cannot be used directly. You must retrain the model on Kaggle using the new dataset.

### A. Retraining on Kaggle

1. **Upload Dataset**: Upload the newly generated `data/training_dataset_with_ner.csv` and your existing `graph_edges.csv` to Kaggle as a dataset.
2. **Update the Training Script** (`scripts/train_fedxgnn_run.py`):
   - Update `CFG["data_path"]` to point to the uploaded `training_dataset_with_ner.csv`.
   - Update `CFG["dynamic_features"]` in the config dictionary to include the new NER columns:
     ```python
     dynamic_features = [
         "temp_k", "preci_mm", "LAI",
         "cases_lag1", "cases_lag2", "cases_lag3",
         "week_sin", "week_cos", "is_monsoon",
         # --- Added NER Features ---
         "ner_symptoms", "ner_diseases", "ner_pathogens", "ner_travel", "ner_total_notes"
     ]
     ```
3. **Run Training**: Execute the notebook cells to train the model on Kaggle's T4 GPU.
4. **Download Weight Checkpoint**: Download the resulting `fedxgnn_best.pt` weight file from `/kaggle/working/`.

### B. Integrating the New Model Locally

Once training is complete, update your local repository files:

1. **Overwrite the Model Checkpoint**: Place the newly trained `fedxgnn_best.pt` inside the `model/` directory of the project, replacing the old file.
2. **Update `backend/server.py`**:
   - Change `DATA_PATH` (line 25) to point to `"training_dataset_with_ner.csv"`.
   - Update `CFG["dynamic_features"]` (line 33) to match the list used during Kaggle training:
     ```python
     dynamic_features= ["temp_k","preci_mm","LAI","cases_lag1","cases_lag2","cases_lag3","week_sin","week_cos","is_monsoon","ner_symptoms","ner_diseases","ner_pathogens","ner_travel","ner_total_notes"]
     ```
3. **Update `run_inference.py`**:
   - Change `DATA_PATH` (line 21) to point to `"data/training_dataset_with_ner.csv"`.
   - Update `CFG["dynamic_features"]` (line 28) to include the NER features.
