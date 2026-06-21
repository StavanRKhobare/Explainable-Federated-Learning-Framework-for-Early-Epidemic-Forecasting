# Baseline Project State (Pre-NER Integration)

This document summarizes the state of the **Explainable Federated Learning Framework for Early Epidemic Forecasting (FedXGNN)** prior to the integration of the Natural Language Processing (NER) layer.

## 1. Core Architecture
The original project was built using a **Split-Federated Spatio-Temporal Graph Neural Network (Split-FedSTGNN)**. The architecture was designed to preserve privacy while forecasting disease outbreaks across 284 Indian districts.

* **Client Layer (Temporal):** Each district hospital (Client) runs a local Gated Recurrent Unit (GRU) model. It processes local time-series data to understand the temporal trend of the disease without sharing raw patient counts.
* **Server Layer (Spatial):** The central server runs a Directed Graph Attention Network (DGAT). It uses a physical border graph (`graph_edges.csv`) to learn how diseases spread geographically from one district to its neighbors.
* **Dual-Task Head:** The server outputs two predictions simultaneously:
  1. Regression: The exact number of predicted cases.
  2. Classification: The probability of an outbreak occurring (`is_outbreak`).

## 2. Baseline Features
Before the NER pipeline was introduced, the model relied exclusively on two categories of data:

1. **Environmental / Meteorological:**
   * Temperature (`temp_k`)
   * Rainfall (`preci_mm`)
   * Leaf Area Index (`LAI`)
   * Monsoon indicator (`is_monsoon`)
2. **Epidemiological (Historical):**
   * Past case counts (`cases_lag1`, `cases_lag2`, `cases_lag3`)
   * 4-week rolling average of cases (`cases_roll4w`)
3. **Temporal:**
   * Cyclical week representations (`week_sin`, `week_cos`)

## 3. The Core Problem: The False Positive Dilemma
While the federated graph architecture successfully modeled the spread of the disease, the **feature set** created a critical flaw in the model's precision.

### The "Fancy Weather Forecaster" Effect
Dengue outbreaks are heavily dependent on mosquito breeding, which requires specific temperature and rainfall conditions. Because the model *only* understood weather and historical numbers, it essentially functioned as a highly sensitive environmental forecaster. 

During the Indian monsoon season (Weeks 24-42), rainfall and temperature are perfectly aligned for mosquitoes across the entire country. 

> [!WARNING]  
> **The Result:** When the model saw peak monsoon conditions, it would flag almost the entire country as "High Risk." For example, in 2019 Week 34, the dashboard showed High-Risk alerts for nearly all **284 districts**, even though only **33 true outbreaks** actually occurred.

### The Impact of Focal Loss
To ensure the early warning system didn't miss actual outbreaks, the model was trained using **Focal Loss**, which heavily mathematically penalizes False Negatives (missing a real outbreak). 
The side effect of this loss function, combined with the lack of specific human clinical data, made the model hyper-paranoid. It preferred to "cry wolf" and warn everyone rather than risk missing a single outbreak.

## 4. Baseline Model Metrics
As a result of the False Positive dilemma, the baseline metrics reflected a highly sensitive but imprecise model:

* **ROC AUC (~0.94 - 0.98):** Very high. The model was excellent at ranking risk globally and knew that dry winter seasons were perfectly safe.
* **F1-Score (~0.16):** Extremely low. Because the model generated hundreds of false alarms during the monsoon, its precision was severely compromised.
* **AUPRC:** Extremely low in early-warning evaluations (predicting without relying on `cases_lag1`). 

### Conclusion (Pre-NER)
The baseline project successfully proved that a Federated Graph Neural Network could model disease spread privately. However, it hit an accuracy ceiling. Without real-time, human-centric clinical data, the AI could not distinguish between a district that just had heavy rain and a district that was actually experiencing an emerging epidemic.
