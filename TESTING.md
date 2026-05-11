# FedXGNN Dashboard — Split-Federated Demo Testing Guide

## About Expected Outputs
The model produces `outbreak_prob` values on a [0, 1] scale.
Based on validation analysis of the Phase 3 model (Epoch 193, AUPRC 0.78):
- **True outbreaks**: Mean probability ~0.60, Max ~0.78
- **Non-outbreaks**: Mean probability ~0.41
- **Threshold**: Predictions ≥ 0.5 are classified as HIGH RISK

---

## Test Case 1 — 🚨 Active Peak Outbreak (High Risk)
**What it proves:** A district with a massive, sustained case load during monsoon season.
This mirrors the real 2021 Muktsar, Punjab outbreak (23,389 cases).
**Expected `outbreak_prob`: ~0.55–0.70**

```json
{
  "districts": [
    {
      "censuscode": 44,
      "weeks": [
        { "temp_k": 289.4, "preci_mm": 3.6, "LAI": 2.0, "cases_lag3": 4309, "cases_lag2": 0, "cases_lag1": 0, "week_sin": -0.78, "week_cos": -0.62, "is_monsoon": 0 },
        { "temp_k": 289.4, "preci_mm": 3.6, "LAI": 2.0, "cases_lag3": 0, "cases_lag2": 4309, "cases_lag1": 0, "week_sin": -0.78, "week_cos": -0.62, "is_monsoon": 0 },
        { "temp_k": 289.4, "preci_mm": 3.6, "LAI": 2.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 4309, "week_sin": -0.78, "week_cos": -0.62, "is_monsoon": 0 },
        { "temp_k": 289.4, "preci_mm": 3.6, "LAI": 2.0, "cases_lag3": 0, "cases_lag2": 4309, "cases_lag1": 23389, "week_sin": -0.78, "week_cos": -0.62, "is_monsoon": 0 }
      ]
    }
  ]
}
```

---

## Test Case 2 — 📉 Declining Outbreak (Lower Risk)
**What it proves:** The model is NOT just a case-count threshold — it understands
trajectory. Even with high absolute numbers, a consistent decline signals the
epidemic is burning out.
**Expected `outbreak_prob`: ~0.40–0.50** (lower than Test 1)

```json
{
  "districts": [
    {
      "censuscode": 572,
      "weeks": [
        { "temp_k": 298.5, "preci_mm": 10.0, "LAI": 8.0, "cases_lag3": 500, "cases_lag2": 300, "cases_lag1": 150, "week_sin": 0.8, "week_cos": 0.5, "is_monsoon": 0 },
        { "temp_k": 298.5, "preci_mm": 10.0, "LAI": 8.0, "cases_lag3": 300, "cases_lag2": 150, "cases_lag1": 80, "week_sin": 0.8, "week_cos": 0.5, "is_monsoon": 0 },
        { "temp_k": 298.5, "preci_mm": 10.0, "LAI": 8.0, "cases_lag3": 150, "cases_lag2": 80, "cases_lag1": 30, "week_sin": 0.8, "week_cos": 0.5, "is_monsoon": 0 },
        { "temp_k": 298.5, "preci_mm": 10.0, "LAI": 8.0, "cases_lag3": 80, "cases_lag2": 30, "cases_lag1": 5, "week_sin": 0.8, "week_cos": 0.5, "is_monsoon": 0 }
      ]
    }
  ]
}
```

---

## Test Case 3 — 🦠 Early Warning Signal (High Risk, Low Numbers)
**What it proves:** The model detects an exponential growth signature (doubling
each week) and high-risk monsoon conditions, even when absolute numbers are low.
**Expected `outbreak_prob`: ~0.48–0.58**

```json
{
  "districts": [
    {
      "censuscode": 572,
      "weeks": [
        { "temp_k": 300.4, "preci_mm": 51.4, "LAI": 8.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 },
        { "temp_k": 300.4, "preci_mm": 55.0, "LAI": 8.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 5, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 },
        { "temp_k": 300.4, "preci_mm": 60.0, "LAI": 8.0, "cases_lag3": 0, "cases_lag2": 5, "cases_lag1": 12, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 },
        { "temp_k": 300.4, "preci_mm": 65.0, "LAI": 8.0, "cases_lag3": 5, "cases_lag2": 12, "cases_lag1": 28, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 }
      ]
    }
  ]
}
```

---

## Test Case 4 — 🔗 Spatial DGAT Propagation (Two Districts)
**What it proves:** District 44 (Muktsar) has an active outbreak. District 94
(New Delhi) has zero reported cases BUT shares graph edges. The Spatial DGAT
propagates risk across the graph — New Delhi's `outbreak_prob` will be elevated
above baseline purely due to graph neighborhood effects.
**Expected: Muktsar prob > New Delhi prob > 0.4**

```json
{
  "districts": [
    {
      "censuscode": 44,
      "weeks": [
        { "temp_k": 304.7, "preci_mm": 23.1, "LAI": 2.0, "cases_lag3": 9494, "cases_lag2": 5904, "cases_lag1": 0, "week_sin": 0.6, "week_cos": 0.8, "is_monsoon": 1 },
        { "temp_k": 304.7, "preci_mm": 23.1, "LAI": 2.0, "cases_lag3": 5904, "cases_lag2": 0, "cases_lag1": 10289, "week_sin": 0.6, "week_cos": 0.8, "is_monsoon": 1 },
        { "temp_k": 304.7, "preci_mm": 23.1, "LAI": 2.0, "cases_lag3": 0, "cases_lag2": 10289, "cases_lag1": 8000, "week_sin": 0.6, "week_cos": 0.8, "is_monsoon": 1 },
        { "temp_k": 304.7, "preci_mm": 23.1, "LAI": 2.0, "cases_lag3": 10289, "cases_lag2": 8000, "cases_lag1": 5000, "week_sin": 0.6, "week_cos": 0.8, "is_monsoon": 1 }
      ]
    },
    {
      "censuscode": 94,
      "weeks": [
        { "temp_k": 297.0, "preci_mm": 2.7, "LAI": 6.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0.6, "week_cos": 0.8, "is_monsoon": 0 },
        { "temp_k": 297.0, "preci_mm": 2.7, "LAI": 6.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0.6, "week_cos": 0.8, "is_monsoon": 0 },
        { "temp_k": 297.0, "preci_mm": 2.7, "LAI": 6.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0.6, "week_cos": 0.8, "is_monsoon": 0 },
        { "temp_k": 297.0, "preci_mm": 2.7, "LAI": 6.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0.6, "week_cos": 0.8, "is_monsoon": 0 }
      ]
    }
  ]
}
```

---

## Test Case 5 — ❄️ Safe Zone / Baseline (Northern Winter)
**What it proves:** District 1 (Kupwara, J&K) is historically cold and dry in
winter. With zero reported cases and sub-zero temperatures, the model correctly
predicts the minimum possible risk. 
**Note:** Because the model was trained with high `pos_weight` to handle data 
imbalance, it rarely outputs 0.0. A value between **0.20 - 0.35** represents 
the model's "absolute safe" baseline.
**Expected: Outbreak Prob ~ 0.29**

```json
{
  "districts": [
    {
      "censuscode": 583,
      "weeks": [
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 10, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 0, "cases_lag2": 10, "cases_lag1": 0, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 10, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 }
      ]
    }
  ]
}
```

---

## Test Case 6 — 🤝 Mutual Low Risk (Neighboring Districts)
**What it proves:** Demonstration of graph stability. When two neighboring districts
(Bangalore Rural and Ramanagara) both have zero cases and safe weather, the
Spatial DGAT confirms a mutual "Safe Zone," preventing noise in one from 
triggering a false alarm in the other.
**Expected: Both Outbreak Probs ~ 0.29**

```json
{
  "districts": [
    {
      "censuscode": 583,
      "weeks": [
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 10, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 0, "cases_lag2": 10, "cases_lag1": 0, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 10, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 }
      ]
    },
    {
      "censuscode": 584,
      "weeks": [
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 10, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 0, "cases_lag2": 10, "cases_lag1": 0, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 10, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 }
      ]
    }
  ]
}
```

---

# 🧠 Understanding "Cases Lag"

### What exactly is Cases Lag?
"Lag" refers to looking backward in time. 
- **cases_lag1**: Number of cases reported **1 week ago**.
- **cases_lag2**: Number of cases reported **2 weeks ago**.
- **cases_lag3**: Number of cases reported **3 weeks ago**.

### Where is it coming from in our Weeks data?
In our input JSON, each "week" object contains these three features. This is because a single week's weather isn't enough to predict an outbreak; the model needs to know the **momentum**. 

For example, if you are predicting for "Week 4", the `cases_lag1` in that object is the case count from "Week 3". By providing these as features, we allow the GRU (Temporal Encoder) to see exactly how the virus is spreading over a 4-week window.

### How does it help the prediction?
1. **Exponential Growth Detection**: If cases go from 5 → 25 → 100 (lags moving upward), the model identifies the "acceleration" of an epidemic.
2. **Incubation Periods**: Dengue/Malaria have incubation periods. Today's weather affects cases 2-3 weeks from now. Lags help the model bridge this time gap.
3. **Autoregressive Strength**: Epidemics are highly "autoregressive"—meaning the best predictor of cases tomorrow is often the number of cases today. Lags provide this critical baseline.

