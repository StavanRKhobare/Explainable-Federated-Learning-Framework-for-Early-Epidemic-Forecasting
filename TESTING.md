# FedXGNN Dashboard — Split-Federated Demo Testing Guide

## About Expected Outputs
The model produces `outbreak_prob` values on a [0, 1] scale.
Based on live validation of the Phase 3 model (Epoch 193, AUPRC 0.78, F1 0.73):
- **Active outbreaks**: Probability ~0.60–0.77
- **Declining outbreaks**: Probability ~0.55–0.65
- **Early warning (low cases, high risk climate)**: Probability ~0.40–0.50
- **Absolute safe baseline**: Probability ~0.29–0.31
- **Threshold**: Predictions ≥ 0.5 are classified as HIGH RISK

> All test cases below are verified against the live `/api/custom-predict` endpoint.

---

## Test Case 1 — 🚨 Active Peak Outbreak (High Risk)
**What it proves:** A district with a massive, sustained case load during a peak epidemic.
This mirrors the real Muktsar, Punjab outbreak (23,389 cases).
The model correctly fires a HIGH RISK alert.

**Expected: `outbreak_prob` ~ **0.71**, `cases_pred` ~ 44**

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

## Test Case 2 — 📉 Declining Outbreak (Moderate Risk)
**What it proves:** The model understands **trajectory**, not just case count.
Even with high absolute numbers, a consistent decline (500 → 300 → 150 → 80 → 30 → 5)
signals the epidemic is burning out, so probability drops below TC1 despite similar totals.

**Expected: `outbreak_prob` ~ **0.62** (lower than TC1, still flagged)**

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

## Test Case 3 — 🦠 Early Warning Signal (Climate Risk, Low Cases)
**What it proves:** The model detects an exponential growth signature (0 → 5 → 12 → 28)
under peak monsoon conditions even when absolute numbers are very low.
This is the **early-warning power** of the GRU + Temporal GAT.

**Expected: `outbreak_prob` ~ **0.41** (WATCH classification)**

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
**What it proves:** Muktsar (code 44) has an active epidemic with thousands of cases.
New Delhi (code 94) has **zero cases** but shares graph edges.
The Spatial DGAT propagates risk across the graph — New Delhi's `outbreak_prob` is
elevated to **0.74** purely due to graph neighborhood effects, even with zero local cases.

**Expected: Muktsar ~ **0.76**, New Delhi ~ **0.74** (both HIGH RISK)**

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

## Test Case 5 — ❄️ Safe Zone / Baseline (Winter, Zero Cases)
**What it proves:** Bangalore Rural (code 583) in cold dry winter conditions with
effectively zero case momentum. The model correctly outputs its minimum possible risk.
**Note:** The model's absolute safe floor is ~0.29–0.31 (not 0.0) because it was trained
with `pos_weight` to handle class imbalance, so the sigmoid baseline is elevated.

**Expected: `outbreak_prob` ~ **0.30** (SAFE / model's minimum baseline)**

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
**What it proves:** Graph stability. Bangalore Rural (583) and Ramanagara (584) are
**confirmed neighboring districts** (`are_neighbors: true`). Both have near-zero cases
and cold dry conditions. The Spatial DGAT correctly confirms a mutual Safe Zone —
neither district inflates the other's risk through graph propagation.

**Expected: Bangalore Rural ~ **0.30**, Ramanagara ~ **0.30** (both SAFE)**

```json
{
  "districts": [
    {
      "censuscode": 583,
      "weeks": [
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 1, "cases_lag2": 1, "cases_lag1": 2, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 1, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 0, "cases_lag2": 2, "cases_lag1": 2, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 1, "cases_lag2": 1, "cases_lag1": 1, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 }
      ]
    },
    {
      "censuscode": 584,
      "weeks": [
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
        { "temp_k": 273.0, "preci_mm": 0.0, "LAI": 1.0, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 0, "week_cos": 1, "is_monsoon": 0 },
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
3. **Autoregressive Strength**: Epidemics are highly "autoregressive" — meaning the best predictor of cases tomorrow is often the number of cases today. Lags provide this critical baseline.

---

# 📊 Quick Reference — What Numbers to Expect

| Scenario | Outbreak Prob | Cases Pred | Classification |
|---|---|---|---|
| Massive active outbreak (23k cases) | ~0.71 | ~44 | 🔴 HIGH RISK |
| Declining outbreak (500→5 cases) | ~0.62 | ~8 | 🟠 HIGH RISK |
| Early warning (monsoon + 28 cases) | ~0.41 | ~1 | 🟡 WATCH |
| Graph propagation (neighbor of active) | ~0.74 | ~45 | 🔴 HIGH RISK |
| Cold winter, 0 cases | ~0.30 | ~0.4 | 🟢 SAFE |
| Two safe neighbors | ~0.30 | ~0.4 | 🟢 SAFE |
