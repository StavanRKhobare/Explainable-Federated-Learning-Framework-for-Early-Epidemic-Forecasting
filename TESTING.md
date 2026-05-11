1. 🚨 The "Explosive Outbreak" (High Risk)
What it proves: The model correctly identifies a dangerous, uncontrolled geometric spike in cases, which is the hallmark of a true epidemic. The sequence: 10 ➔ 12 ➔ 25 ➔ 80 ➔ 250 ➔ 800
{
  "districts": [
    {
      "censuscode": 572,
      "weeks": [
        { "temp_k": 302.1, "preci_mm": 45.0, "LAI": 0.5, "cases_lag3": 10, "cases_lag2": 12, "cases_lag1": 25, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 },
        { "temp_k": 302.1, "preci_mm": 45.0, "LAI": 0.5, "cases_lag3": 12, "cases_lag2": 25, "cases_lag1": 80, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 },
        { "temp_k": 302.1, "preci_mm": 45.0, "LAI": 0.5, "cases_lag3": 25, "cases_lag2": 80, "cases_lag1": 250, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 },
        { "temp_k": 302.1, "preci_mm": 45.0, "LAI": 0.5, "cases_lag3": 80, "cases_lag2": 250, "cases_lag1": 800, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 }
      ]
    }
  ]
}
2. 📉 The "Contained Decline" (Low Risk, High Numbers)
What it proves: The model isn't just a basic threshold trigger. It understands that a negative derivative (cases dropping) means the outbreak is already contained, even if the absolute numbers are still huge. The sequence: 1000 ➔ 800 ➔ 500 ➔ 200 ➔ 80 ➔ 20
{
  "districts": [
    {
      "censuscode": 572,
      "weeks": [
        { "temp_k": 298.5, "preci_mm": 10.0, "LAI": 0.4, "cases_lag3": 1000, "cases_lag2": 800, "cases_lag1": 500, "week_sin": 0.8, "week_cos": 0.5, "is_monsoon": 0 },
        { "temp_k": 298.5, "preci_mm": 10.0, "LAI": 0.4, "cases_lag3": 800, "cases_lag2": 500, "cases_lag1": 200, "week_sin": 0.8, "week_cos": 0.5, "is_monsoon": 0 },
        { "temp_k": 298.5, "preci_mm": 10.0, "LAI": 0.4, "cases_lag3": 500, "cases_lag2": 200, "cases_lag1": 80, "week_sin": 0.8, "week_cos": 0.5, "is_monsoon": 0 },
        { "temp_k": 298.5, "preci_mm": 10.0, "LAI": 0.4, "cases_lag3": 200, "cases_lag2": 80, "cases_lag1": 20, "week_sin": 0.8, "week_cos": 0.5, "is_monsoon": 0 }
      ]
    }
  ]
}
3. 🦠 The "Early Warning" (High Risk, Low Numbers)
What it proves: The model detects the dangerous mathematical signature of an early-stage contagion (doubling every week) before the numbers get out of control. The sequence: 0 ➔ 0 ➔ 1 ➔ 2 ➔ 4 ➔ 8
{
  "districts": [
    {
      "censuscode": 572,
      "weeks": [
        { "temp_k": 300.0, "preci_mm": 35.0, "LAI": 0.48, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 1, "week_sin": 0.9, "week_cos": 0.2, "is_monsoon": 1 },
        { "temp_k": 300.0, "preci_mm": 35.0, "LAI": 0.48, "cases_lag3": 0, "cases_lag2": 1, "cases_lag1": 2, "week_sin": 0.9, "week_cos": 0.2, "is_monsoon": 1 },
        { "temp_k": 300.0, "preci_mm": 35.0, "LAI": 0.48, "cases_lag3": 1, "cases_lag2": 2, "cases_lag1": 4, "week_sin": 0.9, "week_cos": 0.2, "is_monsoon": 1 },
        { "temp_k": 300.0, "preci_mm": 35.0, "LAI": 0.48, "cases_lag3": 2, "cases_lag2": 4, "cases_lag1": 8, "week_sin": 0.9, "week_cos": 0.2, "is_monsoon": 1 }
      ]
    }
  ]
}
4. 🔗 The "Spatial DGAT Propagation" Edge Case
What it proves: This uses two neighboring districts. Bangalore (572) has a massive outbreak. Mysore (577) has exactly 0 cases. However, because the Spatial DGAT passes graph attention weights across their shared border, Mysore's risk level will rise purely because its neighbor is infected. This demonstrates the power of Graph Neural Networks!
{
  "districts": [
    {
      "censuscode": 572,
      "weeks": [
        { "temp_k": 301.0, "preci_mm": 50.0, "LAI": 0.5, "cases_lag3": 10, "cases_lag2": 30, "cases_lag1": 100, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 },
        { "temp_k": 301.0, "preci_mm": 50.0, "LAI": 0.5, "cases_lag3": 30, "cases_lag2": 100, "cases_lag1": 300, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 },
        { "temp_k": 301.0, "preci_mm": 50.0, "LAI": 0.5, "cases_lag3": 100, "cases_lag2": 300, "cases_lag1": 900, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 },
        { "temp_k": 301.0, "preci_mm": 50.0, "LAI": 0.5, "cases_lag3": 300, "cases_lag2": 900, "cases_lag1": 2500, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 }
      ]
    },
    {
      "censuscode": 577,
      "weeks": [
        { "temp_k": 301.0, "preci_mm": 50.0, "LAI": 0.5, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 },
        { "temp_k": 301.0, "preci_mm": 50.0, "LAI": 0.5, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 },
        { "temp_k": 301.0, "preci_mm": 50.0, "LAI": 0.5, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 },
        { "temp_k": 301.0, "preci_mm": 50.0, "LAI": 0.5, "cases_lag3": 0, "cases_lag2": 0, "cases_lag1": 0, "week_sin": 1, "week_cos": 0, "is_monsoon": 1 }
      ]
    }
  ]
}
