export const PRESETS = {
  high_risk: {
    name: "Scenario 1: High Outbreak Risk (Monsoon)",
    description: "Active monsoon season with heavy rainfall, elevated temperatures, rising cases, and high symptom counts in Bangalore and Mysore.",
    data: {
      "districts": [
        {
          "censuscode": 572,
          "weeks": [
            { "temp_k": 300.2, "preci_mm": 15.0, "LAI": 0.45, "cases_lag1": 3, "cases_lag2": 1, "cases_lag3": 0, "week_sin": 0.87, "week_cos": 0.5, "is_monsoon": 1, "ner_symptoms": 2, "ner_diseases": 1, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 3 },
            { "temp_k": 301.1, "preci_mm": 35.5, "LAI": 0.48, "cases_lag1": 5, "cases_lag2": 3, "cases_lag3": 1, "week_sin": 0.97, "week_cos": 0.26, "is_monsoon": 1, "ner_symptoms": 4, "ner_diseases": 2, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 6 },
            { "temp_k": 299.8, "preci_mm": 58.0, "LAI": 0.52, "cases_lag1": 10, "cases_lag2": 5, "cases_lag3": 3, "week_sin": 1.0, "week_cos": 0.0, "is_monsoon": 1, "ner_symptoms": 8, "ner_diseases": 4, "ner_pathogens": 1, "ner_travel": 0, "ner_total_notes": 12 },
            { "temp_k": 298.5, "preci_mm": 85.2, "LAI": 0.56, "cases_lag1": 22, "cases_lag2": 10, "cases_lag3": 5, "week_sin": 0.97, "week_cos": -0.26, "is_monsoon": 1, "ner_symptoms": 15, "ner_diseases": 7, "ner_pathogens": 2, "ner_travel": 1, "ner_total_notes": 22 }
          ]
        },
        {
          "censuscode": 577,
          "weeks": [
            { "temp_k": 298.0, "preci_mm": 8.0, "LAI": 0.42, "cases_lag1": 1, "cases_lag2": 0, "cases_lag3": 0, "week_sin": 0.87, "week_cos": 0.5, "is_monsoon": 1, "ner_symptoms": 1, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 1 },
            { "temp_k": 299.2, "preci_mm": 18.4, "LAI": 0.44, "cases_lag1": 2, "cases_lag2": 1, "cases_lag3": 0, "week_sin": 0.97, "week_cos": 0.26, "is_monsoon": 1, "ner_symptoms": 2, "ner_diseases": 1, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 3 },
            { "temp_k": 297.5, "preci_mm": 38.0, "LAI": 0.47, "cases_lag1": 4, "cases_lag2": 2, "cases_lag3": 1, "week_sin": 1.0, "week_cos": 0.0, "is_monsoon": 1, "ner_symptoms": 4, "ner_diseases": 2, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 6 },
            { "temp_k": 296.8, "preci_mm": 52.5, "LAI": 0.50, "cases_lag1": 9, "cases_lag2": 4, "cases_lag3": 2, "week_sin": 0.97, "week_cos": -0.26, "is_monsoon": 1, "ner_symptoms": 8, "ner_diseases": 3, "ner_pathogens": 1, "ner_travel": 1, "ner_total_notes": 11 }
          ]
        }
      ]
    }
  },
  low_risk: {
    name: "Scenario 2: Dry Winter (Low Risk)",
    description: "Dry season with cool temperatures, minimal rain, zero history of cases, and no clinical symptoms.",
    data: {
      "districts": [
        {
          "censuscode": 572,
          "weeks": [
            { "temp_k": 293.5, "preci_mm": 0.5, "LAI": 0.38, "cases_lag1": 0, "cases_lag2": 0, "cases_lag3": 0, "week_sin": -0.5, "week_cos": 0.86, "is_monsoon": 0, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 0 },
            { "temp_k": 293.1, "preci_mm": 0.0, "LAI": 0.37, "cases_lag1": 0, "cases_lag2": 0, "cases_lag3": 0, "week_sin": -0.6, "week_cos": 0.8, "is_monsoon": 0, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 0 },
            { "temp_k": 292.8, "preci_mm": 1.2, "LAI": 0.36, "cases_lag1": 0, "cases_lag2": 0, "cases_lag3": 0, "week_sin": -0.7, "week_cos": 0.71, "is_monsoon": 0, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 0 },
            { "temp_k": 293.0, "preci_mm": 0.2, "LAI": 0.35, "cases_lag1": 0, "cases_lag2": 0, "cases_lag3": 0, "week_sin": -0.8, "week_cos": 0.6, "is_monsoon": 0, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 1 }
          ]
        },
        {
          "censuscode": 577,
          "weeks": [
            { "temp_k": 292.5, "preci_mm": 0.0, "LAI": 0.35, "cases_lag1": 0, "cases_lag2": 0, "cases_lag3": 0, "week_sin": -0.5, "week_cos": 0.86, "is_monsoon": 0, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 0 },
            { "temp_k": 292.1, "preci_mm": 0.2, "LAI": 0.34, "cases_lag1": 0, "cases_lag2": 0, "cases_lag3": 0, "week_sin": -0.6, "week_cos": 0.8, "is_monsoon": 0, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 0 },
            { "temp_k": 291.9, "preci_mm": 0.5, "LAI": 0.33, "cases_lag1": 0, "cases_lag2": 0, "cases_lag3": 0, "week_sin": -0.7, "week_cos": 0.71, "is_monsoon": 0, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 0 },
            { "temp_k": 292.0, "preci_mm": 0.0, "LAI": 0.32, "cases_lag1": 0, "cases_lag2": 0, "cases_lag3": 0, "week_sin": -0.8, "week_cos": 0.6, "is_monsoon": 0, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 0 }
          ]
        }
      ]
    }
  },
  spatial_spillover: {
    name: "Scenario 3: Spatial Spillover (Neighbors)",
    description: "Bangalore has an active severe outbreak. Mysore (neighbor) currently has zero local cases, demonstrating how spatial graph signals propagate risk.",
    data: {
      "districts": [
        {
          "censuscode": 572,
          "weeks": [
            { "temp_k": 301.2, "preci_mm": 20.0, "LAI": 0.46, "cases_lag1": 10, "cases_lag2": 5, "cases_lag3": 2, "week_sin": 0.87, "week_cos": 0.5, "is_monsoon": 1, "ner_symptoms": 5, "ner_diseases": 2, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 7 },
            { "temp_k": 301.8, "preci_mm": 42.0, "LAI": 0.49, "cases_lag1": 18, "cases_lag2": 10, "cases_lag3": 5, "week_sin": 0.97, "week_cos": 0.26, "is_monsoon": 1, "ner_symptoms": 8, "ner_diseases": 4, "ner_pathogens": 1, "ner_travel": 0, "ner_total_notes": 11 },
            { "temp_k": 300.5, "preci_mm": 70.2, "LAI": 0.53, "cases_lag1": 30, "cases_lag2": 18, "cases_lag3": 10, "week_sin": 1.0, "week_cos": 0.0, "is_monsoon": 1, "ner_symptoms": 14, "ner_diseases": 6, "ner_pathogens": 2, "ner_travel": 0, "ner_total_notes": 18 },
            { "temp_k": 299.0, "preci_mm": 95.8, "LAI": 0.57, "cases_lag1": 55, "cases_lag2": 30, "cases_lag3": 18, "week_sin": 0.97, "week_cos": -0.26, "is_monsoon": 1, "ner_symptoms": 25, "ner_diseases": 12, "ner_pathogens": 3, "ner_travel": 1, "ner_total_notes": 35 }
          ]
        },
        {
          "censuscode": 577,
          "weeks": [
            { "temp_k": 298.0, "preci_mm": 5.0, "LAI": 0.40, "cases_lag1": 0, "cases_lag2": 0, "cases_lag3": 0, "week_sin": 0.87, "week_cos": 0.5, "is_monsoon": 1, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 0 },
            { "temp_k": 298.5, "preci_mm": 10.2, "LAI": 0.41, "cases_lag1": 0, "cases_lag2": 0, "cases_lag3": 0, "week_sin": 0.97, "week_cos": 0.26, "is_monsoon": 1, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 0 },
            { "temp_k": 297.0, "preci_mm": 15.0, "LAI": 0.43, "cases_lag1": 0, "cases_lag2": 0, "cases_lag3": 0, "week_sin": 1.0, "week_cos": 0.0, "is_monsoon": 1, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 0 },
            { "temp_k": 296.2, "preci_mm": 20.5, "LAI": 0.45, "cases_lag1": 0, "cases_lag2": 0, "cases_lag3": 0, "week_sin": 0.97, "week_cos": -0.26, "is_monsoon": 1, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 1 }
          ]
        }
      ]
    }
  },
  clinical_early_warning: {
    name: "Scenario 4: Early Warning (EHR Signal Spike)",
    description: "Coimbatore shows very low actual past cases, but Clinical EHR text mining captures a major surge in fever and pathogen mentions, providing early warning.",
    data: {
      "districts": [
        {
          "censuscode": 632,
          "weeks": [
            { "temp_k": 298.5, "preci_mm": 5.0, "LAI": 0.44, "cases_lag1": 0, "cases_lag2": 0, "cases_lag3": 0, "week_sin": 0.5, "week_cos": 0.86, "is_monsoon": 0, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 0 },
            { "temp_k": 299.0, "preci_mm": 12.0, "LAI": 0.45, "cases_lag1": 1, "cases_lag2": 0, "cases_lag3": 0, "week_sin": 0.6, "week_cos": 0.8, "is_monsoon": 1, "ner_symptoms": 2, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 2 },
            { "temp_k": 298.2, "preci_mm": 24.5, "LAI": 0.47, "cases_lag1": 1, "cases_lag2": 1, "cases_lag3": 0, "week_sin": 0.7, "week_cos": 0.71, "is_monsoon": 1, "ner_symptoms": 6, "ner_diseases": 2, "ner_pathogens": 1, "ner_travel": 0, "ner_total_notes": 8 },
            { "temp_k": 297.8, "preci_mm": 38.0, "LAI": 0.49, "cases_lag1": 2, "cases_lag2": 1, "cases_lag3": 1, "week_sin": 0.8, "week_cos": 0.6, "is_monsoon": 1, "ner_symptoms": 18, "ner_diseases": 5, "ner_pathogens": 2, "ner_travel": 0, "ner_total_notes": 22 }
          ]
        },
        {
          "censuscode": 572,
          "weeks": [
            { "temp_k": 298.0, "preci_mm": 10.0, "LAI": 0.45, "cases_lag1": 2, "cases_lag2": 1, "cases_lag3": 1, "week_sin": 0.5, "week_cos": 0.86, "is_monsoon": 0, "ner_symptoms": 1, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 2 },
            { "temp_k": 299.1, "preci_mm": 15.0, "LAI": 0.46, "cases_lag1": 3, "cases_lag2": 2, "cases_lag3": 1, "week_sin": 0.6, "week_cos": 0.8, "is_monsoon": 1, "ner_symptoms": 2, "ner_diseases": 1, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 3 },
            { "temp_k": 298.5, "preci_mm": 22.0, "LAI": 0.47, "cases_lag1": 3, "cases_lag2": 3, "cases_lag3": 2, "week_sin": 0.7, "week_cos": 0.71, "is_monsoon": 1, "ner_symptoms": 3, "ner_diseases": 1, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 5 },
            { "temp_k": 298.0, "preci_mm": 35.0, "LAI": 0.48, "cases_lag1": 4, "cases_lag2": 3, "cases_lag3": 3, "week_sin": 0.8, "week_cos": 0.6, "is_monsoon": 1, "ner_symptoms": 4, "ner_diseases": 2, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 6 }
          ]
        }
      ]
    }
  },
  travel_importation: {
    name: "Scenario 5: Travel Hub Importation Risk",
    description: "New Delhi records high volumes of clinical notes referencing travel history alongside symptoms, raising risk profile as a travel hub.",
    data: {
      "districts": [
        {
          "censuscode": 94,
          "weeks": [
            { "temp_k": 302.0, "preci_mm": 2.0, "LAI": 0.32, "cases_lag1": 1, "cases_lag2": 0, "cases_lag3": 0, "week_sin": 0.25, "week_cos": 0.96, "is_monsoon": 0, "ner_symptoms": 1, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 1, "ner_total_notes": 4 },
            { "temp_k": 303.5, "preci_mm": 8.0, "LAI": 0.31, "cases_lag1": 2, "cases_lag2": 1, "cases_lag3": 0, "week_sin": 0.35, "week_cos": 0.93, "is_monsoon": 0, "ner_symptoms": 2, "ner_diseases": 1, "ner_pathogens": 0, "ner_travel": 3, "ner_total_notes": 8 },
            { "temp_k": 304.0, "preci_mm": 18.5, "LAI": 0.30, "cases_lag1": 4, "cases_lag2": 2, "cases_lag3": 1, "week_sin": 0.45, "week_cos": 0.89, "is_monsoon": 1, "ner_symptoms": 5, "ner_diseases": 2, "ner_pathogens": 1, "ner_travel": 6, "ner_total_notes": 15 },
            { "temp_k": 303.2, "preci_mm": 25.0, "LAI": 0.30, "cases_lag1": 8, "cases_lag2": 4, "cases_lag3": 2, "week_sin": 0.55, "week_cos": 0.83, "is_monsoon": 1, "ner_symptoms": 10, "ner_diseases": 4, "ner_pathogens": 2, "ner_travel": 12, "ner_total_notes": 28 }
          ]
        },
        {
          "censuscode": 572,
          "weeks": [
            { "temp_k": 299.0, "preci_mm": 5.0, "LAI": 0.45, "cases_lag1": 2, "cases_lag2": 1, "cases_lag3": 1, "week_sin": 0.25, "week_cos": 0.96, "is_monsoon": 0, "ner_symptoms": 1, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 2 },
            { "temp_k": 299.5, "preci_mm": 8.2, "LAI": 0.45, "cases_lag1": 3, "cases_lag2": 2, "cases_lag3": 1, "week_sin": 0.35, "week_cos": 0.93, "is_monsoon": 0, "ner_symptoms": 2, "ner_diseases": 1, "ner_pathogens": 0, "ner_travel": 1, "ner_total_notes": 4 },
            { "temp_k": 298.8, "preci_mm": 12.0, "LAI": 0.46, "cases_lag1": 3, "cases_lag2": 3, "cases_lag3": 2, "week_sin": 0.45, "week_cos": 0.89, "is_monsoon": 0, "ner_symptoms": 2, "ner_diseases": 1, "ner_pathogens": 0, "ner_travel": 1, "ner_total_notes": 5 },
            { "temp_k": 298.2, "preci_mm": 18.5, "LAI": 0.46, "cases_lag1": 4, "cases_lag2": 3, "cases_lag3": 3, "week_sin": 0.55, "week_cos": 0.83, "is_monsoon": 0, "ner_symptoms": 3, "ner_diseases": 1, "ner_pathogens": 0, "ner_travel": 3, "ner_total_notes": 8 }
          ]
        }
      ]
    }
  },
  normal_baseline: {
    name: "Scenario 6: Stable Baseline (Post-Monsoon)",
    description: "Post-monsoon winter conditions. Mild temperature, low case history, stable health reports and clinical activities.",
    data: {
      "districts": [
        {
          "censuscode": 632,
          "weeks": [
            { "temp_k": 297.0, "preci_mm": 2.0, "LAI": 0.44, "cases_lag1": 1, "cases_lag2": 2, "cases_lag3": 1, "week_sin": -0.87, "week_cos": -0.5, "is_monsoon": 0, "ner_symptoms": 1, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 2 },
            { "temp_k": 296.8, "preci_mm": 1.5, "LAI": 0.43, "cases_lag1": 1, "cases_lag2": 1, "cases_lag3": 2, "week_sin": -0.78, "week_cos": -0.62, "is_monsoon": 0, "ner_symptoms": 1, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 1 },
            { "temp_k": 296.5, "preci_mm": 3.0, "LAI": 0.42, "cases_lag1": 2, "cases_lag2": 1, "cases_lag3": 1, "week_sin": -0.65, "week_cos": -0.76, "is_monsoon": 0, "ner_symptoms": 2, "ner_diseases": 1, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 3 },
            { "temp_k": 296.0, "preci_mm": 1.0, "LAI": 0.40, "cases_lag1": 2, "cases_lag2": 2, "cases_lag3": 1, "week_sin": -0.5, "week_cos": -0.86, "is_monsoon": 0, "ner_symptoms": 1, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 2 }
          ]
        },
        {
          "censuscode": 577,
          "weeks": [
            { "temp_k": 296.5, "preci_mm": 1.0, "LAI": 0.42, "cases_lag1": 1, "cases_lag2": 1, "cases_lag3": 0, "week_sin": -0.87, "week_cos": -0.5, "is_monsoon": 0, "ner_symptoms": 0, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 0 },
            { "temp_k": 296.2, "preci_mm": 2.2, "LAI": 0.41, "cases_lag1": 1, "cases_lag2": 1, "cases_lag3": 1, "week_sin": -0.78, "week_cos": -0.62, "is_monsoon": 0, "ner_symptoms": 1, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 1 },
            { "temp_k": 296.0, "preci_mm": 0.5, "LAI": 0.40, "cases_lag1": 1, "cases_lag2": 1, "cases_lag3": 1, "week_sin": -0.65, "week_cos": -0.76, "is_monsoon": 0, "ner_symptoms": 1, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 1 },
            { "temp_k": 295.8, "preci_mm": 0.0, "LAI": 0.39, "cases_lag1": 1, "cases_lag2": 1, "cases_lag3": 1, "week_sin": -0.5, "week_cos": -0.86, "is_monsoon": 0, "ner_symptoms": 1, "ner_diseases": 0, "ner_pathogens": 0, "ner_travel": 0, "ner_total_notes": 2 }
          ]
        }
      ]
    }
  }
};
