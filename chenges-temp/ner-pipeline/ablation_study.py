"""
ablation_study.py
=================
Compares model performance (is_outbreak prediction) with and without NER features.
Since synthetic clinical notes (and thus NER features) are generated for the year 2022,
the evaluation is performed on 2022 data using Stratified 5-Fold Cross-Validation.

Usage:
    python ablation_study.py
"""

import pandas as pd
import numpy as np
import os
import sys

# Avoid unicode print errors on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(
    BASE_DIR,
    "..",
    "Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting-main",
    "Explainable-Federated-Learning-Framework-for-Early-Epidemic-Forecasting-main",
    "data",
    "training_dataset_with_ner.csv"
)

# Feature sets
BASELINE_FEATURES = [
    "temp_k", "preci_mm", "LAI",
    "week_sin", "week_cos", "is_monsoon"
]

NER_FEATURES = [
    "ner_symptoms", "ner_diseases", "ner_pathogens", "ner_travel", "ner_total_notes"
]

ENHANCED_FEATURES = BASELINE_FEATURES + NER_FEATURES
TARGET = "is_outbreak"

def evaluate_features(df, features, target, name):
    """Run Stratified 5-Fold CV and return metrics."""
    X = df[features].values
    y = df[target].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    aucs = []
    auprcs = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)
        
        # Predict probabilities
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        
        aucs.append(auc)
        auprcs.append(auprc)
        
    return {
        "name": name,
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "auprc_mean": np.mean(auprcs),
        "auprc_std": np.std(auprcs),
        "feature_importances": clf.feature_importances_  # Last fold's importance for summary
    }

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Integrated dataset not found at: {DATASET_PATH}")
        print("Please run integrate_ner_features.py first.")
        sys.exit(1)

    print("Loading integrated dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # Use the full dataset since NER features are now active for all years (2009-2022)
    df_eval = df.copy()
    print(f"Total dataset rows: {len(df):,}")
    print(f"Active rows for evaluation: {len(df_eval):,}")
    
    # Check target class distribution
    outbreak_counts = df_eval[TARGET].value_counts()
    print(f"Outbreak distribution across all years:")
    for val, count in outbreak_counts.items():
        pct = count / len(df_eval) * 100
        print(f"  Class {val}: {count:,} ({pct:.1f}%)")

    if len(outbreak_counts) < 2:
        print("❌ Cannot perform evaluation: only one class present in target variable.")
        sys.exit(1)

    print("\nRunning Ablation Study using Stratified 5-Fold Cross-Validation...")
    
    # 1. Evaluate baseline features
    print("Evaluating Baseline Features...")
    baseline_results = evaluate_features(df_eval, BASELINE_FEATURES, TARGET, "Baseline")
    
    # 2. Evaluate enhanced features
    print("Evaluating NER-Enhanced Features...")
    enhanced_results = evaluate_features(df_eval, ENHANCED_FEATURES, TARGET, "NER-Enhanced")

    # ── Display Results ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS (predicting is_outbreak)")
    print("=" * 60)
    
    print(f"{'Metric':<15} | {'Baseline':<20} | {'NER-Enhanced':<20} | {'Improvement':<10}")
    print("-" * 75)
    
    auc_diff = enhanced_results["auc_mean"] - baseline_results["auc_mean"]
    print(f"{'ROC AUC':<15} | {baseline_results['auc_mean']:.4f} ± {baseline_results['auc_std']:.4f} | "
          f"{enhanced_results['auc_mean']:.4f} ± {enhanced_results['auc_std']:.4f} | "
          f"{auc_diff:+.4f}")
          
    auprc_diff = enhanced_results["auprc_mean"] - baseline_results["auprc_mean"]
    print(f"{'AUPRC':<15} | {baseline_results['auprc_mean']:.4f} ± {baseline_results['auprc_std']:.4f} | "
          f"{enhanced_results['auprc_mean']:.4f} ± {enhanced_results['auprc_std']:.4f} | "
          f"{auprc_diff:+.4f}")
    
    print("=" * 75)
    
    print("\nFeature Importances (NER-Enhanced Model):")
    sorted_idx = np.argsort(enhanced_results["feature_importances"])[::-1]
    for idx in sorted_idx:
        feat = ENHANCED_FEATURES[idx]
        imp = enhanced_results["feature_importances"][idx]
        is_ner = "[NER]" if feat in NER_FEATURES else "     "
        print(f"  {is_ner} {feat:<20}: {imp:.4f}")

if __name__ == "__main__":
    main()
