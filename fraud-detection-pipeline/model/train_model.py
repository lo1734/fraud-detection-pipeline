"""
PaySim Fraud Detection — XGBoost Model Training Script
=======================================================
Trains an XGBoost classifier on the PaySim mobile‑money dataset and exports
a serialized model + feature config for deployment inside a Lambda layer.

Usage:
    cd fraud-detection-pipeline/model
    pip install -r requirements-training.txt
    python train_model.py                         # default: data/paysim.csv
    python train_model.py --data path/to/file.csv # custom path
"""

import argparse
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
# Only TRANSFER and CASH_OUT contain fraud — filter the rest to save memory
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# Feature names in the exact order the model expects at inference time.
# The scoring Lambda must construct a vector in this same order.
FEATURE_NAMES = [
    # --- PaySim core features ---
    "type_encoded",          # LabelEncoded transaction type
    "amount",                # Raw transaction amount
    "oldbalanceOrg",         # Sender balance before txn
    "newbalanceOrig",        # Sender balance after txn
    "oldbalanceDest",        # Receiver balance before txn
    "newbalanceDest",        # Receiver balance after txn
    # --- Engineered features ---
    "balance_delta_orig",    # oldbalanceOrg - newbalanceOrig
    "balance_delta_dest",    # newbalanceDest - oldbalanceDest
    "error_balance_orig",    # newbalanceOrig + amount - oldbalanceOrg (should be 0 for legit)
    "error_balance_dest",    # oldbalanceDest + amount - newbalanceDest (should be 0 for legit)
    "amount_ratio_orig",     # amount / (oldbalanceOrg + 1)  — how much of balance is being sent
    "amount_ratio_dest",     # amount / (oldbalanceDest + 1) — relative to receiver balance
    "is_orig_balance_zero",  # newbalanceOrig == 0 after txn (drain flag)
    "is_dest_balance_zero",  # oldbalanceDest == 0 before txn
    "hour_of_day",           # step % 24  — circadian cycle feature
    "day_of_month",          # step // 24 — day feature
    # --- Graph‑bridge features (mapped from pipeline at inference) ---
    "sender_out_degree",
    "sender_avg_out_amount",
    "sender_counterparty_diversity",
    "receiver_in_degree",
    "receiver_avg_in_amount",
    "pair_tx_count",
    "pair_total_volume",
    "structuring_cluster_count",
]


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw PaySim columns."""

    # Encode transaction type to int
    le = LabelEncoder()
    df["type_encoded"] = le.fit_transform(df["type"])

    # Balance deltas
    df["balance_delta_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Balance‑error features (powerful: should be ~0 for legit transactions)
    df["error_balance_orig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["error_balance_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Ratio features
    df["amount_ratio_orig"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    df["amount_ratio_dest"] = df["amount"] / (df["oldbalanceDest"] + 1.0)

    # Binary flags
    df["is_orig_balance_zero"] = (df["newbalanceOrig"] == 0.0).astype(int)
    df["is_dest_balance_zero"] = (df["oldbalanceDest"] == 0.0).astype(int)

    # Temporal features
    df["hour_of_day"] = df["step"] % 24
    df["day_of_month"] = df["step"] // 24

    # Graph‑bridge features: set to neutral defaults during training.
    # At inference these will be populated from the DynamoDB graph queries.
    df["sender_out_degree"] = 0.0
    df["sender_avg_out_amount"] = 0.0
    df["sender_counterparty_diversity"] = 1.0
    df["receiver_in_degree"] = 0.0
    df["receiver_avg_in_amount"] = 0.0
    df["pair_tx_count"] = 0.0
    df["pair_total_volume"] = 0.0
    df["structuring_cluster_count"] = 0.0

    # Save the label encoder classes for inference‑time mapping
    type_mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}

    return df, type_mapping


# ---------------------------------------------------------------------------
# 3. MAIN TRAINING PIPELINE
# ---------------------------------------------------------------------------
def train(data_path: str) -> None:
    print("=" * 70)
    print("  PaySim Fraud Detection — XGBoost Training Pipeline")
    print("=" * 70)

    # ---- Load ----
    t0 = time.time()
    print(f"\n[1/7] Loading dataset from {data_path} …")
    df = pd.read_csv(data_path)
    print(f"      Loaded {len(df):,} rows × {len(df.columns)} columns in {time.time()-t0:.1f}s")

    # ---- Filter to fraud‑eligible types ----
    print(f"\n[2/7] Filtering to fraud-eligible types: {FRAUD_TYPES}")
    df = df[df["type"].isin(FRAUD_TYPES)].copy()
    n_fraud = df["isFraud"].sum()
    n_legit = len(df) - n_fraud
    print(f"      {len(df):,} transactions remaining  |  Fraud: {n_fraud:,} ({n_fraud/len(df)*100:.3f}%)  |  Legit: {n_legit:,}")

    # ---- Engineer features ----
    print("\n[3/7] Engineering features …")
    df, type_mapping = engineer_features(df)

    X = df[FEATURE_NAMES].values
    y = df["isFraud"].values

    # ---- Split ----
    print(f"\n[4/7] Splitting: {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test (stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ---- SMOTE ----
    print("\n[5/7] Applying SMOTE oversampling on training set …")
    t1 = time.time()
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.1)  # 10% minority ratio
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"      Resampled: {len(X_train_res):,} samples ({y_train_res.sum():,} fraud) in {time.time()-t1:.1f}s")

    # ---- Train XGBoost ----
    print("\n[6/7] Training XGBoost classifier …")
    t2 = time.time()

    # scale_pos_weight as additional imbalance handling alongside SMOTE
    neg, pos = np.bincount(y_train_res)
    scale_pos = neg / pos

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos,
        reg_alpha=1.0,
        reg_lambda=2.0,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        use_label_encoder=False,
    )

    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    print(f"      Training completed in {time.time()-t2:.1f}s")

    # ---- Evaluate ----
    print("\n[7/7] Evaluating on test set …")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"  AUC‑ROC:             {auc_roc:.4f}")
    print(f"  AUC‑PR (Avg Prec):   {auc_pr:.4f}")
    print(f"  F1 Score (fraud):    {f1:.4f}")
    print(f"{'─'*50}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # Feature importances
    print("\nTop 10 Feature Importances:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:35s}  {importances[idx]:.4f}")

    # ---- Save artifacts ----
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    model_path = os.path.join(ARTIFACTS_DIR, "fraud_model.joblib")
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to {model_path}  ({os.path.getsize(model_path)/1024/1024:.1f} MB)")

    # Save native XGBoost Booster JSON for lightweight runtime inference (no scikit-learn required)
    json_model_path = os.path.join(ARTIFACTS_DIR, "fraud_model.json")
    model.get_booster().save_model(json_model_path)
    print(f"✅ Booster saved to {json_model_path}  ({os.path.getsize(json_model_path)/1024/1024:.1f} MB)")

    # Feature config for the scoring Lambda
    feature_config = {
        "feature_names": FEATURE_NAMES,
        "type_mapping": type_mapping,
        "threshold": 0.50,  # Classification threshold (tunable)
        "ensemble_weight_ml": 0.70,
        "ensemble_weight_graph": 0.30,
        "metrics": {
            "auc_roc": round(auc_roc, 4),
            "auc_pr": round(auc_pr, 4),
            "f1_fraud": round(f1, 4),
            "test_size": len(X_test),
            "train_fraud_count": int(y_train_res.sum()),
        },
    }

    config_path = os.path.join(ARTIFACTS_DIR, "feature_config.json")
    with open(config_path, "w") as f:
        json.dump(feature_config, f, indent=2, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else float(o) if isinstance(o, (np.floating, np.float64)) else str(o))
    print(f"✅ Feature config saved to {config_path}")

    print(f"\n{'='*70}")
    print("  Training complete! Next steps:")
    print("  1. Run: bash model/build_layer.sh")
    print("  2. Run: sam build && sam deploy")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# 4. CLI ENTRYPOINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PaySim fraud detection model")
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "data", "paysim.csv"),
        help="Path to the PaySim CSV file (default: model/data/paysim.csv)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"❌ Dataset not found at: {args.data}")
        print("   Download from: https://www.kaggle.com/datasets/mtalaltariq/paysim-data/data")
        print(f"   Place the CSV at: {args.data}")
        exit(1)

    train(args.data)
