# models/training/train_tier1.py
import os
import time
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType


def train_and_export_tier1(
    data_path: str = "paysim_cleaned.parquet",
    model_output_path: str = "models/artifacts/xgboost.onnx"
):
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    print("[1/5] Loading cleaned dataset via Polars...")
    df = pl.read_parquet(data_path)

    # Define exact feature matrix for high signal & low memory footprint
    feature_cols = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "errorBalanceOrig",
        "errorBalanceDest",
        "is_transfer",
        "drain_ratio"
    ]
    target_col = "isFraud"

    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df.select(target_col).to_numpy().squeeze().astype(np.int32)

    # Chronological / Stratified Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Calculate scale_pos_weight to handle extreme class imbalance (~0.1% fraud)
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = num_neg / num_pos

    print(f"[2/5] Class Imbalance Ratio: {num_neg:,} neg / {num_pos:,} pos (Scale Weight: {scale_pos_weight:.2f})")

    print("[3/5] Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",  # High-speed histogram-based splitting
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate performance
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    print(f"\n[4/5] Evaluation Results:")
    print(f" • ROC-AUC Score: {roc_auc:.4f}")
    print(f" • PR-AUC Score: {pr_auc:.4f}")
    print(classification_report(y_test, (y_pred_proba > 0.85).astype(int), digits=4))

    # Export to ONNX for sub-15ms inference
    print("[5/5] Compiling and exporting to ONNX format...")
    initial_type = [('input', FloatTensorType([None, len(feature_cols)]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

    with open(model_output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"SUCCESS: Model successfully saved to {model_output_path}")


if __name__ == "__main__":
    train_and_export_tier1()