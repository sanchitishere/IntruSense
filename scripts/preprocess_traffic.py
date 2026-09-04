"""
Preprocessing pipeline for Deep-NIDS-TF.

Usage:
    # Use your own CICIDS2017 / NSL-KDD / UNSW-NB15 style CSV
    python scripts/preprocess_traffic.py --csv data/cicids2017_cleaned.csv --label-col Label

    # No dataset yet? Generate synthetic flow data to test the full pipeline end-to-end
    python scripts/preprocess_traffic.py --synthetic

Output (written to data/processed/):
    X_train.npy, X_val.npy, X_test.npy
    y_train.npy, y_val.npy, y_test.npy
    scaler.joblib, label_encoder.joblib
    feature_names.json
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def generate_synthetic_flows(n_benign=8000, n_attack=2000, n_features=20, seed=42):
    """
    Generates fake network-flow-style tabular data so the pipeline can be
    run and demoed without needing to download a real dataset first.
    Benign traffic: tight Gaussian clusters (normal behavior).
    Attack traffic: shifted/heavier-tailed distributions per attack type,
    mimicking how real attacks (scans, floods, brute force) look statistically
    different from benign flows.
    """
    rng = np.random.default_rng(seed)

    benign = rng.normal(loc=0.0, scale=1.0, size=(n_benign, n_features))
    benign_labels = np.array(["BENIGN"] * n_benign)

    attack_types = ["PortScan", "DDoS", "BruteForce", "Infiltration"]
    per_type = n_attack // len(attack_types)
    attack_chunks, attack_label_chunks = [], []
    for i, name in enumerate(attack_types):
        shift = (i + 1) * 2.5
        scale = 1.0 + i * 0.5
        chunk = rng.normal(loc=shift, scale=scale, size=(per_type, n_features))
        attack_chunks.append(chunk)
        attack_label_chunks.append(np.array([name] * per_type))

    X = np.vstack([benign] + attack_chunks)
    y = np.concatenate([benign_labels] + attack_label_chunks)

    feature_names = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["Label"] = y
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle


def load_and_clean(csv_path, label_col):
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Drop rows/cols that are entirely broken, replace inf with nan then drop
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="any")

    # Drop obvious non-feature / leakage columns if present
    drop_candidates = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "SimillarHTTP"]
    df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Columns: {list(df.columns)}")

    return df, label_col


def build_splits(df, label_col, test_size=0.2, val_size=0.1, seed=42):
    y_raw = df[label_col].astype(str).values
    X = df.drop(columns=[label_col])

    # Keep only numeric columns; one-hot encode any remaining categoricals
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = pd.get_dummies(X, columns=non_numeric, drop_first=True)

    feature_names = X.columns.tolist()
    X = X.values.astype(np.float32)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=seed, stratify=y
    )
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - rel_val), random_state=seed, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, X_val, X_test, y_train, y_val, y_test, scaler, label_encoder, feature_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to raw flow CSV dataset")
    parser.add_argument("--label-col", type=str, default="Label", help="Name of the label column")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic demo data instead")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.synthetic or args.csv is None:
        print("[preprocess] No real dataset provided -> generating synthetic flow data for a demo run.")
        df = generate_synthetic_flows()
        label_col = "Label"
    else:
        print(f"[preprocess] Loading {args.csv} ...")
        df, label_col = load_and_clean(args.csv, args.label_col)

    print(f"[preprocess] {len(df)} rows, label distribution:\n{df[label_col].value_counts()}")

    (X_train, X_val, X_test, y_train, y_val, y_test,
     scaler, label_encoder, feature_names) = build_splits(df, label_col)

    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)

    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(OUT_DIR, "label_encoder.joblib"))

    with open(os.path.join(OUT_DIR, "feature_names.json"), "w") as f:
        json.dump(
            {"feature_names": feature_names, "classes": label_encoder.classes_.tolist()},
            f, indent=2,
        )

    print(f"[preprocess] Done. Train/Val/Test shapes: "
          f"{X_train.shape}/{X_val.shape}/{X_test.shape}")
    print(f"[preprocess] Classes: {label_encoder.classes_.tolist()}")
    print(f"[preprocess] Artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
