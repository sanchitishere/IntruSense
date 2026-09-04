"""
Evaluates the trained classifier + autoencoder on the held-out test set.
Writes a human-readable report to evaluation_report.txt and a
confusion-matrix heatmap image.

Usage:
    python src/evaluate.py
    python src/evaluate.py --benign-label "Normal Traffic"
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "evaluation_report.txt")

# Common names datasets use for the "no attack" class. If your dataset uses
# something else, pass --benign-label explicitly rather than editing this.
BENIGN_ALIASES = {"BENIGN", "Normal Traffic", "Normal", "normal", "benign"}


def resolve_benign_idx(class_names, explicit_label=None):
    if explicit_label is not None:
        if explicit_label not in class_names:
            raise ValueError(f"--benign-label '{explicit_label}' not found in classes {class_names}")
        return class_names.index(explicit_label)

    matches = [c for c in class_names if c in BENIGN_ALIASES]
    if not matches:
        raise ValueError(
            f"Couldn't identify the benign class among {class_names}. "
            f"Pass --benign-label \"<exact class name>\" explicitly."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple classes matched benign aliases: {matches}. "
            f"Pass --benign-label \"<exact class name>\" explicitly to disambiguate."
        )
    return class_names.index(matches[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benign-label", type=str, default=None,
                        help="Exact class name for benign/normal traffic, "
                             "if it's not one of the common aliases.")
    args = parser.parse_args()

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    with open(os.path.join(DATA_DIR, "feature_names.json")) as f:
        meta = json.load(f)
    class_names = meta["classes"]

    clf = tf.keras.models.load_model(os.path.join(MODELS_DIR, "classifier.keras"))
    probs = clf.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = float(np.mean(y_pred == y_test))
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report_text = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # ---- autoencoder-side evaluation: does it flag non-benign as anomalous? ----
    ae_note = ""
    ae_model_path = os.path.join(MODELS_DIR, "autoencoder.keras")
    threshold_path = os.path.join(MODELS_DIR, "ae_threshold.json")
    if os.path.exists(ae_model_path) and os.path.exists(threshold_path):
        ae = tf.keras.models.load_model(ae_model_path)
        with open(threshold_path) as f:
            threshold = json.load(f)["threshold"]
        recon = ae.predict(X_test, verbose=0)
        errors = np.mean(np.square(X_test - recon), axis=1)
        ae_flag = errors > threshold

        benign_idx = resolve_benign_idx(class_names, args.benign_label)
        is_attack = y_test != benign_idx
        ae_detect_rate = float(np.mean(ae_flag[is_attack])) if is_attack.any() else float("nan")
        ae_false_alarm_rate = float(np.mean(ae_flag[~is_attack])) if (~is_attack).any() else float("nan")
        ae_note = (
            f"\nAutoencoder anomaly detector (threshold={threshold:.6f}, "
            f"benign class='{class_names[benign_idx]}'):\n"
            f"  Detection rate on true attack flows : {ae_detect_rate:.4f}\n"
            f"  False alarm rate on benign flows      : {ae_false_alarm_rate:.4f}\n"
        )

    with open(REPORT_PATH, "w") as f:
        f.write("Deep-NIDS-TF Evaluation Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Test set size: {len(y_test)}\n\n")
        f.write(f"Overall accuracy : {acc:.4f}\n")
        f.write(f"Macro precision  : {precision:.4f}\n")
        f.write(f"Macro recall     : {recall:.4f}\n")
        f.write(f"Macro F1         : {f1:.4f}\n\n")
        f.write("Per-class report:\n")
        f.write(report_text + "\n")
        if ae_note:
            f.write(ae_note)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))
    plt.close()

    print(f"[evaluate] accuracy={acc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    print(f"[evaluate] Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()