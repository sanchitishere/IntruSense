"""
Simulated real-time inference engine.

In a real deployment, a separate capture/flow-aggregation process (e.g. an
scapy sniffer feeding a flow exporter like CICFlowMeter) would push rows of
extracted features into a queue that this script drains in small batches.
Here, to keep the project runnable end-to-end without a live network tap,
we simulate that stream by replaying rows from the test set in batches,
which is exactly how you would plug in a real feature-extraction feed later.

Usage:
    python src/detect.py --batch-size 32 --interval 0.5
    python src/detect.py --benign-label "Normal Traffic"
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "detections.jsonl")

# Common names datasets use for the "no attack" class. If your dataset uses
# something else, pass --benign-label explicitly rather than editing this.
BENIGN_ALIASES = {"BENIGN", "Normal Traffic", "Normal", "normal", "benign"}


def resolve_benign_label(class_names, explicit_label=None):
    if explicit_label is not None:
        if explicit_label not in class_names:
            raise ValueError(f"--benign-label '{explicit_label}' not found in classes {class_names}")
        return explicit_label

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
    return matches[0]


def load_artifacts():
    clf = tf.keras.models.load_model(os.path.join(MODELS_DIR, "classifier.keras"))
    ae = tf.keras.models.load_model(os.path.join(MODELS_DIR, "autoencoder.keras"))
    with open(os.path.join(MODELS_DIR, "ae_threshold.json")) as f:
        threshold = json.load(f)["threshold"]
    with open(os.path.join(DATA_DIR, "feature_names.json")) as f:
        meta = json.load(f)
    return clf, ae, threshold, meta["classes"]


def score_batch(clf, ae, threshold, class_names, benign_label, X_batch):
    probs = clf.predict(X_batch, verbose=0)
    pred_idx = np.argmax(probs, axis=1)
    pred_conf = np.max(probs, axis=1)

    recon = ae.predict(X_batch, verbose=0)
    recon_error = np.mean(np.square(X_batch - recon), axis=1)
    is_anomalous = recon_error > threshold

    results = []
    for i in range(len(X_batch)):
        label = class_names[pred_idx[i]]
        verdict = benign_label
        if label != benign_label:
            verdict = f"ATTACK:{label}"
        elif is_anomalous[i]:
            # classifier says benign but reconstruction error is high ->
            # possible unknown/novel attack the classifier wasn't trained on
            verdict = "ANOMALY:unclassified"

        results.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predicted_class": label,
            "confidence": float(pred_conf[i]),
            "reconstruction_error": float(recon_error[i]),
            "verdict": verdict,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between batches")
    parser.add_argument("--limit-batches", type=int, default=20)
    parser.add_argument("--benign-label", type=str, default=None,
                         help="Exact class name for benign/normal traffic, "
                              "if it's not one of the common aliases.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    clf, ae, threshold, class_names = load_artifacts()
    benign_label = resolve_benign_label(class_names, args.benign_label)
    print(f"[detect] Using '{benign_label}' as the benign class")

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    print(f"[detect] Streaming {len(X_test)} flows in batches of {args.batch_size} "
          f"(simulated live traffic)...")

    n_batches = min(args.limit_batches, (len(X_test) + args.batch_size - 1) // args.batch_size)
    alert_count = 0

    with open(LOG_PATH, "a") as log_file:
        for b in range(n_batches):
            start = b * args.batch_size
            end = start + args.batch_size
            batch = X_test[start:end]
            if len(batch) == 0:
                break

            results = score_batch(clf, ae, threshold, class_names, benign_label, batch)
            for r in results:
                log_file.write(json.dumps(r) + "\n")
                if r["verdict"] != benign_label:
                    alert_count += 1
                    print(f"[ALERT] {r['timestamp']} verdict={r['verdict']} "
                          f"confidence={r['confidence']:.3f} recon_err={r['reconstruction_error']:.5f}")
            log_file.flush()
            time.sleep(args.interval)

    print(f"[detect] Done. {alert_count} alerts logged to {LOG_PATH}")


if __name__ == "__main__":
    main()