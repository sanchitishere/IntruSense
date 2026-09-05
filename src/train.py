"""
Trains both the supervised classifier and the benign-only autoencoder,
saves checkpoints to models/, and plots loss curves.

Usage:
    python src/train.py --epochs 30 --batch-size 256
    python src/train.py --epochs 30 --benign-label "Normal Traffic"
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import build_autoencoder, build_classifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

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


def load_split():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    with open(os.path.join(DATA_DIR, "feature_names.json")) as f:
        meta = json.load(f)
    return X_train, X_val, y_train, y_val, meta


def class_weights_from_labels(y, max_weight=15.0):
    """Inverse-frequency class weights, capped to prevent extreme
    multipliers on very rare classes from causing over-prediction
    (high recall but collapsed precision)."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
    return {k: min(v, max_weight) for k, v in weights.items()}


def plot_history(history, out_path, title):
    plt.figure(figsize=(7, 4))
    for key in history.history:
        if "val" not in key:
            plt.plot(history.history[key], label=key)
            val_key = f"val_{key}"
            if val_key in history.history:
                plt.plot(history.history[val_key], label=val_key, linestyle="--")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--benign-label", type=str, default=None,
                         help="Exact class name for benign/normal traffic, "
                              "if it's not one of the common aliases.")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_val, y_train, y_val, meta = load_split()
    num_classes = len(meta["classes"])
    input_dim = X_train.shape[1]
    print(f"[train] input_dim={input_dim}, num_classes={num_classes}, classes={meta['classes']}")

    # ---------- 1. Supervised classifier ----------
    clf = build_classifier(input_dim, num_classes)
    cw = class_weights_from_labels(y_train)
    print(f"[train] class weights: {cw}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "classifier.keras"), save_best_only=True, monitor="val_loss"
        ),
    ]

    history = clf.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=cw,
        callbacks=callbacks,
        verbose=2,
    )
    plot_history(history, os.path.join(MODELS_DIR, "classifier_training_curve.png"), "Classifier training")

    # ---------- 2. Autoencoder (trained on benign traffic only) ----------
    benign_class_idx = resolve_benign_idx(meta["classes"], args.benign_label)
    print(f"[train] Using '{meta['classes'][benign_class_idx]}' as the benign class "
          f"(index {benign_class_idx})")

    benign_mask_train = y_train == benign_class_idx
    benign_mask_val = y_val == benign_class_idx

    X_train_benign = X_train[benign_mask_train]
    X_val_benign = X_val[benign_mask_val]
    print(f"[train] Autoencoder training on {len(X_train_benign)} benign flows "
          f"(val: {len(X_val_benign)})")

    ae = build_autoencoder(input_dim)
    ae_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "autoencoder.keras"), save_best_only=True, monitor="val_loss"
        ),
    ]
    ae_history = ae.fit(
        X_train_benign, X_train_benign,
        validation_data=(X_val_benign, X_val_benign),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=ae_callbacks,
        verbose=2,
    )
    plot_history(ae_history, os.path.join(MODELS_DIR, "autoencoder_training_curve.png"), "Autoencoder training")

    # Compute a reconstruction-error threshold from benign validation data
    # (mean + 2*std is a common heuristic; tune this against your own data).
    recon = ae.predict(X_val_benign, verbose=0)
    errors = np.mean(np.square(X_val_benign - recon), axis=1)
    threshold = float(np.mean(errors) + 2 * np.std(errors))
    with open(os.path.join(MODELS_DIR, "ae_threshold.json"), "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    print(f"[train] Autoencoder anomaly threshold set to {threshold:.6f}")

    print("[train] Done. Models saved to", MODELS_DIR)


if __name__ == "__main__":
    main()