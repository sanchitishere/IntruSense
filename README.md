# Deep-NIDS-TF: TensorFlow Network Intrusion Detection System

A hybrid deep-learning NIDS: a **supervised classifier** labels traffic by known
attack type, and a **benign-only autoencoder** flags statistically unusual
traffic the classifier was never trained on (novel/unknown attacks). A
Streamlit dashboard visualizes evaluation metrics and the live detection feed.

## Architecture

- `scripts/preprocess_traffic.py` — cleans, encodes, and scales raw flow data;
  can generate synthetic demo data if you don't have a real dataset yet.
- `src/model.py` — Keras model definitions (classifier + autoencoder).
- `src/train.py` — trains both models, saves checkpoints + training curves.
- `src/evaluate.py` — computes accuracy/precision/recall/F1, confusion matrix,
  and autoencoder detection/false-alarm rates; writes `evaluation_report.txt`.
- `src/detect.py` — simulated real-time inference: scores batches of flows,
  logs alerts to `logs/detections.jsonl`.
- `dashboard.py` — Streamlit dashboard over the report + live log.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Preprocess (use --synthetic for a demo, or --csv path/to/data.csv --label-col Label for real data)
python scripts/preprocess_traffic.py --synthetic

# 2. Train
python src/train.py --epochs 30 --batch-size 256

# 3. Evaluate
python src/evaluate.py

# 4. Run (simulated) real-time detection
python src/detect.py --batch-size 32 --interval 0.5

# 5. View the dashboard
streamlit run dashboard.py
```

## Using a real dataset

Download a cleaned CICIDS2017 / NSL-KDD / UNSW-NB15 CSV into `data/`, then:

```bash
python scripts/preprocess_traffic.py --csv data/your_dataset.csv --label-col Label
```

The rest of the pipeline (train/evaluate/detect/dashboard) is unchanged —
everything downstream just reads from `data/processed/`.

## Plugging in real live traffic

`src/detect.py` currently replays test-set rows to simulate a live stream.
To go fully real-time, replace the row-replay loop with a queue fed by a
packet-capture + flow-aggregation process (e.g. `scapy` + a CICFlowMeter-style
flow exporter), pushing feature rows through the same `scaler.transform()` →
`clf.predict()` / `ae.predict()` path.

## Optimization ideas to try next

- Export the classifier to TensorFlow Lite (`tf.lite.TFLiteConverter`) or
  ONNX for lower-latency inference, especially on CPU-only deployment targets.
- Try XGBoost/LightGBM as a baseline against the classifier — tree ensembles
  are often competitive on tabular flow features and train much faster.
- Add concept-drift monitoring: periodically compare live feature
  distributions against the training distribution and alert when the
  autoencoder's false-alarm rate drifts upward.
- Swap `StandardScaler` for feature selection/PCA if you add many more raw
  columns from a larger dataset, to keep inference latency low.
