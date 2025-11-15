# test_train_data_v4.py — Evaluate Hybrid GMM + LR Fusion (v4)
import os
import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ---------------- CONFIG ----------------
MODEL_PATH = "models/hier_gmm_model_v4.pkl"
RESULT_PATH = "results/train_data_eval_v4.csv"
CACHE_DIR = "cache_v4"
DATA_PATH = "data/features/features_train.npz"

# ---------------- UTILS ----------------
def load_features(path):
    data = np.load(path, allow_pickle=True)
    return data["features"], data["labels"], data["paths"]

def load_npz(path):
    return np.load(path, allow_pickle=True)["X"]

def zscore(arr):
    return (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-9)

def fuse_features(X_res, X_freq, X_lbp, X_cnn):
    X_res, X_freq, X_lbp, X_cnn = map(zscore, [X_res, X_freq, X_lbp, X_cnn])
    return np.hstack([X_res, X_freq, X_lbp, X_cnn])

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("🔹 Loading cached training features...")

    # Base residual features
    X_res, y_train, _ = load_features(DATA_PATH)

    # Cached features from v4 training
    X_freq = load_npz(os.path.join(CACHE_DIR, "train_freq_v4.npz"))
    X_lbp  = load_npz(os.path.join(CACHE_DIR, "train_lbp.npz"))
    X_cnn  = load_npz(os.path.join(CACHE_DIR, "train_cnn.npz"))

    print(f"✅ Loaded:")
    print(f"   Residual: {X_res.shape}")
    print(f"   Freq:     {X_freq.shape}")
    print(f"   LBP:      {X_lbp.shape}")
    print(f"   CNN:      {X_cnn.shape}")

    print("🔹 Fusing features with z-score normalization...")
    X_train = fuse_features(X_res, X_freq, X_lbp, X_cnn)

    print("🔹 Loading trained model...")
    model_data = load(MODEL_PATH)
    gmm = model_data["gmm"]
    lr_model = model_data["lr"]

    print("🔹 Computing scores and predictions...")
    scores = -gmm.score_samples(X_train)
    preds = lr_model.predict(scores.reshape(-1, 1))

    # ---------------- METRICS ----------------
    acc = accuracy_score(y_train, preds)
    prec = precision_score(y_train, preds, zero_division=0)
    rec = recall_score(y_train, preds, zero_division=0)
    f1 = f1_score(y_train, preds, zero_division=0)
    auc = roc_auc_score(y_train, -scores)

    total = len(y_train)
    correct = np.sum(preds == y_train)
    print("\n📊 ---- TRAINING DATA EVALUATION ----")
    print(f"Total samples: {total}")
    print(f"✅ Correct: {correct} ({(correct/total)*100:.2f}%)")
    print(f"[METRICS] acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, auc={auc:.4f}")

    # ---------------- SAVE RESULTS ----------------
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    df = pd.DataFrame({
        "true_label": y_train,
        "pred_label": preds,
        "score": scores
    })
    df.to_csv(RESULT_PATH, index=False)
    print(f"📂 Results saved → {RESULT_PATH}")
