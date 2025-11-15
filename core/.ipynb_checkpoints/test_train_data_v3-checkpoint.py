# test_train_data_v3.py — Evaluate Hierarchical GMM on training data
import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

# ---------------- CONFIG ----------------
MODEL_PATH = "models/hier_gmm_model_v3_best.pkl"
CACHE_DIR = "cache"
RESULT_PATH = "results/train_data_eval_v3.csv"

# ---------------- UTILS ----------------
def load_features(path):
    data = np.load(path, allow_pickle=True)
    return data["features"], data["labels"], data["paths"]

def load_npz(path):
    return np.load(path, allow_pickle=True)["X"]

def fuse_features(X_res, X_freq, X_lbp, X_cnn):
    return np.hstack([X_res, X_freq, X_lbp, X_cnn])

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("🔹 Loading training data (features)...")

    # Load residual base
    X_res, y_train, _ = load_features("data/features/features_train.npz")

    # Load cached fused feature components
    X_freq = load_npz(os.path.join(CACHE_DIR, "train_freq.npz"))
    X_lbp = load_npz(os.path.join(CACHE_DIR, "train_lbp.npz"))
    X_cnn = load_npz(os.path.join(CACHE_DIR, "train_cnn.npz"))

    print(f"✅ Loaded feature shapes:")
    print(f"   Residual: {X_res.shape}")
    print(f"   Freq:     {X_freq.shape}")
    print(f"   LBP:      {X_lbp.shape}")
    print(f"   CNN:      {X_cnn.shape}")

    # Fuse all features
    print("🔹 Fusing Residual + Frequency + LBP + CNN...")
    X_train = fuse_features(X_res, X_freq, X_lbp, X_cnn)

    # Load trained Hierarchical GMM
    print("🔹 Loading trained model...")
    model_data = load(MODEL_PATH)
    coarse_gmm = model_data["coarse_gmm"]
    fine_gmm = model_data["fine_gmm"]
    threshold = model_data["threshold"]

    # Compute anomaly scores
    print("🔹 Scoring training samples...")
    scores = 0.5 * coarse_gmm.score_samples(X_train) + 0.5 * fine_gmm.score_samples(X_train)
    preds = (scores < threshold).astype(int)  # 0 = real, 1 = fake

    # Compute metrics
    acc = accuracy_score(y_train, preds)
    prec = precision_score(y_train, preds, zero_division=0)
    rec = recall_score(y_train, preds, zero_division=0)
    f1 = f1_score(y_train, preds, zero_division=0)
    auc = roc_auc_score(y_train, scores)

    # Summary counts
    total = len(y_train)
    real_pred = np.sum(preds == 0)
    fake_pred = np.sum(preds == 1)
    correct = np.sum(preds == y_train)
    correct_pct = (correct / total) * 100

    # Display summary
    print("\n📊 ---- TRAINING DATA EVALUATION ----")
    print(f"Total images: {total}")
    print(f"Real predicted: {real_pred}")
    print(f"Fake predicted: {fake_pred}")
    print(f"✅ Correct predictions: {correct} ({correct_pct:.2f}%)")
    print(f"\n[METRICS] acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}, auc={auc:.4f}")

    # Save CSV report
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    df = pd.DataFrame({
        "true_label": y_train,
        "pred_label": preds,
        "score": scores
    })
    df.to_csv(RESULT_PATH, index=False)
    print(f"📂 Results saved → {RESULT_PATH}")
