# eval_gmm_v42.py - Evaluate trained GMM v42 model
import numpy as np, json
from joblib import load
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from skimage.io import imread
from skimage.color import rgb2gray
from tqdm import tqdm
from train_gmm_v42 import (
    extract_frequency_features,
    extract_lbp_hist,
    extract_cnn_features_adaptive,
    fuse_features_by_paths,
    load_features
)

MODEL_PATH = "models/hier_gmm_model_v42.pkl"
RESULTS_PATH = "results/eval_gmm_v42.json"
CACHE_DIR = "cache_v42"
DEVICE = "cuda"

# ---------------- Load model ----------------
model_bundle = load(MODEL_PATH)
gmm = model_bundle["gmm"]
lr = model_bundle["lr"]

# ---------------- Load test data ----------------
X_test_res, y_test, test_paths = load_features("data/features/features_test.npz")

# --- Load cached features (Frequency, LBP, CNN) ---
X_test_freq = np.load(f"{CACHE_DIR}/test_freq_v4.npz")["X"]
X_test_lbp = np.load(f"{CACHE_DIR}/test_lbp.npz")["X"]
X_test_cnn = np.load(f"{CACHE_DIR}/test_cnn.npz")["X"]

# ---------------- Fuse all features ----------------
X_test_fused, y_test_aligned = fuse_features_by_paths(
    X_test_res, X_test_freq, X_test_lbp, X_test_cnn,
    test_paths, test_paths, test_paths, test_paths, y_test
)

# ---------------- Evaluate ----------------
test_scores = -gmm.score_samples(X_test_fused)
test_pred = lr.predict(test_scores.reshape(-1, 1))
test_proba = lr.predict_proba(test_scores.reshape(-1, 1))[:, 1]

# Compute metrics
acc = accuracy_score(y_test_aligned, test_pred)
prec = precision_score(y_test_aligned, test_pred, zero_division=0)
rec = recall_score(y_test_aligned, test_pred, zero_division=0)
f1 = f1_score(y_test_aligned, test_pred, zero_division=0)
auc = roc_auc_score(y_test_aligned, test_proba)

results = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "auc": auc
}

print("✅ Evaluation Results:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)
print(f"💾 Saved evaluation to {RESULTS_PATH}")
