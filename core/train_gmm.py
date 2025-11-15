# train_gmm.py

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
from tqdm import tqdm
import os, json, datetime
from joblib import dump

# -----------------------
# Utils
# -----------------------
def load_features(path):
    data = np.load(path, allow_pickle=True)
    return data['features'], data['labels'], data['paths']

def compute_metrics(labels, preds, scores):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, -scores)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

def evaluate_with_threshold(gmm, feats, labels, threshold):
    scores = gmm.score_samples(feats)
    preds = (scores < threshold).astype(int)  # 0=real, 1=fake
    return compute_metrics(labels, preds, scores)

def save_results(results, out_path="results/gmm_results.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            history = json.load(f)
    else:
        history = []
    history.append(results)
    with open(out_path, "w") as f:
        json.dump(history, f, indent=4)

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # Safety: limit BLAS threads
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
    os.environ["NUMEXPR_NUM_THREADS"] = "8"

    print("🔹 Loading features...")
    X_train, y_train, _ = load_features("data/features/features_train.npz")
    X_val, y_val, _   = load_features("data/features/features_val.npz")
    X_test, y_test, _ = load_features("data/features/features_test.npz")

    X_train_real = X_train[y_train == 0]
    print(f"Train set (real only): {X_train_real.shape}")
    print(f"Validation set: {X_val.shape}, Test set: {X_test.shape}")

    # Subsample real train if too large
    if len(X_train_real) > 50000:
        print(f"⚠️ Too many samples ({len(X_train_real)}). Subsampling 50k for GMM training...")
        idx = np.random.choice(len(X_train_real), 50000, replace=False)
        X_train_real = X_train_real[idx]

    # Fit GMM
    print("\n🔹 Training GMM...")
    gmm = GaussianMixture(
        n_components=4, covariance_type='diag', random_state=42, max_iter=200, verbose=2
    )
    gmm.fit(X_train_real)

    # Validation scores
    print("\n🔹 Tuning threshold on validation set...")
    scores_val = gmm.score_samples(X_val)

    thresholds = np.linspace(scores_val.min(), scores_val.max(), 200)
    best_thresh, best_f1, best_metrics = None, -1, None

    for t in tqdm(thresholds, desc="Searching thresholds"):
        preds = (scores_val < t).astype(int)
        m = compute_metrics(y_val, preds, scores_val)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = t
            best_metrics = m

    print(f"\n✅ Best threshold = {best_thresh:.4f} (F1={best_f1:.4f})")
    print("[Validation Metrics @best threshold]")
    for k,v in best_metrics.items():
        print(f" {k}: {v:.4f}")

    # Evaluate on test
    print("\n🔹 Evaluating on Test set...")
    scores_test = gmm.score_samples(X_test)
    preds_test = (scores_test < best_thresh).astype(int)
    test_metrics = compute_metrics(y_test, preds_test, scores_test)

    print("[Test Metrics]")
    for k,v in test_metrics.items():
        print(f" {k}: {v:.4f}")

    # Save model + threshold
    os.makedirs("models", exist_ok=True)
    dump({"gmm": gmm, "threshold": best_thresh}, "models/gmm_model.pkl")
    print("\n✅ Model + threshold saved to models/gmm_model.pkl")

    # -----------------------
    # Save results to JSON
    # -----------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {
        "timestamp": timestamp,
        "train_samples": int(X_train_real.shape[0]),
        "threshold": float(best_thresh),
        "val_metrics": {k: float(v) for k,v in best_metrics.items()},
        "test_metrics": {k: float(v) for k,v in test_metrics.items()}
    }
    save_results(results)
    print("📂 Results appended to results/gmm_results.json")
