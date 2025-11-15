# train_ocsvm.py

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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

def save_results(results, out_path="results/ocsvm_results.json"):
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
    print("🔹 Loading features...")
    X_train, y_train, _ = load_features("data/features/features_train.npz")
    X_val, y_val, _   = load_features("data/features/features_val.npz")
    X_test, y_test, _ = load_features("data/features/features_test.npz")

    X_train_real = X_train[y_train == 0]
    print(f"Train set (real only): {X_train_real.shape}")
    print(f"Validation set: {X_val.shape}, Test set: {X_test.shape}")

    # Subsample train if too large
    if len(X_train_real) > 50000:
        print(f"⚠️ Too many samples ({len(X_train_real)}). Subsampling 50k...")
        idx = np.random.choice(len(X_train_real), 50000, replace=False)
        X_train_real = X_train_real[idx]

    # Train One-Class SVM
    print("\n🔹 Training One-Class SVM...")
    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    clf.fit(X_train_real)

    # Compute decision scores
    print("\n🔹 Tuning threshold on validation set...")
    val_scores = clf.decision_function(X_val)  # higher = more normal
    thresholds = np.linspace(val_scores.min(), val_scores.max(), 200)
    best_thresh, best_f1, best_metrics = None, -1, None

    for t in tqdm(thresholds, desc="Searching thresholds"):
        preds = (val_scores < t).astype(int)  # anomaly if below threshold
        m = compute_metrics(y_val, preds, val_scores)
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
    test_scores = clf.decision_function(X_test)
    test_preds = (test_scores < best_thresh).astype(int)
    test_metrics = compute_metrics(y_test, test_preds, test_scores)

    print("[Test Metrics]")
    for k,v in test_metrics.items():
        print(f" {k}: {v:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    dump({"ocsvm": clf, "threshold": best_thresh}, "models/ocsvm_model.pkl")
    print("\n✅ Model + threshold saved to models/ocsvm_model.pkl")

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {
        "timestamp": timestamp,
        "train_samples": int(X_train_real.shape[0]),
        "threshold": float(best_thresh),
        "val_metrics": {k: float(v) for k,v in best_metrics.items()},
        "test_metrics": {k: float(v) for k,v in test_metrics.items()}
    }
    save_results(results)
    print("📂 Results appended to results/ocsvm_results.json")
