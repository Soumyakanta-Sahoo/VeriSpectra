# train_gmm_v1.py

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
from tqdm import tqdm
import os, json, datetime
from joblib import dump, load 

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
    # ROC AUC: scores < threshold means Fake (label 1). Lower score is more Fake.
    # roc_auc_score expects higher scores for the positive class (1). 
    # Since our scores are lower for class 1, we pass -scores.
    auc = roc_auc_score(labels, -scores) 
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

def evaluate_with_threshold(gmm, feats, labels, threshold):
    scores = gmm.score_samples(feats)
    preds = (scores < threshold).astype(int)  # 0=real, 1=fake
    return compute_metrics(labels, preds, scores)

def save_results(results, out_path="results/gmm_results_v1.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            history = json.load(f)
    else:
        history = []
    history.append(results)
    with open(out_path, "w") as f:
        json.dump(history, f, indent=4)

def load_previous_best_f1(results_path="results/gmm_results_v1.json"):
    """Loads the best F1 score ever recorded in the history file."""
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                history = json.load(f)
                if history:
                    best_f1_in_history = -1.0
                    for entry in history:
                        if 'test_metrics' in entry and 'f1' in entry['test_metrics']:
                             best_f1_in_history = max(best_f1_in_history, entry['test_metrics']['f1'])
                    return best_f1_in_history
        except json.JSONDecodeError:
            print("⚠️ Could not read results history. Starting comparison from F1=-1.0.")
    return -1.0 
    
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

    # Validation scores (tuning threshold)
    print("\n🔹 Tuning threshold on validation set...")
    scores_val = gmm.score_samples(X_val)

    thresholds = np.linspace(scores_val.min(), scores_val.max(), 200)
    best_thresh, best_f1, best_metrics = None, -1.0, None

    # *** REVISED TUNING STRATEGY ***
    # Goal: Optimize F1, but only for thresholds that maintain a minimum Precision.
    # We want at least 95% Precision (0.95) to significantly reduce FPs.
    MIN_PRECISION_FLOOR = 0.95 

    print(f"Goal: Maximize F1 with MINIMUM Precision of {MIN_PRECISION_FLOOR:.2f}")

    for t in tqdm(thresholds, desc="Searching thresholds"):
        preds = (scores_val < t).astype(int)
        m = compute_metrics(y_val, preds, scores_val)
        
        # Check if Precision meets the minimum floor
        if m["prec"] >= MIN_PRECISION_FLOOR:
            # If it meets the floor, check if the F1 score is better than the current best
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_thresh = t
                best_metrics = m

    # Check if a valid threshold was found
    if best_thresh is None:
        print(f"❌ Could not find a threshold that meets the minimum precision requirement of {MIN_PRECISION_FLOOR}.")
        # Fallback to the best F1 model found historically
        current_f1 = -1.0 
        best_thresh = scores_val.min() # Set to a non-null placeholder
        test_metrics = {'acc': 0, 'prec': 0, 'rec': 0, 'f1': 0, 'auc': 0} # Set to null metrics
    else:
        # Evaluate on test with the newly found constrained threshold
        print(f"\n✅ Best constrained threshold = {best_thresh:.4f} (F1={best_f1:.4f}, Prec={best_metrics['prec']:.4f})")
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
        current_f1 = test_metrics['f1']


    # --- Model Comparison and Save Logic (Preserving the best model) ---
    previous_best_f1 = load_previous_best_f1()
    
    print("\n--- Model Comparison ---")
    print(f"Current Model Test F1: {current_f1:.4f}")
    print(f"Best F1 in History: {previous_best_f1:.4f}")

    # Only save if the new constrained F1 is better than the historical best F1
    if current_f1 > previous_best_f1:
        print("🎉 New model is BETTER! Overwriting saved model.")
        
        # Save model + threshold
        os.makedirs("models", exist_ok=True)
        dump({"gmm": gmm, "threshold": best_thresh}, "models/gmm_model_v1.pkl")
        print("✅ Model + threshold saved to models/gmm_model_v1.pkl")

    else:
        print("📉 Current model is NOT better than the previous best. Model NOT saved.")

    # Save results to JSON history file (always save the history)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {
        "timestamp": timestamp,
        "train_samples": int(X_train_real.shape[0]),
        "threshold": float(best_thresh) if best_thresh is not None else float('nan'),
        "val_metrics": {k: float(v) for k,v in best_metrics.items()} if best_metrics is not None else {},
        "test_metrics": {k: float(v) for k,v in test_metrics.items()}
    }
    save_results(results)
    print("📂 Results appended to results/gmm_results_v1.json")