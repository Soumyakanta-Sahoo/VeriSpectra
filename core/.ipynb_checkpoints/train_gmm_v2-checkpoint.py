# train_gmm_v2.py - Single Training and Best Model Saving

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)
from tqdm import tqdm
import os, json, datetime, sys
from joblib import dump
from copy import deepcopy

# --- CONFIGURATION ---
NUM_TRIALS = 1                   # ✅ Only one trial
MODEL_PREFIX = "models/gmm_model_v2"
RESULTS_PATH = "results/gmm_results_v2.json"
MIN_PRECISION_FLOOR = 0.97
SUBSAMPLE_LIMIT = 50000         # Cap training samples for speed/memory
# -----------------------

# -----------------------
# Utility Functions
# -----------------------
def load_features(path):
    data = np.load(path, allow_pickle=True)
    return data['features'], data['labels'], data['paths']

def compute_metrics(labels, preds, scores):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, -scores)  # Negative since lower = fake
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

def load_previous_history(results_path=RESULTS_PATH):
    """Load previous training results (if exist)."""
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                history = json.load(f)
                if history and history[-1].get("summary_type") in ("top_k", "best_model"):
                    history.pop()  # remove previous summary
                return history
        except json.JSONDecodeError:
            sys.stderr.write(f"⚠️ Corrupted results file at {results_path}. Restarting history.\n")
    return []

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # Limit CPU threads
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
    os.environ["NUMEXPR_NUM_THREADS"] = "8"

    np.random.seed(42)

    print("🔹 Loading features...")
    X_train, y_train, _ = load_features("data/features/features_train.npz")
    X_val, y_val, _ = load_features("data/features/features_val.npz")
    X_test, y_test, _ = load_features("data/features/features_test.npz")

    X_train_real = X_train[y_train == 0]
    print(f"Train set (real only): {X_train_real.shape}")
    print(f"Validation set: {X_val.shape}, Test set: {X_test.shape}")

    # ✅ Subsample for speed/memory safety
    if len(X_train_real) > SUBSAMPLE_LIMIT:
        print(f"⚠️ Too many samples ({len(X_train_real)}). Subsampling {SUBSAMPLE_LIMIT:,} for GMM training...")
        idx = np.random.choice(len(X_train_real), SUBSAMPLE_LIMIT, replace=False)
        X_train_real = X_train_real[idx]

    previous_history = load_previous_history()
    current_run_results = []

    # --- SINGLE TRIAL TRAINING ---
    print("\n====================== TRAINING GMM ======================")
    random_state = 42
    try:
        gmm = GaussianMixture(
            n_components=4,
            covariance_type='diag',
            random_state=random_state,
            max_iter=200,
            verbose=0
        )
        gmm.fit(X_train_real)
    except Exception as e:
        print(f"❌ GMM training failed: {e}")
        sys.exit(1)

    # --- Threshold Tuning on Validation Set ---
    print("🔹 Tuning threshold on validation set...")
    scores_val = gmm.score_samples(X_val)
    thresholds = np.linspace(scores_val.min(), scores_val.max(), 200)
    best_thresh, best_f1, best_metrics = None, -1.0, None

    for t in tqdm(thresholds, desc="Threshold search"):
        preds = (scores_val < t).astype(int)
        m = compute_metrics(y_val, preds, scores_val)
        if m["prec"] >= MIN_PRECISION_FLOOR and m["f1"] > best_f1:
            best_f1, best_thresh, best_metrics = m["f1"], t, m

    if best_thresh is None:
        print(f"❌ Could not find valid threshold (min precision {MIN_PRECISION_FLOOR}). Exiting.")
        sys.exit(1)

    print(f"\n✅ Best threshold: {best_thresh:.4f} | F1={best_f1:.4f} | Prec={best_metrics['prec']:.4f}")

    # --- Evaluate on Test Set ---
    print("🔹 Evaluating on Test set...")
    scores_test = gmm.score_samples(X_test)
    preds_test = (scores_test < best_thresh).astype(int)
    test_metrics = compute_metrics(y_test, preds_test, scores_test)

    print("[Test Metrics]")
    for k, v in test_metrics.items():
        print(f" {k}: {v:.4f}")

    # --- Save Best Model ---
    os.makedirs("models", exist_ok=True)
    save_path = f"{MODEL_PREFIX}_best.pkl"
    dump({"gmm": gmm, "threshold": best_thresh}, save_path)
    print(f"\n🏆 Saved BEST model to {save_path}")

    # --- Save Results Summary ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_entry = {
        "timestamp": timestamp,
        "trial_num": 0,
        "random_state": random_state,
        "train_samples": int(X_train_real.shape[0]),
        "threshold": float(best_thresh),
        "val_metrics": {k: float(v) for k, v in best_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }

    current_run_results.append(result_entry)
    clean_history = previous_history + current_run_results

    summary_entry = {
        "timestamp": timestamp,
        "summary_type": "best_model",
        "num_trials": NUM_TRIALS,
        "min_precision_floor": MIN_PRECISION_FLOOR,
        "best_model": {
            "threshold": float(best_thresh),
            "f1": float(test_metrics["f1"]),
            "precision": float(test_metrics["prec"]),
            "recall": float(test_metrics["rec"]),
            "model_file": save_path
        }
    }
    clean_history.append(summary_entry)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(clean_history, f, indent=4)

    print(f"📂 Results saved to {RESULTS_PATH} with BEST model summary.")
