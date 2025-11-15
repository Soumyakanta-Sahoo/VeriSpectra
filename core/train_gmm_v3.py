# train_gmm_v3.py - Hierarchical & Hybrid Feature Deepfake Detection
# (Adaptive Hybrid CNN + VRAM monitoring + Checkpointing + Resumable caches)

import os, json, datetime, sys, warnings, time
# ---- Environment / Thread safety ----
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from joblib import dump
from scipy.fftpack import dct
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.io import imread
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image



# ---------------- CONFIG ----------------
NUM_TRIALS = 1
MODEL_PREFIX = "models/hier_gmm_model_v3"
RESULTS_PATH = "results/hier_gmm_results_v3.json"
MIN_PRECISION_FLOOR = 0.97
SUBSAMPLE_LIMIT = 50000
CACHE_DIR = "cache"
LOG_PATH = "results/train_gmm_v3_log.txt"
CHECKPOINT_EVERY = 5000
BASE_BATCH = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ---- Logging helper ----
def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

# ---------------- UTILS ----------------
def load_features(path):
    data = np.load(path, allow_pickle=True)
    return data["features"], data["labels"], data["paths"]

def extract_frequency_features(img_array):
    dct_coeffs = dct(dct(img_array.T, norm="ortho").T, norm="ortho")
    return np.abs(dct_coeffs).flatten()[:256]

def extract_lbp_hist(img_path, P=8, R=1, n_bins=10):
    try:
        img = imread(img_path)
        gray = rgb2gray(img)
        gray_uint8 = (gray * 255).astype(np.uint8)
        lbp = local_binary_pattern(gray_uint8, P, R, method="uniform")
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        return hist
    except Exception as e:
        log(f"⚠️ LBP read error for {img_path}: {e}")
        return np.zeros(n_bins)

# ---------------- ADAPTIVE CNN EXTRACTOR ----------------
def get_gpu_info():
    if not torch.cuda.is_available():
        return None
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {"total": props.total_memory,
            "used": torch.cuda.memory_allocated(idx),
            "reserved": torch.cuda.memory_reserved(idx)}

def extract_cnn_features_adaptive(img_paths, cache_file, base_batch=BASE_BATCH):
    if os.path.exists(cache_file):
        log(f"🔁 Loading cached CNN features from {cache_file}")
        return np.load(cache_file)["X"]

    device = DEVICE
    batch_size = base_batch
    try:
        model = models.efficientnet_b0(weights="IMAGENET1K_V1").to(device)
    except Exception as e:
        log(f"⚠️ EfficientNet to GPU failed: {e}. Using CPU.")
        device = "cpu"
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.eval()

    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    def load_batch_tensor(paths):
        imgs = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            except Exception as e:
                log(f"⚠️ Image load error {p}: {e}")
                imgs.append(torch.zeros(3, 224, 224))
        return torch.stack(imgs)

    features = []
    i = 0
    total = len(img_paths)
    saved_count = 0
    start_t = time.time()

    with torch.no_grad():
        pbar = tqdm(total=total, desc="CNN Adaptive", unit="img")
        while i < total:
            batch_paths = img_paths[i:i+batch_size]
            batch_tensor = load_batch_tensor(batch_paths)
            try:
                batch_tensor = batch_tensor.to(device)
                out = model.features(batch_tensor).mean([2,3]).cpu().numpy()
                features.append(out)
                i += batch_size
                pbar.update(batch_size)
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    log(f"⚠️ CUDA OOM at index {i} with batch_size {batch_size}")
                    if device=="cuda" and batch_size>1:
                        batch_size = max(1, batch_size//2)
                        log(f"➡️ Reducing batch_size to {batch_size} and retrying")
                        torch.cuda.empty_cache()
                        continue
                    elif device=="cuda" and batch_size==1:
                        log("➡️ Switching model to CPU due to persistent OOM")
                        device="cpu"
                        model.to(device)
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
                else:
                    log(f"⚠️ Runtime error at index {i}: {e}")
                    i += batch_size
                    pbar.update(batch_size)
            finally:
                try: del batch_tensor, out
                except: pass
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            # checkpoint
            current_count = sum(len(chunk) for chunk in features)
            if current_count - saved_count >= CHECKPOINT_EVERY or i >= total:
                arr = np.concatenate(features, axis=0)
                np.savez_compressed(cache_file, X=arr)
                log(f"💾 Checkpoint saved: {cache_file} ({arr.shape[0]} features)")
                saved_count = arr.shape[0]

        pbar.close()

    features_all = np.concatenate(features, axis=0)
    np.savez_compressed(cache_file, X=features_all)
    log(f"✅ CNN extraction complete ({features_all.shape[0]} x {features_all.shape[1]})")
    log(f"⏱️ Time elapsed: {(time.time()-start_t)/60:.2f} min")
    return features_all

# ---------------- METRICS ----------------
def compute_metrics(labels, preds, scores):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, -scores)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

def load_previous_history(results_path=RESULTS_PATH):
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                history = json.load(f)
                if history and history[-1].get("summary_type") in ("top_k","best_model"):
                    history.pop()
                return history
        except json.JSONDecodeError:
            log(f"⚠️ Corrupted results file at {results_path}. Restarting history.")
    return []

# ---------------- FUSE FEATURES WITH ALIGNMENT ----------------
def fuse_features_by_paths(residual_feats, freq_feats, lbp_feats, cnn_feats,
                           paths_res, paths_freq, paths_lbp, paths_cnn, y_labels):
    # Find common paths across all feature sets
    common_paths = set(paths_res) & set(paths_freq) & set(paths_lbp) & set(paths_cnn)
    common_paths = sorted(list(common_paths))

    # Create lookup dicts for faster indexing
    idx_res = {p: i for i, p in enumerate(paths_res)}
    idx_freq = {p: i for i, p in enumerate(paths_freq)}
    idx_lbp = {p: i for i, p in enumerate(paths_lbp)}
    idx_cnn = {p: i for i, p in enumerate(paths_cnn)}

    fused_list = []
    y_aligned_list = []

    # Fuse features per image path
    for p in tqdm(common_paths, desc="Fusing features"):
        fused_vec = np.hstack([
            residual_feats[idx_res[p]],
            freq_feats[idx_freq[p]],
            lbp_feats[idx_lbp[p]],
            cnn_feats[idx_cnn[p]]
        ])
        fused_list.append(fused_vec)
        y_aligned_list.append(y_labels[idx_res[p]])

    fused = np.array(fused_list)
    y_aligned = np.array(y_aligned_list)
    return fused, y_aligned

# ---------------- MAIN ----------------
if __name__ == "__main__":
    np.random.seed(42)
    log("🔹 Loading base residual features...")
    X_train_res, y_train, train_paths = load_features("data/features/features_train.npz")
    X_val_res, y_val, val_paths = load_features("data/features/features_val.npz")
    X_test_res, y_test, test_paths = load_features("data/features/features_test.npz")

    # Frequency
    freq_cache_train = os.path.join(CACHE_DIR, "train_freq.npz")
    freq_cache_val = os.path.join(CACHE_DIR, "val_freq.npz")
    freq_cache_test = os.path.join(CACHE_DIR, "test_freq.npz")

    if os.path.exists(freq_cache_train):
        X_train_freq = np.load(freq_cache_train)["X"]
    else:
        X_train_freq = np.array([extract_frequency_features(x) for x in tqdm(X_train_res)])
        np.savez_compressed(freq_cache_train, X=X_train_freq)

    if os.path.exists(freq_cache_val):
        X_val_freq = np.load(freq_cache_val)["X"]
    else:
        X_val_freq = np.array([extract_frequency_features(x) for x in tqdm(X_val_res)])
        np.savez_compressed(freq_cache_val, X=X_val_freq)

    if os.path.exists(freq_cache_test):
        X_test_freq = np.load(freq_cache_test)["X"]
    else:
        X_test_freq = np.array([extract_frequency_features(x) for x in tqdm(X_test_res)])
        np.savez_compressed(freq_cache_test, X=X_test_freq)

    # LBP
    lbp_cache_train = os.path.join(CACHE_DIR, "train_lbp.npz")
    lbp_cache_val = os.path.join(CACHE_DIR, "val_lbp.npz")
    lbp_cache_test = os.path.join(CACHE_DIR, "test_lbp.npz")

    if os.path.exists(lbp_cache_train):
        X_train_lbp = np.load(lbp_cache_train)["X"]
    else:
        X_train_lbp = np.array([extract_lbp_hist(p) for p in tqdm(train_paths)])
        np.savez_compressed(lbp_cache_train, X=X_train_lbp)

    if os.path.exists(lbp_cache_val):
        X_val_lbp = np.load(lbp_cache_val)["X"]
    else:
        X_val_lbp = np.array([extract_lbp_hist(p) for p in tqdm(val_paths)])
        np.savez_compressed(lbp_cache_val, X=X_val_lbp)

    if os.path.exists(lbp_cache_test):
        X_test_lbp = np.load(lbp_cache_test)["X"]
    else:
        X_test_lbp = np.array([extract_lbp_hist(p) for p in tqdm(test_paths)])
        np.savez_compressed(lbp_cache_test, X=X_test_lbp)

    # CNN
    cnn_cache_train = os.path.join(CACHE_DIR, "train_cnn.npz")
    cnn_cache_val = os.path.join(CACHE_DIR, "val_cnn.npz")
    cnn_cache_test = os.path.join(CACHE_DIR, "test_cnn.npz")

    X_train_cnn = extract_cnn_features_adaptive(train_paths, cnn_cache_train)
    X_val_cnn = extract_cnn_features_adaptive(val_paths, cnn_cache_val)
    X_test_cnn = extract_cnn_features_adaptive(test_paths, cnn_cache_test)

    # Fuse
    log("🔹 Fusing Residual + Frequency + LBP + CNN features...")
    X_train, y_train_aligned = fuse_features_by_paths(
        X_train_res, X_train_freq, X_train_lbp, X_train_cnn,
        train_paths, train_paths, train_paths, train_paths, y_train
    )
    X_val, y_val_aligned = fuse_features_by_paths(
        X_val_res, X_val_freq, X_val_lbp, X_val_cnn,
        val_paths, val_paths, val_paths, val_paths, y_val
    )
    X_test, y_test_aligned = fuse_features_by_paths(
        X_test_res, X_test_freq, X_test_lbp, X_test_cnn,
        test_paths, test_paths, test_paths, test_paths, y_test
    )

    # Subsample real
    X_train_real = X_train[y_train_aligned==0]
    if len(X_train_real) > SUBSAMPLE_LIMIT:
        idx = np.random.choice(len(X_train_real), SUBSAMPLE_LIMIT, replace=False)
        X_train_real = X_train_real[idx]

    previous_history = load_previous_history()

    # ---- GMM training ----
    log("🔹 Training Coarse GMM...")
    coarse_gmm = GaussianMixture(n_components=4, covariance_type="diag", random_state=42, max_iter=200)
    coarse_gmm.fit(X_train_real)
    log("✅ Coarse GMM trained.")

    log("🔹 Training Fine GMM...")
    scores_train = coarse_gmm.score_samples(X_train)
    ambiguous_idx = np.where(
        (scores_train < np.percentile(scores_train,90)) & 
        (scores_train > np.percentile(scores_train,10))
    )[0]
    if len(ambiguous_idx)==0:
        ambiguous_idx = np.arange(min(1000, X_train.shape[0]))
    fine_gmm = GaussianMixture(n_components=4, covariance_type="diag", random_state=42, max_iter=200)
    fine_gmm.fit(X_train[ambiguous_idx])
    log("✅ Fine GMM trained.")

    # Threshold tuning
    log("🔹 Tuning threshold on validation set...")
    scores_val = 0.5*coarse_gmm.score_samples(X_val) + 0.5*fine_gmm.score_samples(X_val)
    thresholds = np.linspace(scores_val.min(), scores_val.max(), 200)
    best_thresh, best_f1, best_metrics = None, -1.0, None
    for t in tqdm(thresholds, desc="Threshold search"):
        preds = (scores_val < t).astype(int)
        m = compute_metrics(y_val_aligned, preds, scores_val)
        if m["prec"] >= MIN_PRECISION_FLOOR and m["f1"] > best_f1:
            best_f1, best_thresh, best_metrics = m["f1"], t, m
    if best_thresh is None:
        log("❌ Could not find valid threshold. Exiting.")
        sys.exit(1)
    log(f"✅ Best threshold: {best_thresh:.4f} | F1={best_f1:.4f} | Prec={best_metrics['prec']:.4f}")

    # Final evaluation
    scores_test = 0.5*coarse_gmm.score_samples(X_test) + 0.5*fine_gmm.score_samples(X_test)
    preds_test = (scores_test < best_thresh).astype(int)
    test_metrics = compute_metrics(y_test_aligned, preds_test, scores_test)
    log("[Test Metrics] " + ", ".join([f"{k}:{v:.4f}" for k,v in test_metrics.items()]))

    # Save model & results
    os.makedirs("models", exist_ok=True)
    save_path = f"{MODEL_PREFIX}_best.pkl"
    dump({"coarse_gmm": coarse_gmm, "fine_gmm": fine_gmm, "threshold": best_thresh}, save_path)
    log(f"🏆 Saved BEST Hierarchical Model → {save_path}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_entry = {
        "timestamp": timestamp,
        "train_samples": int(X_train_real.shape[0]),
        "threshold": float(best_thresh),
        "val_metrics": {k: float(v) for k,v in best_metrics.items()},
        "test_metrics": {k: float(v) for k,v in test_metrics.items()}
    }
    summary_entry = {
        "timestamp": timestamp,
        "summary_type": "best_model",
        "best_model": {
            "threshold": float(best_thresh),
            "f1": float(test_metrics["f1"]),
            "precision": float(test_metrics["prec"]),
            "recall": float(test_metrics["rec"]),
            "auc": float(test_metrics["auc"]),
            "model_file": save_path
        }
    }
    history = previous_history + [result_entry, summary_entry]
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(history, f, indent=4)
    log(f"📂 Results logged at {RESULTS_PATH}")
