# train_gmm_v4.py - Hierarchical & Hybrid Feature Deepfake Detection v4
# Improvements:
# - Enhanced FFT features (6 bands)
# - Multi-radius LBP (radii 1,2,3)
# - Per-feature z-score normalization
# - Weighted fusion with Logistic Regression on GMM scores

import os, json, datetime, sys, warnings, time
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
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
MODEL_PREFIX = "models/hier_gmm_model_v4"
RESULTS_PATH = "results/hier_gmm_results_v4.json"
MIN_PRECISION_FLOOR = 0.97
SUBSAMPLE_LIMIT = 50000
CACHE_DIR = "cache_v4"
LOG_PATH = "results/train_gmm_v4_log.txt"
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

# ---------------- FFT FEATURES ----------------
def extract_frequency_features(img_array, n_bands=6):
    # Convert RGB to grayscale if needed
    if img_array.ndim == 3:
        img_array = rgb2gray(img_array)

    h, w = img_array.shape
    F = np.fft.fft2(img_array)
    F = np.fft.fftshift(F)
    mag = np.abs(F)
    total = mag.sum() + 1e-9

    cx, cy = w // 2, h // 2
    lx, ly = int(0.25 * w / 2), int(0.25 * h / 2)
    low = mag[cy - ly:cy + ly, cx - lx:cx + lx].sum()
    high = total - low
    hf_ratio = high / total

    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    rmax = np.max(r)

    features = [hf_ratio]
    for k in range(1, n_bands + 1):
        mask = (r >= (k - 1) * rmax / n_bands) & (r < k * rmax / n_bands)
        if mask.sum() == 0:
            features += [0.0, 0.0]
        else:
            band = mag[mask]
            features += [band.mean(), band.std()]

    return np.array(features, dtype=np.float32)


# ---------------- LBP FEATURES ----------------
def extract_lbp_hist(img_path, radii=[1,2,3], P=8, n_bins=10):
    try:
        img = imread(img_path)
        gray = rgb2gray(img)
        gray_uint8 = (gray*255).astype(np.uint8)
        all_hist = []
        for r in radii:
            lbp = local_binary_pattern(gray_uint8, P, r, method="uniform")
            hist, _ = np.histogram(lbp, bins=n_bins, range=(0,n_bins), density=True)
            all_hist.extend(hist)
        return np.array(all_hist, dtype=np.float32)
    except Exception as e:
        log(f"⚠️ LBP read error for {img_path}: {e}")
        return np.zeros(n_bins*len(radii))

# ---------------- CNN FEATURES ----------------
def get_gpu_info():
    if not torch.cuda.is_available(): return None
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
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    def load_batch_tensor(paths):
        imgs = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            except:
                imgs.append(torch.zeros(3,224,224))
        return torch.stack(imgs)

    features = []
    i, total, saved_count = 0, len(img_paths), 0
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
                if "out of memory" in msg and device=="cuda" and batch_size>1:
                    batch_size = max(1,batch_size//2)
                    log(f"➡️ Reducing batch_size to {batch_size}")
                    torch.cuda.empty_cache()
                    continue
                elif "out of memory" in msg and device=="cuda" and batch_size==1:
                    log("➡️ Switching model to CPU")
                    device="cpu"; model.to(device); torch.cuda.empty_cache()
                    continue
                else:
                    i += batch_size
                    pbar.update(batch_size)
            finally:
                try: del batch_tensor, out
                except: pass
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            current_count = sum(len(chunk) for chunk in features)
            if current_count - saved_count >= CHECKPOINT_EVERY or i>=total:
                arr = np.concatenate(features, axis=0)
                np.savez_compressed(cache_file, X=arr)
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
            with open(results_path,"r") as f:
                history = json.load(f)
                if history and history[-1].get("summary_type") in ("top_k","best_model"):
                    history.pop()
                return history
        except: log(f"⚠️ Corrupted results file at {results_path}. Restarting history.")
    return []

# ---------------- FUSION WITH Z-SCORE ----------------
def fuse_features_by_paths(residual_feats, freq_feats, lbp_feats, cnn_feats,
                           paths_res, paths_freq, paths_lbp, paths_cnn, y_labels):
    common_paths = set(paths_res) & set(paths_freq) & set(paths_lbp) & set(paths_cnn)
    common_paths = sorted(list(common_paths))
    idx_res = {p:i for i,p in enumerate(paths_res)}
    idx_freq = {p:i for i,p in enumerate(paths_freq)}
    idx_lbp = {p:i for i,p in enumerate(paths_lbp)}
    idx_cnn = {p:i for i,p in enumerate(paths_cnn)}

    res_array = np.array(residual_feats)
    freq_array = np.array(freq_feats)
    lbp_array = np.array(lbp_feats)
    cnn_array = np.array(cnn_feats)

    def zscore_block(arr):
        return (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True)+1e-9)

    res_array = zscore_block(res_array)
    freq_array = zscore_block(freq_array)
    lbp_array = zscore_block(lbp_array)
    cnn_array = zscore_block(cnn_array)

    fused_list, y_aligned_list = [], []
    for p in tqdm(common_paths, desc="Fusing features"):
        fused_vec = np.hstack([
            res_array[idx_res[p]],
            freq_array[idx_freq[p]],
            lbp_array[idx_lbp[p]],
            cnn_array[idx_cnn[p]]
        ])
        fused_list.append(fused_vec)
        y_aligned_list.append(y_labels[idx_res[p]])

    fused = np.array(fused_list)
    y_aligned = np.array(y_aligned_list)
    return fused, y_aligned

# ---------------- MAIN ----------------
if __name__=="__main__":
    np.random.seed(42)

    log("🔹 Loading base residual features...")
    X_train_res, y_train, train_paths = load_features("data/features/features_train.npz")
    X_val_res, y_val, val_paths = load_features("data/features/features_val.npz")
    X_test_res, y_test, test_paths = load_features("data/features/features_test.npz")

    # ---------------- Frequency + LBP + CNN Features ----------------
    log("🔹 Loading / extracting frequency features...")
    freq_cache_train = os.path.join(CACHE_DIR, "train_freq_v4.npz")
    freq_cache_val = os.path.join(CACHE_DIR, "val_freq_v4.npz")
    freq_cache_test = os.path.join(CACHE_DIR, "test_freq_v4.npz")

    if os.path.exists(freq_cache_train):
        X_train_freq = np.load(freq_cache_train)["X"]
    else:
        X_train_freq = np.array([extract_frequency_features(imread(p)) for p in tqdm(train_paths)])
        np.savez_compressed(freq_cache_train, X=X_train_freq)

    if os.path.exists(freq_cache_val):
        X_val_freq = np.load(freq_cache_val)["X"]
    else:
        X_val_freq = np.array([extract_frequency_features(imread(p)) for p in tqdm(val_paths)])
        np.savez_compressed(freq_cache_val, X=X_val_freq)

    if os.path.exists(freq_cache_test):
        X_test_freq = np.load(freq_cache_test)["X"]
    else:
        X_test_freq = np.array([extract_frequency_features(imread(p)) for p in tqdm(test_paths)])
        np.savez_compressed(freq_cache_test, X=X_test_freq)

    log("🔹 Loading / extracting LBP features...")
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

    log("🔹 Extracting CNN features...")
    cnn_cache_train = os.path.join(CACHE_DIR, "train_cnn.npz")
    cnn_cache_val = os.path.join(CACHE_DIR, "val_cnn.npz")
    cnn_cache_test = os.path.join(CACHE_DIR, "test_cnn.npz")

    X_train_cnn = extract_cnn_features_adaptive(train_paths, cnn_cache_train)
    X_val_cnn = extract_cnn_features_adaptive(val_paths, cnn_cache_val)
    X_test_cnn = extract_cnn_features_adaptive(test_paths, cnn_cache_test)

    # ---------------- Feature Fusion ----------------
    log("🔹 Fusing all features with z-score normalization...")
    X_train_fused, y_train_aligned = fuse_features_by_paths(
        X_train_res, X_train_freq, X_train_lbp, X_train_cnn,
        train_paths, train_paths, train_paths, train_paths, y_train
    )
    X_val_fused, y_val_aligned = fuse_features_by_paths(
        X_val_res, X_val_freq, X_val_lbp, X_val_cnn,
        val_paths, val_paths, val_paths, val_paths, y_val
    )
    X_test_fused, y_test_aligned = fuse_features_by_paths(
        X_test_res, X_test_freq, X_test_lbp, X_test_cnn,
        test_paths, test_paths, test_paths, test_paths, y_test
    )

    # ---------------- Train GMM ----------------
    log("🔹 Training hierarchical GMM (coarse)...")
    gmm_coarse = GaussianMixture(n_components=4, covariance_type="full", random_state=42,verbose=2, verbose_interval=1)
    gmm_coarse.fit(X_train_fused)

    # Compute anomaly scores
    train_scores = -gmm_coarse.score_samples(X_train_fused)
    val_scores = -gmm_coarse.score_samples(X_val_fused)
    test_scores = -gmm_coarse.score_samples(X_test_fused)

    # ---------------- Logistic Regression on GMM scores ----------------
    log("🔹 Training Logistic Regression on GMM scores...")
    lr_model = LogisticRegression(max_iter=500)
    lr_model.fit(train_scores.reshape(-1,1), y_train_aligned)

    train_pred = lr_model.predict(train_scores.reshape(-1,1))
    val_pred = lr_model.predict(val_scores.reshape(-1,1))
    test_pred = lr_model.predict(test_scores.reshape(-1,1))

    # ---------------- Compute Metrics ----------------
    train_metrics = compute_metrics(y_train_aligned, train_pred, train_scores)
    val_metrics = compute_metrics(y_val_aligned, val_pred, val_scores)
    test_metrics = compute_metrics(y_test_aligned, test_pred, test_scores)

    log(f"✅ Train Metrics: {train_metrics}")
    log(f"✅ Val Metrics: {val_metrics}")
    log(f"✅ Test Metrics: {test_metrics}")

    # ---------------- Save Model ----------------
    os.makedirs("models", exist_ok=True)
    dump({"gmm": gmm_coarse, "lr": lr_model}, MODEL_PREFIX+".pkl")
    log(f"💾 Model saved to {MODEL_PREFIX}.pkl")

    # ---------------- Save Results ----------------
    results = {
        "timestamp": str(datetime.datetime.now()),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": MODEL_PREFIX+".pkl"
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH,"w") as f:
        json.dump(results,f,indent=4)
    log(f"💾 Results saved to {RESULTS_PATH}")

