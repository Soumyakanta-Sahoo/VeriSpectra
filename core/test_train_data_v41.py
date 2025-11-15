# test_train_data_v41.py — Evaluate Hybrid GMM + LR Fusion (v41)
import os, sys, json, datetime, time
import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# ---------------- CONFIG ----------------
MODEL_PATH = "models/hier_gmm_model_v41.pkl"
RESULT_PATH = "results/train_data_eval_v41.csv"
CACHE_DIR = "cache_v41"
DATA_PATH = "data/features/features_train.npz"
BASE_BATCH = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------- UTILS ----------------
def load_features(path):
    data = np.load(path, allow_pickle=True)
    return data["features"], data["labels"], data["paths"]

def load_npz(path):
    return np.load(path, allow_pickle=True)["X"]

def zscore(arr):
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std==0] = 1.0
    return (arr - mean) / (std + 1e-9)

def fuse_features(X_res, X_freq, X_lbp, X_cnn):
    X_res, X_freq, X_lbp, X_cnn = map(zscore, [X_res, X_freq, X_lbp, X_cnn])
    return np.hstack([X_res, X_freq, X_lbp, X_cnn])

def extract_lbp_hist(img_path, radii=[1,2,3], P=8, n_bins=10):
    try:
        img = imread(img_path)
        gray = rgb2gray(img)
        gray_uint8 = (gray*255).astype(np.uint8)
        all_hist = []
        for r in radii:
            lbp = local_binary_pattern(gray_uint8, P, r, method="uniform")
            hist,_ = np.histogram(lbp, bins=n_bins, range=(0,n_bins), density=True)
            all_hist.extend(hist.tolist())
        return np.array(all_hist, dtype=np.float32)
    except:
        return np.zeros(n_bins*len(radii), dtype=np.float32)

def extract_cnn_features_adaptive(img_paths, cache_file, base_batch=BASE_BATCH):
    if os.path.exists(cache_file):
        return np.load(cache_file)["X"]
    device = DEVICE
    model = models.efficientnet_b0(weights="IMAGENET1K_V1").to(device)
    model.eval()
    preprocess = T.Compose([T.Resize((224,224)), T.ToTensor(),
                            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    features = []
    for i in tqdm(range(0, len(img_paths), base_batch), desc="CNN features"):
        batch_paths = img_paths[i:i+base_batch]
        batch_tensor = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            out = model.features(batch_tensor).mean([2,3]).cpu().numpy()
        features.append(out)
    features_all = np.concatenate(features, axis=0)
    np.savez_compressed(cache_file, X=features_all)
    return features_all

# ---------------- METRICS ----------------
def compute_metrics(labels, preds, probs):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = float("nan")
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

# ---------------- MAIN ----------------
if __name__=="__main__":
    print("🔹 Loading base residual features...")
    X_res, y_train, paths = load_features(DATA_PATH)

    # Load cached features
    X_freq = load_npz(os.path.join(CACHE_DIR, "train_freq_v4.npz"))
    X_lbp  = load_npz(os.path.join(CACHE_DIR, "train_lbp.npz"))
    X_cnn  = extract_cnn_features_adaptive(paths, os.path.join(CACHE_DIR, "train_cnn.npz"))

    print(f"✅ Shapes: Residual={X_res.shape}, Freq={X_freq.shape}, LBP={X_lbp.shape}, CNN={X_cnn.shape}")

    print("🔹 Fusing features...")
    X_fused = fuse_features(X_res, X_freq, X_lbp, X_cnn)

    print("🔹 Loading trained model...")
    model_data = load(MODEL_PATH)
    gmm = model_data["gmm"]
    lr_model = model_data["lr"]
    threshold = model_data.get("threshold",0.5)

    print("🔹 Scoring and predicting...")
    scores = -gmm.score_samples(X_fused)
    probs = lr_model.predict_proba(scores.reshape(-1,1))[:,1]
    preds = (probs >= threshold).astype(int)

    metrics = compute_metrics(y_train, preds, probs)
    correct = np.sum(preds==y_train)
    total = len(y_train)

    print("\n📊 ---- TRAINING DATA EVALUATION ----")
    print(f"Total samples: {total}, Correct: {correct} ({correct/total*100:.2f}%)")
    print(f"[METRICS] acc={metrics['acc']:.4f}, prec={metrics['prec']:.4f}, rec={metrics['rec']:.4f}, f1={metrics['f1']:.4f}, auc={metrics['auc']:.4f}")

    # Save results
    df = pd.DataFrame({"true_label": y_train, "pred_label": preds, "score": scores})
    df.to_csv(RESULT_PATH, index=False)
    print(f"📂 Results saved → {RESULT_PATH}")
