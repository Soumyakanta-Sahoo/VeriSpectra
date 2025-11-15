import os, json, numpy as np, cv2, torch, joblib
from tqdm import tqdm
from train_gmm_v42 import (
    extract_frequency_features,
    extract_lbp_hist,
    extract_cnn_features_adaptive
)
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
UNSEEN_DIR = "Sample_img"
CACHE_CNN = "cache_v42/unseen_cnn.npz"
MODEL_PATH = "models/hier_gmm_model_v42.pkl"

def log(msg): print(msg, flush=True)

# ---------------- LOAD IMAGES ----------------
img_paths = sorted([
    os.path.join(UNSEEN_DIR, f)
    for f in os.listdir(UNSEEN_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])
log(f"🖼️ Found {len(img_paths)} unseen images.")

# ---------------- FEATURE EXTRACTION ----------------
log("🔹 Extracting frequency features...")
freq_feats = []
for p in tqdm(img_paths):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    freq_feats.append(extract_frequency_features(img))
freq_feats = np.array(freq_feats)

log("🔹 Extracting LBP features...")
lbp_feats = []
for p in tqdm(img_paths):
    try:
        lbp_feats.append(extract_lbp_hist(p))
    except Exception as e:
        print(f"[⚠️] LBP read error for {p}: {e}")
lbp_feats = np.array(lbp_feats)

log("🔹 Extracting CNN features...")
cnn_feats = extract_cnn_features_adaptive(img_paths, CACHE_CNN)

# ---------------- FUSION (Z-SCORE) ----------------
def zscore_block(arr):
    return (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-9)

freq_z = zscore_block(freq_feats)
lbp_z = zscore_block(lbp_feats)
cnn_z = zscore_block(cnn_feats)

# Align by min length in case of mismatch
min_len = min(len(freq_z), len(lbp_z), len(cnn_z))
freq_z, lbp_z, cnn_z = freq_z[:min_len], lbp_z[:min_len], cnn_z[:min_len]

X_unseen = np.hstack([freq_z, lbp_z, cnn_z])
log(f"✅ Feature shape after fusion: {X_unseen.shape}")

# ---------------- LOAD MODEL & INFER ----------------
log("🔹 Loading trained GMM model...")
gmm = joblib.load(MODEL_PATH)

log("🔹 Computing log-likelihood scores...")
scores = -gmm.score_samples(X_unseen)

# ---------------- OUTPUT ----------------
results = {os.path.basename(p): float(s) for p, s in zip(img_paths[:min_len], scores)}
np.save("results_v42_unseen.npy", results)
log("✅ Inference complete. Saved scores → results_v42_unseen.npy")

for name, score in list(results.items())[:5]:
    print(f"{name:25s}  score={score:.4f}")
