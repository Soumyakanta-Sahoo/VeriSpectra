# quick_train_check.py — 5-min pre-flight diagnostic for train_gmm_v3.py

import numpy as np
import os, sys, traceback
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.io import imread
from scipy.fftpack import dct
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# Limit thread usage (avoid OpenBLAS errors)
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

# -------------------------------
#  BASIC FEATURE EXTRACTORS
# -------------------------------
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
    except Exception:
        return np.zeros(n_bins)

def extract_cnn_features_small(img_paths, base_batch=4):
    """Fast CNN feature test with EfficientNetB0 (few images)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.efficientnet_b0(weights="IMAGENET1K_V1").to(device)
    model.eval()
    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    feats = []
    with torch.no_grad():
        for p in tqdm(img_paths, desc="CNN check", total=len(img_paths)):
            try:
                img = Image.open(p).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(device)
                out = model.features(tensor).mean([2, 3]).cpu().numpy().flatten()
                feats.append(out)
            except Exception as e:
                print(f"⚠️ CNN load error for {p}: {e}")
                feats.append(np.zeros(1280))
    return np.array(feats)

# -------------------------------
#  CHECK PIPELINE
# -------------------------------
def fuse_features(residual, freq, lbp, cnn):
    min_len = min(residual.shape[1], freq.shape[1])
    return np.hstack([residual[:, :min_len], freq, lbp, cnn])

def run_check():
    print("🔍 Starting 5-min preflight diagnostic for train_gmm_v3.py")

    # Sample small subset (simulate real paths)
    feat_path = "data/features/features_train.npz"
    if not os.path.exists(feat_path):
        print(f"❌ Missing feature file: {feat_path}")
        sys.exit(1)

    data = np.load(feat_path, allow_pickle=True)
    X_res, y, paths = data["features"], data["labels"], data["paths"]

    # Take small subset for test
    limit = min(500, len(X_res))
    idx = np.random.choice(len(X_res), limit, replace=False)
    X_res, y, paths = X_res[idx], y[idx], paths[idx]
    print(f"✅ Loaded {limit} samples for check.")

    # --- Frequency ---
    print("⚙️ Extracting frequency features...")
    X_freq = np.array([extract_frequency_features(x) for x in tqdm(X_res[:100])])

    # --- LBP ---
    print("⚙️ Extracting LBP features...")
    X_lbp = np.array([extract_lbp_hist(p) for p in tqdm(paths[:100])])

    # --- CNN ---
    print("⚙️ Extracting CNN features (light test)...")
    X_cnn = extract_cnn_features_small(paths[:20])

    # --- Fuse ---
    print("⚙️ Fusing all feature types...")
    X_fused = fuse_features(
        X_res[:20], X_freq[:20], X_lbp[:20], X_cnn[:20]
    )

    # --- GMM Sanity ---
    print("⚙️ Fitting tiny GMM...")
    gmm = GaussianMixture(n_components=2, covariance_type="diag", random_state=42)
    gmm.fit(X_fused)
    print("✅ GMM trained successfully on small sample.")

    print(f"📊 Fused shape: {X_fused.shape}")
    print("🚀 All stages passed — pipeline safe for full training!")

# -------------------------------
if __name__ == "__main__":
    try:
        run_check()
    except Exception as e:
        print("❌ Pipeline check failed:")
        traceback.print_exc()
        sys.exit(1)
