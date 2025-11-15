# test_folder_images.py - Test Hierarchical GMM on folder images
import os
import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as T

# ---------------- CONFIG ----------------
MODEL_PATH = "models/hier_gmm_model_v3_best.pkl"
IMG_FOLDER = "Sample_img"
CACHE_DIR = "cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_BATCH = 8

# ---------------- FEATURE EXTRACTORS ----------------
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
    except:
        return np.zeros(n_bins)

def extract_cnn_features_adaptive(img_paths, base_batch=BASE_BATCH):
    device = DEVICE
    batch_size = base_batch
    try:
        model = models.efficientnet_b0(weights="IMAGENET1K_V1").to(device)
    except:
        device = "cpu"
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.eval()

    preprocess = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
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
    i = 0
    total = len(img_paths)
    with torch.no_grad():
        pbar = tqdm(total=total, desc="CNN Features", unit="img")
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
                if "out of memory" in str(e).lower() and device=="cuda" and batch_size>1:
                    batch_size = max(1, batch_size//2)
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
    pbar.close()
    return np.concatenate(features, axis=0)

# ---------------- FUSE FEATURES ----------------
def fuse_features(residual_feats, freq_feats, lbp_feats, cnn_feats):
    return np.hstack([residual_feats, freq_feats, lbp_feats, cnn_feats])

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Load model
    model_data = load(MODEL_PATH)
    coarse_gmm = model_data["coarse_gmm"]
    fine_gmm = model_data["fine_gmm"]
    threshold = model_data["threshold"]

    # Collect image paths
    img_paths = [os.path.join(IMG_FOLDER,f) for f in os.listdir(IMG_FOLDER)
                 if f.lower().endswith((".png",".jpg",".jpeg"))]
    if not img_paths:
        print("No images found in folder:", IMG_FOLDER)
        exit()

    print(f"🔹 Found {len(img_paths)} images. Extracting features...")

    # ---------------- Residual (placeholder: use zeros, replace if you have residuals) ----------------
    residual_feats = np.zeros((len(img_paths), 1822), dtype=np.float32)

    # Frequency
    freq_feats = np.array([extract_frequency_features(residual_feats[i]) for i in range(len(img_paths))])

    # LBP
    lbp_feats = np.array([extract_lbp_hist(p) for p in img_paths])

    # CNN
    cnn_feats = extract_cnn_features_adaptive(img_paths)

    # Fuse
    fused_feats = fuse_features(residual_feats, freq_feats, lbp_feats, cnn_feats)

    # Predict
    scores = 0.5*coarse_gmm.score_samples(fused_feats) + 0.5*fine_gmm.score_samples(fused_feats)
    preds = (scores < threshold).astype(int)

    # Save results
    label_map = {0: "real", 1: "fake"}

    results = pd.DataFrame({
        "image": [os.path.basename(p) for p in img_paths],
        "score": scores,
        "prediction": [label_map[p] for p in preds]
    })

    results_path = "results/sample_img_predictions.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results.to_csv(results_path, index=False)
    print(f"✅ Predictions saved at {results_path}")
