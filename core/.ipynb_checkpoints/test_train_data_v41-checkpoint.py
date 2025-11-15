# test_gmm_v41_folder.py — Test Hierarchical GMM + LR on a folder of images

import os, sys, time, json
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from joblib import load
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import pandas as pd

# ---------------- CONFIG ----------------
MODEL_PATH = "models/hier_gmm_model_v41.pkl"  # trained model
CACHE_DIR = "cache_v41"
RESULT_CSV = "results/test_predictions_v41.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_BATCH = 8
N_BANDS = 6  # FFT bands
LBP_RADII = [1,2,3]
LBP_P = 8
LBP_N_BINS = 10

# ---------------- UTILS ----------------
def log(msg): print(msg)

# FFT features
def extract_frequency_features(img_array, n_bands=N_BANDS):
    if img_array.ndim == 3:
        img_array = rgb2gray(img_array)
    h, w = img_array.shape
    F = np.fft.fft2(img_array)
    F = np.fft.fftshift(F)
    mag = np.abs(F)
    total = mag.sum() + 1e-9
    cx, cy = w//2, h//2
    lx, ly = max(1, min(w//8, w//2-1)), max(1, min(h//8, h//2-1))
    low = mag[max(0,cy-ly):min(h,cy+ly), max(0,cx-lx):min(w,cx+lx)].sum()
    high = total - low
    hf_ratio = high / total
    Y, X = np.ogrid[:h,:w]
    r = np.sqrt((X-cx)**2 + (Y-cy)**2)
    rmax = np.max(r) if np.max(r)>0 else 1.0
    features = [hf_ratio]
    for k in range(1, n_bands+1):
        mask = (r >= (k-1)*rmax/n_bands) & (r < k*rmax/n_bands)
        if mask.sum()==0:
            features += [0.0, 0.0]
        else:
            band = mag[mask]
            features += [float(band.mean()), float(band.std())]
    return np.array(features, dtype=np.float32)

# LBP features
def extract_lbp_hist(img_path, radii=LBP_RADII, P=LBP_P, n_bins=LBP_N_BINS):
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
        return np.zeros(len(radii)*n_bins, dtype=np.float32)

# CNN features
def extract_cnn_features(img_paths, base_batch=BASE_BATCH):
    device = DEVICE
    batch_size = base_batch
    try:
        model = models.efficientnet_b0(weights="IMAGENET1K_V1").to(device)
    except:
        device="cpu"
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.eval()
    preprocess = T.Compose([T.Resize((224,224)), T.ToTensor(),
                            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    features = []
    i, total = 0, len(img_paths)
    with torch.no_grad():
        pbar = tqdm(total=total, desc="CNN Features")
        while i<total:
            batch_paths = img_paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                try: img = Image.open(p).convert("RGB"); imgs.append(preprocess(img))
                except: imgs.append(torch.zeros(3,224,224))
            batch_tensor = torch.stack(imgs).to(device)
            try:
                if hasattr(model, "features"):
                    out = model.features(batch_tensor).mean([2,3]).cpu().numpy()
                else:
                    out = model(batch_tensor)
                    out = out.cpu().numpy() if isinstance(out,torch.Tensor) else out
                features.append(out)
            except:
                features.append(np.zeros((len(batch_paths),1280), dtype=np.float32))
            finally:
                i += batch_size
                pbar.update(len(batch_paths))
                del batch_tensor
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        pbar.close()
    return np.concatenate(features, axis=0)

# Z-score fusion
def fuse_features(res_feats, freq_feats, lbp_feats, cnn_feats):
    def zscore(x): return (x - x.mean(0,keepdims=True)) / (x.std(0,keepdims=True)+1e-9)
    res_feats, freq_feats, lbp_feats, cnn_feats = map(zscore, [res_feats,freq_feats,lbp_feats,cnn_feats])
    return np.hstack([res_feats,freq_feats,lbp_feats,cnn_feats])

# ---------------- MAIN ----------------
if __name__=="__main__":
    if len(sys.argv)<2:
        log("Usage: python test_gmm_v41_folder.py /path/to/images_folder")
        sys.exit(1)

    img_folder = sys.argv[1]
    img_paths = sorted([os.path.join(img_folder,f) for f in os.listdir(img_folder)
                        if f.lower().endswith((".png",".jpg",".jpeg"))])
    if not img_paths:
        log("No images found in folder."); sys.exit(1)

    log(f"🔹 Extracting features from {len(img_paths)} images...")
    X_res = np.zeros((len(img_paths), 2048), dtype=np.float32)  # placeholder if residuals not computed
    X_freq = np.array([extract_frequency_features(imread(p)) for p in tqdm(img_paths, desc="FFT")])
    X_lbp = np.array([extract_lbp_hist(p) for p in tqdm(img_paths, desc="LBP")])
    X_cnn = extract_cnn_features(img_paths)

    X_fused = fuse_features(X_res, X_freq, X_lbp, X_cnn)

    log("🔹 Loading trained model...")
    model_data = load(MODEL_PATH)
    gmm = model_data["gmm"]
    lr = model_data["lr"]
    threshold = model_data.get("threshold",0.5)

    scores = -gmm.score_samples(X_fused)
    probs = lr.predict_proba(scores.reshape(-1,1))[:,1]
    preds = (probs>=threshold).astype(int)

    # ---------------- SAVE & PRINT ----------------
    df = pd.DataFrame({"image":img_paths, "pred_label":preds, "probability":probs})
    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
    df.to_csv(RESULT_CSV, index=False)
    log(f"✅ Predictions saved to {RESULT_CSV}")

    log("🔹 Sample predictions:")
    for i in range(min(10,len(df))):
        print(f"{df.image[i]} → {'FAKE' if df.pred_label[i]==1 else 'REAL'} ({df.probability[i]:.4f})")
