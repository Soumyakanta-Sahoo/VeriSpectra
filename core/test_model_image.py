import os
import argparse
import numpy as np
from PIL import Image
from joblib import load
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

# -----------------------
# Feature Extraction
# -----------------------

def load_and_preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    return img, img_np

def to_tensor_pil(img_pil):
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    return tf(img_pil)

def fft_features_gray(img_np):
    h,w = img_np.shape
    F = np.fft.fft2(img_np)
    F = np.fft.fftshift(F)
    mag = np.abs(F)
    total = mag.sum() + 1e-9
    cx, cy = w//2, h//2
    lx, ly = int(0.25*w/2), int(0.25*h/2)
    low = mag[cy-ly:cy+ly, cx-lx:cx+lx].sum()
    high = total - low
    hf_ratio = high / total
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X-cx)**2 + (Y-cy)**2)
    rmax = np.max(r)
    features = [hf_ratio]
    for k in range(1,5):
        mask = (r >= (k-1)*rmax/4) & (r < k*rmax/4)
        if mask.sum() == 0:
            features += [0.0, 0.0]
        else:
            band = mag[mask]
            features += [band.mean(), band.std()]
    return np.array(features, dtype=np.float32)

def noise_residual_stats(img_np):
    blurred = cv2.GaussianBlur(img_np, (7,7), 0)
    residual = (img_np.astype(np.float32) - blurred.astype(np.float32))
    stats = []
    for c in range(3):
        ch = residual[:,:,c]
        stats.append(ch.mean())
        stats.append(ch.std())
        stats.append(np.percentile(ch, 1))
        stats.append(np.percentile(ch, 99))
    return np.array(stats, dtype=np.float32)

def srm_filters_stats(img_np):
    kernels = [
        np.array([[0,0,0],[0,1,-1],[0,0,0]], dtype=np.float32),
        np.array([[ -1, 2, -1],[2, -4, 2],[-1,2,-1]], dtype=np.float32),
        np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
    ]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    feats = []
    for k in kernels:
        resp = cv2.filter2D(gray, -1, k)
        feats.append(resp.mean())
        feats.append(resp.std())
        feats.append(np.percentile(resp, 95))
    return np.array(feats, dtype=np.float32)

class BackboneEmbedding:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.net = models.efficientnet_b4(pretrained=True)
        self.net.classifier = nn.Identity()
        self.dim = 1792
        self.net.eval().to(self.device)
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    def extract(self, img_pil):
        tensor = self.tf(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(tensor)
            if out.ndim == 4:
                out = out.view(out.shape[0], -1)
            return out.cpu().numpy()[0]

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", required=True, help="Path to input image (jpg/png)")
    parser.add_argument("--model_path", default="models/gmm_model.pkl", help="Trained model (.pkl)")
    args = parser.parse_args()

    print("🔹 Loading image:", args.img_path)
    pil_img, img_np = load_and_preprocess(args.img_path)

    print("🔹 Extracting features...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = BackboneEmbedding(device=device)
    emb = backbone.extract(pil_img)
    fft = fft_features_gray(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY))
    noise = noise_residual_stats(img_np)
    srm = srm_filters_stats(img_np)

    extra = np.hstack([fft, noise, srm])
    extra_norm = (extra - extra.mean()) / (extra.std() + 1e-9)
    features = np.hstack([emb, extra_norm]).reshape(1, -1)

    print("🔹 Loading model:", args.model_path)
    model_dict = load(args.model_path)
    if "gmm" in model_dict:
        model = model_dict["gmm"]
        score = model.score_samples(features)[0]
    elif "ocsvm" in model_dict:
        model = model_dict["ocsvm"]
        score = model.decision_function(features)[0]
    else:
        raise ValueError("Model file must contain 'gmm' or 'ocsvm' key")

    threshold = model_dict["threshold"]
    prediction = "FAKE" if score < threshold else "REAL"

    print("\n🧪 Prediction:", prediction)
    print("📉 Score:", round(score, 4))
    print("🔻 Threshold:", round(threshold, 4))
