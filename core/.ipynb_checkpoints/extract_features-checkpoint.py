"""
extract_features.py

Usage:
    python extract_features.py --input_dir data/raw --out_dir data/features --split train,val,test --batch 64

What it does:
 - Scans input_dir/{real,fake}/{split} for images
 - Extracts:
    * backbone embedding (EfficientNet-b4 or ResNet50 fallback)
    * frequency features (FFT high-frequency energy + band stats)
    * noise residual stats (img - gaussian_blur(img))
    * SRM-like filter pooled responses
 - Saves per-split .npz files with arrays: features, labels, paths
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import cv2
from glob import glob
from tqdm import tqdm

# -----------------------
# Utils / feature funcs
# -----------------------
def imread_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img)

def to_tensor_pil(img_pil):
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    return tf(img_pil)

def fft_features_gray(img_np):
    # img_np: HxW (grayscale), float [0,255]
    h,w = img_np.shape
    F = np.fft.fft2(img_np)
    F = np.fft.fftshift(F)
    mag = np.abs(F)
    total = mag.sum() + 1e-9
    # define a central low-freq square (25% of size)
    cx, cy = w//2, h//2
    lx, ly = int(0.25*w/2), int(0.25*h/2)
    low = mag[cy-ly:cy+ly, cx-lx:cx+lx].sum()
    high = total - low
    hf_ratio = high / total
    # banded statistics: split into 4 annular bands
    features = [hf_ratio]
    # create radial bins
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X-cx)**2 + (Y-cy)**2)
    rmax = np.max(r)
    for k in range(1,5):
        mask = (r >= (k-1)*rmax/4) & (r < k*rmax/4)
        if mask.sum() == 0:
            features += [0.0, 0.0]
        else:
            band = mag[mask]
            features += [band.mean(), band.std()]
    return np.array(features, dtype=np.float32)  # length 1 + 4*2 = 9

def noise_residual_stats(img_np):
    # img_np: HxWx3, uint8
    # simple denoise via Gaussian blur -> residual = original - blurred
    blurred = cv2.GaussianBlur(img_np, (7,7), 0)
    residual = (img_np.astype(np.float32) - blurred.astype(np.float32))
    # compute channel-wise mean/std and histogram moments
    stats = []
    for c in range(3):
        ch = residual[:,:,c]
        stats.append(ch.mean())
        stats.append(ch.std())
        stats.append(np.percentile(ch, 1))
        stats.append(np.percentile(ch, 99))
    return np.array(stats, dtype=np.float32)  # 12 dims

def srm_filters_stats(img_np):
    # three high-pass kernels (simple approximations)
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
    return np.array(feats, dtype=np.float32)  # 9 dims

# -----------------------
# Backbone wrapper
# -----------------------
class BackboneEmbedding:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # try EfficientNet-B4 first
        try:
            self.net = models.efficientnet_b4(pretrained=True)
            # remove classifier -> use feature before classifier
            # torchvision's effnet b4 has classifier[1] linear; classifier is Sequential
            if hasattr(self.net, 'classifier'):
                # Save the in_features
                self.net.classifier = nn.Identity()
                self.dim = 1792
            else:
                # fallback
                raise Exception("unexpected efficientnet structure")
        except Exception as e:
            print("EfficientNet-b4 not available, falling back to ResNet50:", e)
            rn = models.resnet50(pretrained=True)
            modules = list(rn.children())[:-1]
            self.net = nn.Sequential(*modules)
            self.dim = 2048
        self.net.eval().to(self.device)

    @torch.no_grad()
    def embed_batch(self, pil_batch):
        # pil_batch: list of PIL images (or already transformed tensors stacked)
        # create tensor
        tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        tensors = torch.stack([tf(p) for p in pil_batch], dim=0).to(self.device)
        with torch.no_grad():
            out = self.net(tensors)
            if out.ndim == 4:
                out = out.view(out.shape[0], -1)
            out = out.cpu().numpy()
        return out  # shape (B, dim)

# -----------------------
# Main extract loop
# -----------------------
def gather_image_paths(base_dir, splits):
    # expects base_dir/{real,fake}/{split} structure
    items = []
    for label_name, lbl in [('real',0), ('fake',1)]:
        for s in splits:
            d = os.path.join(base_dir, label_name, s)
            if not os.path.exists(d): continue
            paths = glob(os.path.join(d, '*'))
            for p in paths:
                if not p.lower().endswith(('.jpg','.jpeg','.png')): continue
                items.append((p, lbl, s))
    return items

def process_and_save(base_dir, out_dir, splits, batch_size=64):
    os.makedirs(out_dir, exist_ok=True)
    items = gather_image_paths(base_dir, splits)
    # group by split
    split_map = {s:[] for s in splits}
    for p,l,s in items:
        split_map[s].append((p,l))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = BackboneEmbedding(device=device)

    for s in splits:
        entries = split_map[s]
        if len(entries) == 0:
            print(f"no entries for split {s}, skipping")
            continue
        N = len(entries)
        feats_all = []
        labels_all = []
        paths_all = []
        # process in batches for backbone embedding
        for i in tqdm(range(0, N, batch_size), desc=f"split={s}"):
            batch = entries[i:i+batch_size]
            pil_batch = []
            # precompute per-image cheap features
            fft_feats = []
            noise_feats = []
            srm_feats = []
            for p, lbl in batch:
                img_np = imread_rgb(p)
                # small safety: ensure shape
                if img_np.ndim != 3:
                    img_np = np.stack([img_np]*3, axis=-1)
                pil = Image.fromarray(img_np)
                pil_batch.append(pil)

                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
                fft_feats.append(fft_features_gray(gray))
                noise_feats.append(noise_residual_stats(img_np))
                srm_feats.append(srm_filters_stats(img_np))

            # backbone embeddings
            emb = backbone.embed_batch(pil_batch)  # (B, d)
            # stack extra features
            extra = np.hstack([np.vstack(fft_feats), np.vstack(noise_feats), np.vstack(srm_feats)])
            # normalize extra per-batch to reduce scale issues
            extra = (extra - extra.mean(axis=0, keepdims=True)) / (extra.std(axis=0, keepdims=True) + 1e-9)
            # concatenate
            fused = np.hstack([emb, extra])  # shape (B, dim + extra_dim)
            for j, (p,l) in enumerate(batch):
                feats_all.append(fused[j])
                labels_all.append(l)
                paths_all.append(p)

        feats_arr = np.vstack(feats_all).astype(np.float32)
        labels_arr = np.array(labels_all, dtype=np.int32)
        paths_arr = np.array(paths_all, dtype=object)
        out_path = os.path.join(out_dir, f"features_{s}.npz")
        np.savez_compressed(out_path, features=feats_arr, labels=labels_arr, paths=paths_arr)
        print("Saved", out_path, "shape:", feats_arr.shape)

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="DiffFake-SingleImage/data/raw", help="base raw dir")
    parser.add_argument("--out_dir", type=str, default="DiffFake-SingleImage/data/features", help="where feature .npz saved")
    parser.add_argument("--split", type=str, default="train,val,test", help="comma separated splits to process")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    splits = args.split.split(",")
    process_and_save(args.input_dir, args.out_dir, splits, batch_size=args.batch)
