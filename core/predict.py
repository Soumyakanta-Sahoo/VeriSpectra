# predict.py

import os
import argparse
import numpy as np
from joblib import load
from tqdm import tqdm

# -----------------------
# Utils
# -----------------------

def load_model(path="models/gmm_model.pkl"):
    model_data = load(path)
    return model_data["gmm"], model_data["threshold"]

def predict_from_features(features, gmm, threshold):
    scores = gmm.score_samples(features)
    preds = (scores < threshold).astype(int)
    return preds, scores

def load_feature_file(path):
    data = np.load(path, allow_pickle=True)
    return data['features'], data.get('paths', None)

def print_results(preds, scores, paths=None):
    for i in range(len(preds)):
        label = "FAKE" if preds[i] == 1 else "REAL"
        score = scores[i]
        if paths is not None:
            print(f"[{label}] {paths[i]}  (score={score:.4f})")
        else:
            print(f"[{label}] (score={score:.4f})")

# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_file", type=str, help="Path to .npz file containing features")
    parser.add_argument("--output_file", type=str, help="Optional: path to save predictions as .csv")
    args = parser.parse_args()

    if not args.feature_file:
        print("⚠️ Please provide --feature_file path to a .npz file")
        exit(1)

    print("🔹 Loading model...")
    gmm, threshold = load_model()

    print("🔹 Loading features...")
    features, paths = load_feature_file(args.feature_file)

    print("🔹 Predicting...")
    preds, scores = predict_from_features(features, gmm, threshold)

    print("🔹 Results:")
    print_results(preds, scores, paths)

    if args.output_file:
        print(f"\n📄 Saving results to {args.output_file}")
        import pandas as pd
        df = pd.DataFrame({
            "path": paths if paths is not None else [f"sample_{i}" for i in range(len(preds))],
            "score": scores,
            "predicted_label": ["FAKE" if p == 1 else "REAL" for p in preds]
        })
        df.to_csv(args.output_file, index=False)
        print("✅ Done.")
