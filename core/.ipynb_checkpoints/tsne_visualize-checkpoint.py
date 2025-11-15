# tsne_visualize.py

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_data(npz_path, csv_path=None):
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    paths = data["paths"]

    df = pd.DataFrame({
        "path": paths,
        "true_label": labels
    })

    if csv_path:
        df_pred = pd.read_csv(csv_path)
        df_pred["predicted_label"] = df_pred["predicted_label"].map({"REAL": 0, "FAKE": 1})
        df["filename"] = df["path"].apply(lambda p: os.path.basename(str(p)))
        df_pred["filename"] = df_pred["path"].apply(lambda p: os.path.basename(str(p)))
        df = df.merge(df_pred[["filename", "predicted_label"]], on="filename", how="left")
        df["error"] = df["true_label"] != df["predicted_label"]
    else:
        df["predicted_label"] = None
        df["error"] = False

    return features, df

def plot_tsne(features, df, out_path, show_errors=True):
    print("🔹 Running t-SNE on features...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, init='pca')
    X_2d = tsne.fit_transform(features)
    df["tsne_x"] = X_2d[:, 0]
    df["tsne_y"] = X_2d[:, 1]

    print("🔹 Plotting...")
    plt.figure(figsize=(10, 8))
    if show_errors and "error" in df:
        # plot correct
        correct = df[df["error"] == False]
        plt.scatter(correct["tsne_x"], correct["tsne_y"],
                    c=correct["true_label"].map({0: 'blue', 1: 'red'}),
                    label="Correct", s=10, alpha=0.5)

        # plot incorrect
        incorrect = df[df["error"] == True]
        plt.scatter(incorrect["tsne_x"], incorrect["tsne_y"],
                    c='black', label="Misclassified", s=15, marker='x')
    else:
        plt.scatter(df["tsne_x"], df["tsne_y"],
                    c=df["true_label"].map({0: 'blue', 1: 'red'}),
                    s=10, alpha=0.5)

    plt.title("t-SNE of Feature Embeddings (Real=Blue, Fake=Red)")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"✅ Saved plot to {out_path}")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Path to .npz feature file")
    parser.add_argument("--csv", help="Optional: path to prediction csv file")
    parser.add_argument("--out", default="results/tsne_plot.png", help="Output plot path")
    args = parser.parse_args()

    features, df = load_data(args.npz, args.csv)
    plot_tsne(features, df, args.out)
