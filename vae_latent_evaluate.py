# ==============================================================
# vae_evaluate.py
# Evaluates Unconditional VAE model and visualizes latent space
# (supports PCA / UMAP / both)
# ==============================================================

import os, math, csv, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# UMAP (install via pip install umap-learn)
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️ UMAP not installed. Run `pip install umap-learn` to enable UMAP visualization.")

from synthetic_combined_test_dataset import SyntheticCombinedTestDataset

# --------------------------------------------------------------
# Geometry helpers (same as training)
# --------------------------------------------------------------
def _has(kps, *names):
    return all(name in kps and kps[name] is not None for name in names)

def _angle_wrt_vertical(p, q):
    return math.atan2(q["x"] - p["x"], -(q["y"] - p["y"]))

def _cos_sin(th): return math.cos(th), math.sin(th)
def _l2(p, q): return math.hypot(q["x"] - p["x"], q["y"] - p["y"])

BONES_14 = {
    "shoulder_width": ("shoulder_left", "shoulder_right"),
    "hip_width": ("hip_left", "hip_right"),
    "torso_left": ("hip_left", "shoulder_left"),
    "torso_right": ("hip_right", "shoulder_right"),
    "neck_left": ("nose", "shoulder_left"),
    "neck_right": ("nose", "shoulder_right"),
    "forearm_left": ("wrist_left", "elbow_left"),
    "upperarm_left": ("elbow_left", "shoulder_left"),
    "forearm_right": ("wrist_right", "elbow_right"),
    "upperarm_right": ("elbow_right", "shoulder_right"),
    "lowerleg_left": ("ankle_left", "knee_left"),
    "upperleg_left": ("knee_left", "hip_left"),
    "lowerleg_right": ("ankle_right", "knee_right"),
    "upperleg_right": ("knee_right", "hip_right"),
}

def build_angle_len_wh_features(k, bbox, bone_stats, mean_logW, std_logW, mean_logH, std_logH):
    feats, names = [], []
    lens = {}
    for name, (a, b) in BONES_14.items():
        if _has(k, a, b):
            bot, top = k[a], k[b]
            th = _angle_wrt_vertical(bot, top)
            feats += list(_cos_sin(th))
            lens[name] = _l2(bot, top)
        else:
            feats += [0, 0]
            lens[name] = 0.0
        names.append(name)

    z_lens = []
    for name in BONES_14:
        s = bone_stats[name]
        val = max(lens[name], 1e-6)
        z = (math.log(val) - s["mean_log"]) / s["std_log"]
        z_lens.append(z)

    x0, y0, x1, y1 = bbox
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    z_w = (math.log(w) - mean_logW) / std_logW
    z_h = (math.log(h) - mean_logH) / std_logH

    full_vec = torch.tensor(feats + z_lens + [z_w, z_h], dtype=torch.float32)
    return full_vec, names


# --------------------------------------------------------------
# VAE model
# --------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, x_dim, z_dim=16, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x):
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_hat = self.dec(z)
        return x_hat, mu, logvar


# --------------------------------------------------------------
# Region weighting
# --------------------------------------------------------------
def region_weight_vector(names, include_bbox=True):
    wts = []
    for n in names:
        if "leg" in n or "torso" in n or "hip" in n:
            w = 2.0
        elif "arm" in n:
            w = 1.0
        elif "neck" in n or "shoulder_width" in n:
            w = 0.5
        else:
            w = 1.0
        wts += [w, w]
    for n in names:
        if "leg" in n or "torso" in n or "hip" in n:
            w = 2.0
        elif "arm" in n:
            w = 1.0
        elif "neck" in n or "shoulder_width" in n:
            w = 0.5
        else:
            w = 1.0
        wts.append(w)
    if include_bbox:
        wts += [4.0, 4.0]
    return torch.tensor(wts, dtype=torch.float32)


# --------------------------------------------------------------
# Collate
# --------------------------------------------------------------
def collate_examples(batch, mean_logW, std_logW, mean_logH, std_logH, bone_stats):
    vecs, labels, metas = [], [], []
    names_ref = None
    for (_, tgt, _) in batch:
        vec, names = build_angle_len_wh_features(
            tgt["keypoints"], tgt["bbox"],
            bone_stats, mean_logW, std_logW, mean_logH, std_logH,
        )
        vecs.append(vec)
        labels.append(int(tgt["synthetic"]))
        metas.append({"frame_path": tgt["frame_path"], "id": tgt.get("id")})
        if names_ref is None:
            names_ref = names
    return (
        torch.stack(vecs, 0),
        torch.tensor(labels),
        names_ref,
        metas,
    )


# --------------------------------------------------------------
# Visualization helper
# --------------------------------------------------------------
def visualize_latent_space(mus_all, labels_all, mode="both"):
    if mode not in ["none", "pca", "umap", "both"]:
        print(f"⚠️ Unknown viz mode '{mode}', skipping latent visualization.")
        return

    if mode in ["pca", "both"]:
        print("🧭 Visualizing latent μ distribution (PCA)...")
        pca = PCA(n_components=2)
        z2d_pca = pca.fit_transform(mus_all)
        plt.figure(figsize=(7, 6))
        if len(np.unique(labels_all)) > 1:
            plt.scatter(z2d_pca[:, 0], z2d_pca[:, 1],
                        c=labels_all, cmap="coolwarm", s=8, alpha=0.6)
            plt.colorbar(label="Label (0=Normal, 1=Anomaly)")
        else:
            plt.scatter(z2d_pca[:, 0], z2d_pca[:, 1], s=8, alpha=0.6)
        plt.title("Latent μ distribution (PCA 2D projection)")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.grid(True); plt.axis("equal")
        plt.show()

    if mode in ["umap", "both"] and HAS_UMAP:
        print("🌌 Visualizing latent μ distribution (UMAP)...")
        umap_2d = UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42)
        z2d_umap = umap_2d.fit_transform(mus_all)
        plt.figure(figsize=(7, 6))
        if len(np.unique(labels_all)) > 1:
            plt.scatter(z2d_umap[:, 0], z2d_umap[:, 1],
                        c=labels_all, cmap="coolwarm", s=8, alpha=0.6)
            plt.colorbar(label="Label (0=Normal, 1=Anomaly)")
        else:
            plt.scatter(z2d_umap[:, 0], z2d_umap[:, 1], s=8, alpha=0.6)
        plt.title("Latent Space (UMAP 2D projection)")
        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
        plt.grid(True); plt.axis("equal")
        plt.show()

    # Histograms of each latent dim
    print("📈 Plotting per-dimension histograms...")
    fig, axes = plt.subplots(4, 4, figsize=(10, 8))
    axes = axes.flatten()
    for i in range(min(16, mus_all.shape[1])):
        axes[i].hist(mus_all[:, i], bins=40, color="steelblue", alpha=0.8)
        axes[i].set_title(f"z{i}")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn_root", type=str, default="./syn_images/all")
    ap.add_argument("--normal_root", type=str, default="./annotations_orig/test")
    ap.add_argument("--ckpt", type=str, default="checkpoints_vae/vae_ep004.pth")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out_csv", type=str, default="vae_anomalies.csv")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--viz", type=str, default="both",
                    choices=["none", "pca", "umap", "both"],
                    help="Visualize latent μ distribution (none/pca/umap/both)")
    args = ap.parse_args()

    print(f"🔹 Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mean_logW, std_logW = ckpt["mean_logW"], ckpt["std_logW"]
    mean_logH, std_logH = ckpt["mean_logH"], ckpt["std_logH"]
    bone_stats = ckpt["bone_stats"]

    test_ds = SyntheticCombinedTestDataset(args.syn_root, args.normal_root)
    sample_path, sample_tgt, _ = test_ds[0]
    sample_vec, names = build_angle_len_wh_features(
        sample_tgt["keypoints"], sample_tgt["bbox"],
        bone_stats, mean_logW, std_logW, mean_logH, std_logH,
    )
    data_dim = sample_vec.numel()

    model = VAE(data_dim)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval().to(args.device)

    wts = region_weight_vector(names, include_bbox=True).to(args.device)
    loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda b: collate_examples(
            b, mean_logW, std_logW, mean_logH, std_logH, bone_stats
        ),
    )

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------
    scores_all, labels_all, mus_all = [], [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Scoring (VAE)")
        for vecs, labels, names_ref, _ in pbar:
            vecs = vecs.to(args.device)
            x_hat, mu, logvar = model(vecs)
            per_dim_err = torch.abs(vecs - x_hat)
            weighted = (per_dim_err * wts).mean(dim=1)
            scores_all.append(weighted.cpu().numpy())
            labels_all.append(labels.numpy())
            mus_all.append(mu.cpu().numpy())

    scores_all = np.concatenate(scores_all)
    labels_all = np.concatenate(labels_all)
    mus_all = np.concatenate(mus_all)

    # ----------------------------------------------------------
    # Latent visualization
    # ----------------------------------------------------------
    if args.viz != "none":
        visualize_latent_space(mus_all, labels_all, mode=args.viz)

    # ----------------------------------------------------------
    # Thresholding + metrics
    # ----------------------------------------------------------
    if args.threshold is None and len(np.unique(labels_all)) > 1:
        fpr, tpr, thr = roc_curve(labels_all, scores_all)
        THRESH = float(thr[np.argmax(tpr - fpr)])
        print(f"🔎 Auto threshold: {THRESH:.6f}")
    else:
        THRESH = args.threshold if args.threshold is not None else float(np.median(scores_all))

    preds = (scores_all > THRESH).astype(int)

    if len(np.unique(labels_all)) > 1:
        prec, rec, f1, _ = precision_recall_fscore_support(labels_all, preds, average="binary")
        cm = confusion_matrix(labels_all, preds)
        auc = roc_auc_score(labels_all, scores_all)
        print("\n=== EVALUATION ===")
        print(f"Threshold: {THRESH:.3f}")
        print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  AUC: {auc:.3f}")
        print("Confusion matrix:\n", cm)
    else:
        print("⚠️ Only one class present; skipping metrics.")

    # ----------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------
    out_path = args.out_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_path", "label", "score"])
        idx = 0
        for batch_idx, (_, labels, _, metas) in enumerate(loader):
            for i in range(len(labels)):
                writer.writerow([
                    metas[i]["frame_path"],
                    int(labels[i]),
                    float(scores_all[idx])
                ])
                idx += 1
    print(f"\n💾 Saved per-sample anomaly scores to: {out_path}")

    # ----------------------------------------------------------
    # Score histogram
    # ----------------------------------------------------------
    plt.figure()
    plt.hist(scores_all[labels_all == 0], bins=60, alpha=0.6, label="Normal")
    plt.hist(scores_all[labels_all == 1], bins=60, alpha=0.6, label="Anomalous")
    plt.axvline(THRESH, linestyle="--", color="k", label="Threshold")
    plt.legend()
    plt.xlabel("Weighted reconstruction error")
    plt.ylabel("Count")
    plt.title("Unconditional VAE anomaly scores")
    plt.show()


if __name__ == "__main__":
    main()
