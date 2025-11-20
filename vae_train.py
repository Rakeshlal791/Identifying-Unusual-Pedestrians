# ==============================================================
# vae_train.py (updated to use dataset.py)
# ==============================================================

import os, math, json, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from dataset import PedestrianInstanceDataset  # ✅ updated import

# ==============================================================
# 2) Pose features: angles (cos,sin) + per-bone length + bbox (logW, logH)
# ==============================================================

def _has(kps, *names): return all(name in kps and kps[name].get("visible", 1) for name in names)
def _angle_wrt_vertical(p, q): return math.atan2(q["x"] - p["x"], -(q["y"] - p["y"]))
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

def build_angles_lengths_wh(k, bbox):
    feats_angles, lengths = [], {}
    for name, (a, b) in BONES_14.items():
        if _has(k, a, b):
            bot, top = k[a], k[b]
            th = _angle_wrt_vertical(bot, top)
            feats_angles += list(_cos_sin(th))
            lengths[name] = _l2(bot, top)
        else:
            feats_angles += [0.0, 0.0]
            lengths[name] = 0.0

    x0, y0, x1, y1 = bbox
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    return torch.tensor(feats_angles, dtype=torch.float32), lengths, w, h


# ==============================================================
# 3) Dataset wrapper — builds x0 only (no cond)
# ==============================================================

class PoseDataset(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds
        self.bone_names = list(BONES_14.keys())

        log_ws, log_hs = [], []
        bone_logs = {bn: [] for bn in self.bone_names}

        for _, target in base_ds:
            x0, y0, x1, y1 = target["bbox"]
            w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
            log_ws.append(math.log(w))
            log_hs.append(math.log(h))
            _, lens, _, _ = self._extract_raw(target)
            for bn, L in lens.items():
                bone_logs[bn].append(math.log(max(L, 1e-6)))

        self.mean_logW = np.mean(log_ws)
        self.std_logW  = np.std(log_ws) + 1e-6
        self.mean_logH = np.mean(log_hs)
        self.std_logH  = np.std(log_hs) + 1e-6

        self.bone_stats = {
            bn: {"mean_log": np.mean(vals), "std_log": np.std(vals) + 1e-6}
            for bn, vals in bone_logs.items()
        }

    def _extract_raw(self, target):
        ang_vec, lens, w, h = build_angles_lengths_wh(target["keypoints"], target["bbox"])
        return ang_vec, lens, w, h

    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        _, target = self.base[i]
        ang_vec, lens, w, h = self._extract_raw(target)

        z_len = []
        for bn in self.bone_names:
            s = self.bone_stats[bn]
            L = max(lens[bn], 1e-6)
            z = (math.log(L) - s["mean_log"]) / s["std_log"]
            z_len.append(z)

        z_w = (math.log(w) - self.mean_logW) / self.std_logW
        z_h = (math.log(h) - self.mean_logH) / self.std_logH
        x0 = torch.cat([ang_vec, torch.tensor(z_len, dtype=torch.float32),
                        torch.tensor([z_w, z_h], dtype=torch.float32)], dim=0)
        return x0


# ==============================================================
# 4) VAE
# ==============================================================

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim=16, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, x_dim)
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


def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    rec = F.l1_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kl, rec.item(), kl.item()


# ==============================================================
# 5) Training Loop
# ==============================================================

def train_vae(model, loader, device="cuda", epochs=500, lr=1e-3, beta_anneal=True, save_dir="./checkpoints_vae"):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = epochs * len(loader)
    step = 0
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {ep}/{epochs}", unit="batch")

        for x0 in pbar:
            x0 = x0.to(device)
            beta = min(1.0, step / total_steps) if beta_anneal else 1.0

            x_hat, mu, logvar = model(x0)
            loss, rec, kl = vae_loss(x0, x_hat, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", rec=f"{rec:.4f}", kl=f"{kl:.4f}", beta=f"{beta:.3f}")

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {ep}: mean loss={avg_loss:.6f}")

        torch.save({
            "model": model.state_dict(),
            "mean_logW": loader.dataset.mean_logW,
            "std_logW": loader.dataset.std_logW,
            "mean_logH": loader.dataset.mean_logH,
            "std_logH": loader.dataset.std_logH,
            "bone_stats": loader.dataset.bone_stats,
            "bone_names": loader.dataset.bone_names,
        }, os.path.join(save_dir, f"vae_ep{ep:03d}.pth"))

    print("💾 Training complete. Models saved in:", save_dir)


# ==============================================================
# 6) Main
# ==============================================================

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ROOT_ANN = "./annotations_orig"
    ROOT_IMG = "../A_OccDet/datasets/ECP/resources/data/imgs"
    SPLIT = "train"
    BATCH_SIZE = 2
    EPOCHS = 500
    LR = 1e-3

    base_ds = PedestrianInstanceDataset(ROOT_ANN, ROOT_IMG, split=SPLIT)
    wrapped = PoseDataset(base_ds)
    loader = DataLoader(wrapped, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    x0_sample = wrapped[0]
    data_dim = x0_sample.numel()
    model = VAE(data_dim, z_dim=16, hidden=256)

    print(f"🚀 Training VAE on {DEVICE} | {len(wrapped)} samples | feature dim={data_dim}")
    train_vae(model, loader, device=DEVICE, epochs=EPOCHS, lr=LR)
