import os, argparse, csv, glob
import numpy as np
from PIL import Image
import cv2

import torch
import pyiqa
from skimage.restoration import estimate_sigma


# ---------------- IO ----------------
def imread_rgb_float(path, max_size=None):
    img = Image.open(path).convert("RGB")
    if max_size is not None:
        w, h = img.size
        s = max(w, h)
        if s > max_size:
            scale = max_size / s
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return (np.asarray(img).astype(np.float32) / 255.0)

def clamp01(x):
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def rgb2y_float(x):
    return (0.299 * x[...,0] + 0.587 * x[...,1] + 0.114 * x[...,2]).astype(np.float32)

def to_torch_1x3(x):
    return torch.from_numpy(x).permute(2,0,1).unsqueeze(0)  # 1x3xHxW


# ---------------- metrics ----------------
def blockiness_score(x, block=8):
    """Higher = more block artifacts. We want LOW. Clamped to >=0."""
    y = rgb2y_float(x)
    H, W = y.shape
    xb = [k for k in range(block, W, block)]
    yb = [k for k in range(block, H, block)]
    if len(xb) == 0 or len(yb) == 0:
        return 0.0

    bv = np.mean([np.mean(np.abs(y[:, x0] - y[:, x0-1])) for x0 in xb])
    xv = [xx for xx in range(1, W) if (xx % block) != 0]
    ev = np.mean([np.mean(np.abs(y[:, xx] - y[:, xx-1])) for xx in xv])

    bh = np.mean([np.mean(np.abs(y[y0, :] - y[y0-1, :])) for y0 in yb])
    yh = [yy for yy in range(1, H) if (yy % block) != 0]
    eh = np.mean([np.mean(np.abs(y[yy, :] - y[yy-1, :])) for yy in yh])

    v = float((bv - ev) + (bh - eh))
    return max(0.0, v)

def noise_level_score(x):
    """Estimated noise level in [0,1]. Lower is better."""
    return float(np.mean(estimate_sigma(x, channel_axis=2)))

def sharpness_score(x, blur_sigma=1.0):
    """
    Noise-robust sharpness:
    - convert to uint8 gray
    - blur to suppress noise
    - Laplacian with supported depth (CV_16S), then var, then log1p
    """
    u8 = (clamp01(x) * 255.0 + 0.5).astype(np.uint8)
    y = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    if blur_sigma > 0:
        y = cv2.GaussianBlur(y, (0, 0), blur_sigma)
    lap = cv2.Laplacian(y, ddepth=cv2.CV_16S, ksize=3)
    lap = lap.astype(np.float32)
    return float(np.log1p(lap.var()))

def gradient_score(x, blur_sigma=1.0):
    """Noise-robust mean gradient magnitude (blur first)."""
    u8 = (clamp01(x) * 255.0 + 0.5).astype(np.uint8)
    y = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    if blur_sigma > 0:
        y = cv2.GaussianBlur(y, (0, 0), blur_sigma)
    y = y.astype(np.float32)

    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    return float(np.mean(mag))


def norm01(v):
    v = np.asarray(v, np.float32)
    if float(v.max() - v.min()) < 1e-12:
        return np.zeros_like(v)
    return (v - float(v.min())) / (float(v.max()) - float(v.min()))


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--max_size", type=int, default=768)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--blur_sigma", type=float, default=1.0)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.indir, args.pattern)))
    if not paths:
        raise SystemExit(f"No images found: {os.path.join(args.indir, args.pattern)}")

    niqe = pyiqa.create_metric("niqe", device=args.device)
    bris = pyiqa.create_metric("brisque", device=args.device)
    piqe = pyiqa.create_metric("piqe", device=args.device)

    rows = []
    with torch.no_grad():
        for p in paths:
            x = imread_rgb_float(p, max_size=args.max_size)
            xt = to_torch_1x3(x)

            NIQE = float(niqe(xt))   # lower better
            BRIS = float(bris(xt))   # lower better
            PIQE = float(piqe(xt))   # lower better
            BLK  = blockiness_score(x)       # lower better
            NOI  = noise_level_score(x)      # lower better
            SHP  = sharpness_score(x, blur_sigma=args.blur_sigma)  # higher better
            GRD  = gradient_score(x, blur_sigma=args.blur_sigma)   # higher better

            rows.append([os.path.basename(p), NIQE, BRIS, PIQE, BLK, NOI, SHP, GRD])

    # composite score (denoise-oriented)
    NIQE_v = np.array([r[1] for r in rows], np.float32)
    BRIS_v = np.array([r[2] for r in rows], np.float32)
    PIQE_v = np.array([r[3] for r in rows], np.float32)
    BLK_v  = np.array([r[4] for r in rows], np.float32)
    NOI_v  = np.array([r[5] for r in rows], np.float32)
    SHP_v  = np.array([r[6] for r in rows], np.float32)
    GRD_v  = np.array([r[7] for r in rows], np.float32)

    good_niqe  = 1 - norm01(NIQE_v)
    good_bris  = 1 - norm01(BRIS_v)
    good_piqe  = 1 - norm01(PIQE_v)
    good_blk   = 1 - norm01(BLK_v)
    good_noise = 1 - norm01(NOI_v)   # key: lower noise better
    good_shp   = norm01(SHP_v)
    good_grd   = norm01(GRD_v)

    score = (
        0.22*good_niqe +
        0.18*good_bris +
        0.08*good_piqe +
        0.22*good_noise +
        0.15*good_blk +
        0.10*good_shp +
        0.05*good_grd
    )

    out_rows = []
    for r, sc in zip(rows, score):
        out_rows.append(r + [float(sc)])
    out_rows.sort(key=lambda x: x[-1], reverse=True)

    out_csv = args.out_csv or os.path.join(args.indir, "metrics.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "Score", "NIQE", "BRISQUE", "PIQE", "Blockiness", "NoiseLevel", "Sharpness", "Grad"])
        for r in out_rows:
            w.writerow([r[0], r[-1], r[1], r[2], r[3], r[4], r[5], r[6], r[7]])

    # print
    print(f"{'file':>22s} {'Score':>8s} {'NIQE':>10s} {'BRISQUE':>10s} {'PIQE':>10s} {'Block':>10s} {'NoiseLv':>9s} {'Sharp':>10s} {'Grad':>8s}")
    for r in out_rows:
        print(f"{r[0]:>22s} {r[-1]:8.6f} {r[1]:10.6f} {r[2]:10.6f} {r[3]:10.6f} {r[4]:10.6f} {r[5]:9.6f} {r[6]:10.6f} {r[7]:8.6f}")

    print("\n[best]", out_rows[0][0])
    print("[saved]", out_csv)


if __name__ == "__main__":
    main()
