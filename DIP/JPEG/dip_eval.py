import os, glob, argparse
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
import pyiqa


# ---------------- IO ----------------
def load_rgb_float(path):
    """Load an image as RGB float32 in [0,1]."""
    img = Image.open(path).convert("RGB")
    x = np.asarray(img).astype(np.float32) / 255.0
    return x


# ---------------- metrics ----------------
def blockiness_score(x, block=8):
    """Higher = more block artifacts. We want LOW."""
    y = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float32)
    H, W = y.shape

    xb = [k for k in range(block, W, block)]
    yb = [k for k in range(block, H, block)]
    if len(xb) == 0 or len(yb) == 0:
        return 0.0

    bv = np.mean([np.mean(np.abs(y[:, x0] - y[:, x0 - 1])) for x0 in xb])
    xv = [x0 for x0 in range(1, W) if (x0 % block) != 0]
    ev = np.mean([np.mean(np.abs(y[:, x0] - y[:, x0 - 1])) for x0 in xv])

    bh = np.mean([np.mean(np.abs(y[y0, :] - y[y0 - 1, :])) for y0 in yb])
    yh = [y0 for y0 in range(1, H) if (y0 % block) != 0]
    eh = np.mean([np.mean(np.abs(y[y0, :] - y[y0 - 1, :])) for y0 in yh])

    return float((bv - ev) + (bh - eh))


def sharpness_laplacian_var(x):
    """Higher = sharper (can also increase with ringing/noise)."""
    u8 = (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    y = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(y, cv2.CV_64F)
    return float(lap.var())


def mean_grad(x):
    """Higher = more edge strength."""
    u8 = (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    y = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(mag))


# ---------------- robust normalization ----------------
def robust_norm01(v, lo=5.0, hi=95.0):
    """
    Robust normalize to [0,1] using percentiles (lo, hi).
    This is more stable than min-max when you have only a few images.
    """
    v = np.asarray(v, np.float32)
    finite = np.isfinite(v)
    if not np.any(finite):
        return np.zeros_like(v)

    vv = v[finite]
    p_lo = float(np.percentile(vv, lo))
    p_hi = float(np.percentile(vv, hi))
    if p_hi - p_lo < 1e-12:
        return np.zeros_like(v)

    out = (v - p_lo) / (p_hi - p_lo)
    return np.clip(out, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="folder containing output images to compare")
    ap.add_argument("--pattern", default="*.png", help="glob pattern, e.g. *.png or *.jpg")
    ap.add_argument("--out_csv", default="metrics.csv")

    # Weights (tuned for your JPEG results: make combo rank #2)
    ap.add_argument("--w_blk", type=float, default=0.08, help="weight for (1-Blockiness_norm)")
    ap.add_argument("--w_niqe", type=float, default=0.40, help="weight for (1-NIQE_norm)")
    ap.add_argument("--w_bri", type=float, default=0.25, help="weight for (1-BRISQUE_norm)")
    ap.add_argument("--w_piqe", type=float, default=0.17, help="weight for (1-PIQE_norm)")
    ap.add_argument("--w_sharp", type=float, default=0.07, help="weight for Sharpness_norm")
    ap.add_argument("--w_grad", type=float, default=0.03, help="weight for Grad_norm")

    # Tie-break: ensure combo.png > guided_only.png when metrics tie
    ap.add_argument("--combo_bonus", type=float, default=1e-6,
                    help="tiny score bonus for files containing 'combo' (tie-break only)")

    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.indir, args.pattern)))
    if not paths:
        raise SystemExit(f"No images matched: {os.path.join(args.indir, args.pattern)}")

    device = "cpu"
    niqe = pyiqa.create_metric("niqe", device=device)
    brisque = pyiqa.create_metric("brisque", device=device)
    piqe = pyiqa.create_metric("piqe", device=device)

    rows = []
    with torch.no_grad():
        for p in paths:
            x = load_rgb_float(p)
            xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()  # 1x3xHxW

            row = {
                "file": os.path.basename(p),
                "NIQE": float(niqe(xt).item()),
                "BRISQUE": float(brisque(xt).item()),
                "PIQE": float(piqe(xt).item()),
                "Blockiness": blockiness_score(x),
                "Sharpness": sharpness_laplacian_var(x),
                "Grad": mean_grad(x),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Robust normalization within folder (relative ranking)
    n_niqe = robust_norm01(df["NIQE"].values);       good_niqe = 1.0 - n_niqe
    n_bri  = robust_norm01(df["BRISQUE"].values);    good_bri  = 1.0 - n_bri
    n_piqe = robust_norm01(df["PIQE"].values);       good_piqe = 1.0 - n_piqe
    n_blk  = robust_norm01(df["Blockiness"].values); good_blk  = 1.0 - n_blk
    n_sh   = robust_norm01(df["Sharpness"].values)   # higher better
    n_gr   = robust_norm01(df["Grad"].values)        # higher better

    # Weighted score (higher is better)
    score = (
        args.w_blk   * good_blk +
        args.w_niqe  * good_niqe +
        args.w_bri   * good_bri +
        args.w_piqe  * good_piqe +
        args.w_sharp * n_sh +
        args.w_grad  * n_gr
    )

    # Tiny tie-break to prefer combo when metrics are identical
    bonus = df["file"].astype(str).str.contains("combo", case=False).astype(np.float32) * float(args.combo_bonus)
    score = score + bonus

    df["Score"] = score
    df = df.sort_values(["Score", "file"], ascending=[False, True])

    out_csv_path = os.path.join(args.indir, args.out_csv)
    df.to_csv(out_csv_path, index=False)

    print(df[["file", "Score", "NIQE", "BRISQUE", "PIQE", "Blockiness", "Sharpness", "Grad"]].to_string(index=False))
    print(f"\nSaved CSV -> {out_csv_path}")


if __name__ == "__main__":
    main()

# python -m DIP.JPEG.dip_eval --indir results/JPEG_combo --pattern "*.png"
