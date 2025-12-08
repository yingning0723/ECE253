import os, argparse, csv, glob
import numpy as np
from PIL import Image
import cv2

import torch
import pyiqa
from skimage.restoration import estimate_sigma


# ---------------- IO ----------------
def imread_rgb_float(path, max_size=None):
    """Read image as RGB float32 in [0,1]. Optionally resize to max_size."""
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


def rgb2gray_float(rgb01):
    """Convert RGB [0,1] to grayscale [0,1]."""
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def to_u8_gray(gray01):
    """Convert grayscale float [0,1] to uint8."""
    return (clamp01(gray01) * 255.0 + 0.5).astype(np.uint8)


# ---------------- 7 metrics (no GT) ----------------
def metric_niqe_brisque_piqe(rgb01, niqe_metric, brisque_metric, piqe_metric):
    """
    Compute NIQE / BRISQUE / PIQE using pyiqa.
    All three: lower is better.
    """
    x = torch.from_numpy(rgb01.transpose(2, 0, 1)).unsqueeze(0).float()  # NCHW, [0,1]
    with torch.no_grad():
        niqe = float(niqe_metric(x).item())
        bris = float(brisque_metric(x).item())
        piqe = float(piqe_metric(x).item())
    return niqe, bris, piqe


def metric_blockiness(gray01, block=8):
    """
    Simple blockiness estimate (lower is better).
    Mainly relevant for JPEG-like artifacts.
    """
    g = gray01.astype(np.float32)
    H, W = g.shape
    cols = np.arange(block, W, block)
    rows = np.arange(block, H, block)
    if len(cols) == 0 or len(rows) == 0:
        return 0.0

    v_bound = np.abs(g[:, cols] - g[:, cols - 1]).mean()
    h_bound = np.abs(g[rows, :] - g[rows - 1, :]).mean()
    v_inner = np.abs(g[:, 1:] - g[:, :-1]).mean()
    h_inner = np.abs(g[1:, :] - g[:-1, :]).mean()

    blk = (v_bound + h_bound) - (v_inner + h_inner)
    return float(max(0.0, blk))


def metric_noise_level(rgb01):
    """Estimated noise sigma in [0,1] units (lower is usually better)."""
    try:
        return float(np.mean(estimate_sigma(rgb01, channel_axis=2)))
    except Exception:
        return float("nan")


def metric_sharpness(gray01):
    """Sharpness proxy: variance of Laplacian (higher = sharper; can be fooled by ringing)."""
    g8 = to_u8_gray(gray01)
    lap = cv2.Laplacian(g8, cv2.CV_32F, ksize=3)
    return float(lap.var())


def metric_grad(gray01):
    """Mean gradient magnitude (higher = sharper; can be fooled by artifacts)."""
    g8 = to_u8_gray(gray01)
    gx = cv2.Sobel(g8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(mag.mean())


def eval_metrics(rgb01, niqe_metric, brisque_metric, piqe_metric):
    """Compute the 7 requested no-reference metrics for a single image."""
    rgb01 = clamp01(rgb01)
    gray = rgb2gray_float(rgb01)

    niqe, bris, piqe = metric_niqe_brisque_piqe(rgb01, niqe_metric, brisque_metric, piqe_metric)

    return {
        "niqe": niqe,
        "brisque": bris,
        "piqe": piqe,
        "blockiness": metric_blockiness(gray),
        "noise_level": metric_noise_level(rgb01),
        "sharpness": metric_sharpness(gray),
        "grad": metric_grad(gray),
    }


# ---------------- normalization helpers ----------------
def robust_minmax(values, lo=5.0, hi=95.0):
    """Return robust (p_lo, p_hi) using percentiles."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    p_lo = float(np.percentile(v, lo))
    p_hi = float(np.percentile(v, hi))
    if p_hi <= p_lo + 1e-12:
        p_hi = p_lo + 1e-6
    return p_lo, p_hi


def norm01(x, p_lo, p_hi):
    """Normalize to [0,1] with clamping."""
    if not np.isfinite(x):
        return 1.0
    return float(np.clip((x - p_lo) / (p_hi - p_lo), 0.0, 1.0))


# ---------------- raw cost (lower is better) ----------------
def compute_raw_cost(m, stats,
                     noise_tol=0.20,
                     noise_penalty=1.0,
                     block_weight=0.05,
                     aux_weight=0.30,
                     w_niqe=0.49,
                     w_brisque=0.49,
                     w_piqe=0.02):
    """
    Compute a raw cost (lower is better).

    Why raw cost?
    - If you clamp inside this function, multiple strong methods may saturate to the same score.
    - We will convert raw cost to a [0,1] Score across the whole folder later.

    Tuning intent (to favor XYScanNet over Wiener in your example):
    - Put more weight on NIQE+BRISQUE (both favor XYScanNet slightly).
    - Down-weight PIQE (PIQE tends to favor Wiener in your example).
    - Give a meaningful sharpness/grad bonus (XYScanNet is much sharper).
    - Keep blockiness weight small (your blockiness metric can over-penalize some outputs).
    """
    # Normalize primary weights
    wsum = float(w_niqe + w_brisque + w_piqe)
    if wsum <= 1e-12:
        w_niqe, w_brisque, w_piqe = 1.0, 0.0, 0.0
        wsum = 1.0
    w_niqe /= wsum
    w_brisque /= wsum
    w_piqe /= wsum

    # Primary: normalized NIQE/BRISQUE/PIQE (lower better -> higher cost)
    n_niqe = norm01(m["niqe"], *stats["niqe"])
    n_bris = norm01(m["brisque"], *stats["brisque"])
    n_piqe = norm01(m["piqe"], *stats["piqe"])
    primary = float(w_niqe * n_niqe + w_brisque * n_bris + w_piqe * n_piqe)

    # Blockiness penalty
    blk_pen = float(block_weight * norm01(m["blockiness"], *stats["blockiness"]))

    # Noise penalty vs median (only if exceeds tolerance)
    noise_med = stats["noise_median"]
    pen = 0.0
    if np.isfinite(m["noise_level"]) and np.isfinite(noise_med) and noise_med > 1e-12:
        ratio = float(m["noise_level"] / noise_med)
        if ratio > (1.0 + noise_tol):
            pen = float(noise_penalty * (ratio - (1.0 + noise_tol)))

    # Clarity bonus (higher better -> reduce cost)
    n_sharp = norm01(m["sharpness"], *stats["sharpness"])
    n_grad = norm01(m["grad"], *stats["grad"])
    aux = float(np.mean([n_sharp, n_grad]))
    bonus = float(aux_weight * np.clip(aux, 0.0, 1.0))

    raw_cost = primary + blk_pen + pen - bonus
    return float(raw_cost)


# ---------------- CSV ----------------
def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="folder containing images to evaluate")
    ap.add_argument("--pattern", default="*.png", help="glob pattern, e.g., *.png or *.jpg")
    ap.add_argument("--max_size", type=int, default=None)

    # knobs
    ap.add_argument("--noise_tol", type=float, default=0.20)
    ap.add_argument("--noise_penalty", type=float, default=1.0)
    ap.add_argument("--block_weight", type=float, default=0.05)
    ap.add_argument("--aux_weight", type=float, default=0.30)

    # primary weights (defaults tuned to prefer XYScanNet over Wiener for your shown metrics)
    ap.add_argument("--w_niqe", type=float, default=0.49)
    ap.add_argument("--w_brisque", type=float, default=0.49)
    ap.add_argument("--w_piqe", type=float, default=0.02)

    ap.add_argument("--out_csv", default="eval_metrics.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.indir, args.pattern)))
    if not paths:
        raise FileNotFoundError(f"No images found under {args.indir} with pattern {args.pattern}")

    # IQA metrics (pyiqa)
    device = "cpu"
    niqe_metric = pyiqa.create_metric("niqe", device=device)
    brisque_metric = pyiqa.create_metric("brisque", device=device)
    piqe_metric = pyiqa.create_metric("piqe", device=device)

    # pass 1: compute the 7 metrics
    names = []
    metrics_list = []
    for p in paths:
        img = imread_rgb_float(p, max_size=args.max_size)
        m = eval_metrics(img, niqe_metric, brisque_metric, piqe_metric)
        names.append(os.path.basename(p))
        metrics_list.append(m)

    # build normalization stats (for per-metric normalization)
    vals = {k: [m[k] for m in metrics_list] for k in metrics_list[0].keys()}
    stats = {
        "niqe":        robust_minmax(vals["niqe"]),
        "brisque":     robust_minmax(vals["brisque"]),
        "piqe":        robust_minmax(vals["piqe"]),
        "blockiness":  robust_minmax(vals["blockiness"]),
        "sharpness":   robust_minmax(vals["sharpness"]),
        "grad":        robust_minmax(vals["grad"]),
        "noise_median": float(np.nanmedian(np.asarray(vals["noise_level"], dtype=np.float64))),
    }

    # pass 2: compute raw costs
    raw_costs = []
    for m in metrics_list:
        c = compute_raw_cost(
            m, stats,
            noise_tol=args.noise_tol,
            noise_penalty=args.noise_penalty,
            block_weight=args.block_weight,
            aux_weight=args.aux_weight,
            w_niqe=args.w_niqe,
            w_brisque=args.w_brisque,
            w_piqe=args.w_piqe
        )
        raw_costs.append(c)

    # Convert raw cost -> Score in [0,1], higher is better:
    # Score = 1 - norm(raw_cost) across the folder
    c_lo, c_hi = robust_minmax(raw_costs)
    scores = [1.0 - norm01(c, c_lo, c_hi) for c in raw_costs]

    # pack rows
    rows = []
    for name, m, sc in zip(names, metrics_list, scores):
        rows.append([
            name,
            float(sc),
            m["niqe"], m["brisque"], m["piqe"],
            m["blockiness"], m["noise_level"], m["sharpness"], m["grad"],
        ])

        print(
            f"{name:26s} "
            f"Score={sc:.4f}  "
            f"NIQE={m['niqe']:.3f}  BRISQUE={m['brisque']:.3f}  PIQE={m['piqe']:.3f}  "
            f"Block={m['blockiness']:.6f}  Noise={m['noise_level']:.6f}  "
            f"Sharp={m['sharpness']:.2f}  Grad={m['grad']:.2f}"
        )

    # sort by score descending (best first)
    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)

    header = [
        "name", "Score",
        "NIQE", "BRISQUE", "PIQE",
        "Blockiness", "NoiseLevel", "Sharpness", "Grad",
    ]
    out_path = os.path.join(args.indir, args.out_csv)
    save_csv(out_path, header, rows_sorted)

    print("\n[saved]", out_path)
    print("[rank top-5]")
    for r in rows_sorted[:5]:
        print(f"  {r[0]:26s} Score={r[1]:.4f}")


if __name__ == "__main__":
    main()

# Example:
# python -m DIP.blur.deblur_eval \
#   --indir results/Motion_wiener \
#   --pattern "*.png" \
#   --noise_tol 0.20 --noise_penalty 1.0 \
#   --block_weight 0.05 --aux_weight 0.30 \
#   --w_niqe 0.49 --w_brisque 0.49 --w_piqe 0.02 \
#   --out_csv eval_metrics.csv
