import os, argparse, csv
import numpy as np
from PIL import Image
import imageio.v3 as iio

import cv2  # NEW

from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_nl_means,
    estimate_sigma,
    denoise_wavelet,
)
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


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

def imwrite_rgb_float(path, x):
    x = np.clip(x, 0.0, 1.0)
    iio.imwrite(path, (x * 255.0 + 0.5).astype(np.uint8))

def clamp01(x):
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# ---------------- noise synth ----------------
def add_gaussian_noise(x, sigma_255, seed=0):
    rng = np.random.default_rng(seed)
    sigma = float(sigma_255) / 255.0
    n = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return clamp01(x + n)


# ---------------- helpers: color + unsharp ----------------
def rgb_to_ycbcr_u8(rgb01):
    u8 = (clamp01(rgb01) * 255.0 + 0.5).astype(np.uint8)
    # OpenCV expects BGR
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)  # note: OpenCV uses YCrCb order
    return ycc  # uint8

def ycbcr_u8_to_rgb01(ycc_u8):
    bgr = cv2.cvtColor(ycc_u8, cv2.COLOR_YCrCb2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return clamp01(rgb)

def unsharp_on_luma(rgb01, radius=1.1, amount=0.35):
    """
    Light unsharp masking on Y (luma) only.
    radius: Gaussian sigma
    amount: strength
    """
    ycc = rgb_to_ycbcr_u8(rgb01).astype(np.float32)
    Y = ycc[..., 0]
    blur = cv2.GaussianBlur(Y, (0, 0), sigmaX=radius, sigmaY=radius)
    sharp = Y + float(amount) * (Y - blur)
    ycc[..., 0] = np.clip(sharp, 0, 255)
    return ycbcr_u8_to_rgb01(ycc.astype(np.uint8))


# ---------------- denoisers ----------------
def denoise_bm3d_rgb(x, sigma_255):
    """Original: BM3D on RGB (more likely to oversoften)."""
    import bm3d as _bm3d
    sigma = float(sigma_255) / 255.0
    out = _bm3d.bm3d(x, sigma_psd=sigma, stage_arg=_bm3d.BM3DStages.ALL_STAGES)
    return clamp01(out.astype(np.float32))

def denoise_bm3d_ycbcr(x, sigmaY_255, sigmaC_scale=0.5, do_unsharp=True, unsharp_radius=1.1, unsharp_amount=0.35):
    """
    Recommended: BM3D on luma Y strongly, chroma weakly.
    - sigmaY_255: noise sigma for Y channel (same as injected sigma usually)
    - sigmaC_scale: chroma sigma = sigmaY * scale (0~1). 0.0 means no chroma denoise.
    - do_unsharp: light sharpen on Y after denoise to recover crispness
    """
    import bm3d as _bm3d
    ycc = rgb_to_ycbcr_u8(x).astype(np.float32) / 255.0  # to [0,1]
    Y = ycc[..., 0]
    Cr = ycc[..., 1]
    Cb = ycc[..., 2]

    sigY = float(sigmaY_255) / 255.0
    sigC = float(sigmaY_255 * sigmaC_scale) / 255.0

    Yd = _bm3d.bm3d(Y, sigma_psd=sigY, stage_arg=_bm3d.BM3DStages.ALL_STAGES).astype(np.float32)
    if sigmaC_scale > 0:
        Crd = _bm3d.bm3d(Cr, sigma_psd=sigC, stage_arg=_bm3d.BM3DStages.ALL_STAGES).astype(np.float32)
        Cbd = _bm3d.bm3d(Cb, sigma_psd=sigC, stage_arg=_bm3d.BM3DStages.ALL_STAGES).astype(np.float32)
    else:
        Crd, Cbd = Cr, Cb

    out_ycc = np.stack([Yd, Crd, Cbd], axis=-1)
    out_u8 = (clamp01(out_ycc) * 255.0 + 0.5).astype(np.uint8)
    out = ycbcr_u8_to_rgb01(out_u8)

    if do_unsharp:
        out = unsharp_on_luma(out, radius=unsharp_radius, amount=unsharp_amount)

    return out

def denoise_wavelet_baseline(x, wavelet="db2"):
    sig = float(np.mean(estimate_sigma(x, channel_axis=2)))
    out = denoise_wavelet(
        x, sigma=sig, wavelet=wavelet,
        method="BayesShrink", mode="soft",
        channel_axis=2, rescale_sigma=True
    )
    return clamp01(out.astype(np.float32))

def denoise_nlm_baseline(x, patch=5, dist=7, h_factor=0.8):
    sig = float(np.mean(estimate_sigma(x, channel_axis=2)))
    h = (h_factor * sig) if sig > 0 else 0.06
    out = denoise_nl_means(
        x, h=h, sigma=sig,
        patch_size=patch, patch_distance=dist,
        fast_mode=True, channel_axis=2
    )
    return clamp01(out.astype(np.float32))

def denoise_tv_baseline(x, weight=0.10, iters=200):
    out = denoise_tv_chambolle(x, weight=weight, channel_axis=2, max_num_iter=iters)
    return clamp01(out.astype(np.float32))


# ---------------- evaluation ----------------
def eval_metrics(clean, test):
    return {
        "psnr": float(psnr(clean, test, data_range=1.0)),
        "ssim": float(ssim(clean, test, channel_axis=2, data_range=1.0)),
    }

def save_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "psnr", "ssim"])
        w.writerows(rows)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input image (treated as clean-ish)")
    ap.add_argument("--sigma", type=float, default=25.0, help="Gaussian sigma in [0,255]")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--outdir", default="results_gauss_all")
    ap.add_argument("--max_size", type=int, default=768)

    # BM3D sigma (default uses injected --sigma)
    ap.add_argument("--bm3d_sigma", type=float, default=None)

    # NEW: choose BM3D mode
    ap.add_argument("--bm3d_mode", default="ycbcr", choices=["rgb", "ycbcr"],
                    help="rgb=BM3D on RGB; ycbcr=BM3D on luma with weak chroma denoise + optional unsharp")

    # NEW: ycbcr bm3d knobs
    ap.add_argument("--chroma_scale", type=float, default=0.5,
                    help="chroma sigma = sigmaY * chroma_scale (0 disables chroma denoise)")
    ap.add_argument("--unsharp", action="store_true", help="enable unsharp on luma after BM3D (recommended)")
    ap.add_argument("--unsharp_radius", type=float, default=1.1)
    ap.add_argument("--unsharp_amount", type=float, default=0.35)

    # Wavelet/NLM/TV params
    ap.add_argument("--wavelet", default="db2")
    ap.add_argument("--nlm_patch", type=int, default=5)
    ap.add_argument("--nlm_dist", type=int, default=7)
    ap.add_argument("--nlm_hf", type=float, default=0.8)
    ap.add_argument("--tv_weight", type=float, default=0.10)
    ap.add_argument("--tv_iters", type=int, default=200)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    clean = imread_rgb_float(args.input, max_size=args.max_size)
    imwrite_rgb_float(os.path.join(args.outdir, "clean.png"), clean)

    noisy = add_gaussian_noise(clean, args.sigma, seed=args.seed)
    noisy_name = f"noisy_sigma{args.sigma:g}_seed{args.seed}.png"
    imwrite_rgb_float(os.path.join(args.outdir, noisy_name), noisy)

    bm3d_sigma = args.sigma if args.bm3d_sigma is None else args.bm3d_sigma

    outs = {}
    # --- BM3D (improved) ---
    if args.bm3d_mode == "rgb":
        outs["bm3d"] = denoise_bm3d_rgb(noisy, bm3d_sigma)
    else:
        outs["bm3d"] = denoise_bm3d_ycbcr(
            noisy, sigmaY_255=bm3d_sigma,
            sigmaC_scale=args.chroma_scale,
            do_unsharp=args.unsharp,
            unsharp_radius=args.unsharp_radius,
            unsharp_amount=args.unsharp_amount,
        )

    outs["wavelet"] = denoise_wavelet_baseline(noisy, wavelet=args.wavelet)
    outs["nlm"] = denoise_nlm_baseline(noisy, patch=args.nlm_patch, dist=args.nlm_dist, h_factor=args.nlm_hf)
    outs["tv"] = denoise_tv_baseline(noisy, weight=args.tv_weight, iters=args.tv_iters)

    rows = []

    def save_eval(name, img):
        imwrite_rgb_float(os.path.join(args.outdir, f"{name}.png"), img)
        m = eval_metrics(clean, img)
        rows.append([name, m["psnr"], m["ssim"]])
        print(f"{name:8s} PSNR={m['psnr']:.3f}  SSIM={m['ssim']:.5f}")

    m_noisy = eval_metrics(clean, noisy)
    rows.append(["noisy", m_noisy["psnr"], m_noisy["ssim"]])
    print(f"{'noisy':8s} PSNR={m_noisy['psnr']:.3f}  SSIM={m_noisy['ssim']:.5f}")

    for k in ["bm3d", "wavelet", "nlm", "tv"]:
        save_eval(k, outs[k])

    save_csv(os.path.join(args.outdir, "metrics.csv"), rows)

    print("\n[saved]")
    print(" - clean.png")
    print(f" - {noisy_name}")
    print(" - bm3d.png / wavelet.png / nlm.png / tv.png")
    print(" - metrics.csv")
    print("in:", args.outdir)


if __name__ == "__main__":
    main()


# python -m DIP.noise.denoise_BM3D \
#   --input "data/self_collected/Gaussian_sample1.jpg" \
#   --sigma 25 --seed 0 \
#   --bm3d_mode ycbcr --chroma_scale 0.5 --unsharp \
#   --outdir results/Gaussian_noise
