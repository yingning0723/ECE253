import os, argparse
import numpy as np
from PIL import Image
import imageio.v3 as iio
import cv2

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
    """Read image as RGB float32 in [0,1]. Optionally resize to max_size."""
    img = Image.open(path).convert("RGB")
    if max_size is not None:
        w, h = img.size
        s = max(w, h)
        if s > max_size:
            scale = max_size / s
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return (np.asarray(img).astype(np.float32) / 255.0)


def imwrite_rgb_float(path, x):
    """Write RGB float32 [0,1] to uint8 image."""
    x = np.clip(x, 0.0, 1.0)
    iio.imwrite(path, (x * 255.0 + 0.5).astype(np.uint8))


def clamp01(x):
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def rgb2y(x):
    """Luma approximation in [0,1]."""
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.float32)


def save_as_jpeg_and_reload(rgb01, jpg_path, quality=20, subsampling=2):
    """
    Save an RGB float image as JPEG and reload it as RGB float.
    This creates a JPEG-compressed version of the input for deblocking experiments.

    quality: 1~95 (lower => stronger artifacts)
    subsampling: 0 (4:4:4), 1 (4:2:2), 2 (4:2:0)  [Pillow convention]
    """
    rgb01 = clamp01(rgb01)
    u8 = (rgb01 * 255.0 + 0.5).astype(np.uint8)
    pil = Image.fromarray(u8, mode="RGB")
    pil.save(jpg_path, format="JPEG", quality=int(quality), subsampling=int(subsampling), optimize=True)
    # reload
    out = Image.open(jpg_path).convert("RGB")
    return (np.asarray(out).astype(np.float32) / 255.0)


# ---------------- classic blocks ----------------
def guided_filter_gray(I, p, r=8, eps=1e-3):
    """
    Guided filter for grayscale guidance I and input p.
    NOTE: This implementation uses an O(H) loop for box filter; OK for moderate image sizes.
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    h, w = I.shape

    def box(img):
        ii = np.pad(img, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
        y0 = np.clip(np.arange(h) - r, 0, h)
        y1 = np.clip(np.arange(h) + r + 1, 0, h)
        x0 = np.clip(np.arange(w) - r, 0, w)
        x1 = np.clip(np.arange(w) + r + 1, 0, w)

        out = np.empty((h, w), np.float32)
        for i in range(h):
            A = ii[y1[i], :] - ii[y0[i], :]
            out[i, :] = A[x1] - A[x0]
        return out

    N = box(np.ones((h, w), np.float32))
    mean_I = box(I) / N
    mean_p = box(p) / N
    mean_Ip = box(I * p) / N
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box(I * I) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box(a) / N
    mean_b = box(b) / N
    q = mean_a * I + mean_b
    return q


def guided_filter_rgb(x, r=8, eps=1e-3):
    """Guided filter each RGB channel using luma as guidance."""
    I = rgb2y(x)
    out = np.zeros_like(x, np.float32)
    for c in range(3):
        out[..., c] = guided_filter_gray(I, x[..., c], r=r, eps=eps)
    return clamp01(out)


def tv_denoise(x, weight=0.10, iters=200):
    out = denoise_tv_chambolle(x, weight=weight, channel_axis=2, max_num_iter=iters)
    return clamp01(out.astype(np.float32))


def nlm_denoise(x, patch=5, dist=7, h_factor=0.8, fast_mode=True):
    sigma = float(np.mean(estimate_sigma(x, channel_axis=2)))
    h = (h_factor * sigma) if sigma > 0 else 0.06
    out = denoise_nl_means(
        x, h=h, sigma=sigma,
        patch_size=patch, patch_distance=dist,
        fast_mode=fast_mode, channel_axis=2
    )
    return clamp01(out.astype(np.float32))


def wavelet_denoise(x, wavelet="db2", method="BayesShrink", mode="soft"):
    sigma = float(np.mean(estimate_sigma(x, channel_axis=2)))
    out = denoise_wavelet(
        x, sigma=sigma, wavelet=wavelet,
        method=method, mode=mode,
        channel_axis=2, rescale_sigma=True
    )
    return clamp01(out.astype(np.float32))


def bilateral_filter(x, d=7, sigma_color=50, sigma_space=5):
    u8 = (clamp01(x) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    out = cv2.bilateralFilter(bgr, d=d, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return clamp01(rgb)


def unsharp_mask(x, radius=1.2, amount=0.6):
    """Light sharpening to bring back edges after smoothing."""
    u8 = (clamp01(x) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR).astype(np.float32)
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=radius, sigmaY=radius)
    sharp = bgr + float(amount) * (bgr - blur)
    rgb = cv2.cvtColor(np.clip(sharp, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return clamp01(rgb)


# ---------------- scoring (no-reference) ----------------
def blockiness_score(x, block=8):
    """Higher means more block artifacts. We want LOW."""
    y = rgb2y(x)
    H, W = y.shape

    xb = [k for k in range(block, W, block)]
    if len(xb) == 0:
        return 0.0
    bv = np.mean([np.mean(np.abs(y[:, x0] - y[:, x0 - 1])) for x0 in xb])

    xv = [xx for xx in range(1, W) if (xx % block) != 0]
    ev = np.mean([np.mean(np.abs(y[:, xx] - y[:, xx - 1])) for xx in xv])

    yb = [k for k in range(block, H, block)]
    bh = np.mean([np.mean(np.abs(y[yy, :] - y[yy - 1, :])) for yy in yb])

    yh = [yy for yy in range(1, H) if (yy % block) != 0]
    eh = np.mean([np.mean(np.abs(y[yy, :] - y[yy - 1, :])) for yy in yh])

    return float((bv - ev) + (bh - eh))


def sharpness_score(x):
    """Higher means sharper (but too high can mean noisy/ringing)."""
    u8 = (clamp01(x) * 255.0 + 0.5).astype(np.uint8)
    y = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(y, cv2.CV_64F)
    return float(lap.var())


def gradient_score(x):
    """Mean gradient magnitude; higher often correlates with more edges."""
    u8 = (clamp01(x) * 255.0 + 0.5).astype(np.uint8)
    y = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(mag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="results_combo")
    ap.add_argument("--max_size", type=int, default=768)

    # JPEG degradation controls
    ap.add_argument("--jpeg_q", type=int, default=20, help="JPEG quality (1~95). Lower => stronger artifacts.")
    ap.add_argument("--jpeg_subsample", type=int, default=2, choices=[0, 1, 2],
                    help="JPEG chroma subsampling: 0=4:4:4, 1=4:2:2, 2=4:2:0")

    # Optional clean reference for PSNR/SSIM (original image recommended)
    ap.add_argument("--ref", default=None, help="optional clean reference image for PSNR/SSIM selection")

    # Quick knobs
    ap.add_argument("--strength", default="medium", choices=["light", "medium", "strong"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load original image
    orig = imread_rgb_float(args.input, max_size=args.max_size)
    imwrite_rgb_float(os.path.join(args.outdir, "orig.png"), orig)

    # Create JPEG-compressed input and use it as the pipeline input
    jpg_path = os.path.join(args.outdir, f"input_q{args.jpeg_q}.jpg")
    y = save_as_jpeg_and_reload(orig, jpg_path, quality=args.jpeg_q, subsampling=args.jpeg_subsample)
    imwrite_rgb_float(os.path.join(args.outdir, f"jpeg_q{args.jpeg_q}.png"), y)

    # Optional reference image for PSNR/SSIM
    ref = None
    if args.ref is not None:
        ref = imread_rgb_float(args.ref, max_size=args.max_size)
    else:
        # If ref is not provided, use the original image as reference (common for synthetic JPEG experiments)
        ref = orig

    # Force same size
    H = min(ref.shape[0], y.shape[0])
    W = min(ref.shape[1], y.shape[1])
    ref = ref[:H, :W, :]
    y = y[:H, :W, :]

    # Parameter presets (tuned for JPEG artifacts)
    if args.strength == "light":
        guided_r, guided_eps = 6, 2e-3
        tv_w, tv_it = 0.06, 150
        nlm_hf = 0.6
        un_amt = 0.4
    elif args.strength == "strong":
        guided_r, guided_eps = 10, 1e-3
        tv_w, tv_it = 0.14, 250
        nlm_hf = 1.0
        un_amt = 0.6
    else:
        guided_r, guided_eps = 8, 1e-3
        tv_w, tv_it = 0.10, 200
        nlm_hf = 0.8
        un_amt = 0.5

    # Candidate pipelines (classic combinations)
    def pipe_guided_tv_unsharp(img):
        x = guided_filter_rgb(img, r=guided_r, eps=guided_eps)
        x = tv_denoise(x, weight=tv_w, iters=tv_it)
        x = unsharp_mask(x, radius=1.2, amount=un_amt)
        return x

    def pipe_wavelet_guided(img):
        x = wavelet_denoise(img, wavelet="db2")
        x = guided_filter_rgb(x, r=guided_r, eps=guided_eps)
        return x

    def pipe_nlm_tv(img):
        x = nlm_denoise(img, patch=5, dist=7, h_factor=nlm_hf)
        x = tv_denoise(x, weight=tv_w * 0.7, iters=tv_it)
        return x

    def pipe_bilateral_tv(img):
        x = bilateral_filter(img, d=7, sigma_color=60, sigma_space=5)
        x = tv_denoise(x, weight=tv_w * 0.8, iters=tv_it)
        return x

    def pipe_tv_only(img):
        return tv_denoise(img, weight=tv_w, iters=tv_it)

    def pipe_guided_only(img):
        return guided_filter_rgb(img, r=guided_r, eps=guided_eps)

    candidates = [
        ("guided_tv_unsharp", pipe_guided_tv_unsharp),
        ("wavelet_guided",    pipe_wavelet_guided),
        ("nlm_tv",            pipe_nlm_tv),
        ("bilateral_tv",      pipe_bilateral_tv),
        ("tv_only",           pipe_tv_only),
        ("guided_only",       pipe_guided_only),
    ]

    outs = {}
    scores = {}

    # Evaluate each candidate
    for name, fn in candidates:
        out = fn(y)
        outs[name] = out
        imwrite_rgb_float(os.path.join(args.outdir, f"{name}.png"), out)

        p = psnr(ref, out, data_range=1.0)
        s = ssim(ref, out, channel_axis=2, data_range=1.0)
        scores[name] = (p, s)

        print(f"{name:16s} PSNR={p:6.2f} SSIM={s:6.4f}")

    # Select best by reference (SSIM priority, then PSNR)
    best = sorted(scores.items(), key=lambda kv: (kv[1][1], kv[1][0]), reverse=True)[0][0]
    print("[best by ref]", best)

    imwrite_rgb_float(os.path.join(args.outdir, "best_combo.png"), outs[best])
    print("[saved] best_combo.png in", args.outdir)


if __name__ == "__main__":
    main()

# python -m DIP.JPEG.dip_combined \
#   --input data/self_collected/JPEG_sample.jpg \
#   --jpeg_q 20 \
#   --strength medium \
#   --outdir results/JPEG_combo
