import os, argparse, csv
import numpy as np
from PIL import Image
import imageio.v3 as iio

import cv2

from skimage.restoration import wiener
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


# ---------------- helpers: color ----------------
def rgb_to_ycrcb_u8(rgb01):
    """Convert RGB float [0,1] -> OpenCV YCrCb uint8."""
    u8 = (clamp01(rgb01) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)  # OpenCV order: Y, Cr, Cb
    return ycc

def ycrcb_u8_to_rgb01(ycc_u8):
    """Convert OpenCV YCrCb uint8 -> RGB float [0,1]."""
    bgr = cv2.cvtColor(ycc_u8, cv2.COLOR_YCrCb2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return clamp01(rgb)


# ---------------- padding to reduce boundary artifacts ----------------
def pad_reflect_rgb(x, pad):
    """Reflect-pad an RGB image to reduce FFT boundary artifacts in deconvolution."""
    return np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

def crop_rgb(x, pad):
    """Crop back to original size after padding."""
    if pad <= 0:
        return x
    return x[pad:-pad, pad:-pad, :]


# ---------------- motion blur PSF + blur ----------------
def motion_psf(length, angle_deg):
    """
    Create a motion blur PSF (line kernel) with enough support to avoid rotation clipping.
    - length: blur length in pixels
    - angle_deg: rotation angle in degrees (counterclockwise)
    """
    length = int(max(1, round(length)))

    # IMPORTANT: make kernel larger than length so rotation does not clip the line
    k = int(2 * length + 1)
    if k % 2 == 0:
        k += 1

    psf = np.zeros((k, k), dtype=np.float32)
    c = k // 2

    x0 = c - length // 2
    x1 = c + length // 2
    cv2.line(psf, (x0, c), (x1, c), 1.0, thickness=1)

    M = cv2.getRotationMatrix2D((c, c), float(angle_deg), 1.0)
    psf = cv2.warpAffine(
        psf, M, (k, k),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    psf /= (psf.sum() + 1e-12)
    return psf.astype(np.float32)

def apply_motion_blur(rgb01, psf, border=cv2.BORDER_REFLECT):
    """Convolve an RGB image with a PSF using OpenCV (reflect border reduces artifacts)."""
    out = np.zeros_like(rgb01, dtype=np.float32)
    for c in range(3):
        out[..., c] = cv2.filter2D(rgb01[..., c], ddepth=-1, kernel=psf, borderType=border)
    return clamp01(out)


# ---------------- Wiener deconvolution ----------------
def wiener_deblur_rgb(rgb01_blur, psf, balance):
    """
    Wiener deconvolution on each RGB channel.
    balance acts like a regularization term even when no noise is added.
    """
    out = np.zeros_like(rgb01_blur, dtype=np.float32)
    for c in range(3):
        out[..., c] = wiener(rgb01_blur[..., c], psf, balance=float(balance), clip=True).astype(np.float32)
    return clamp01(out)

def wiener_deblur_ycrcb(rgb01_blur, psf, balance_y, chroma_scale=0.0):
    """
    Wiener deconvolution mainly on luma (Y) only.
    - chroma_scale=0.0 is recommended to avoid color artifacts.
    """
    ycc = rgb_to_ycrcb_u8(rgb01_blur).astype(np.float32) / 255.0
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]

    Yd = wiener(Y, psf, balance=float(balance_y), clip=True).astype(np.float32)

    if chroma_scale > 0:
        bC = float(balance_y) * float(chroma_scale)
        Crd = wiener(Cr, psf, balance=bC, clip=True).astype(np.float32)
        Cbd = wiener(Cb, psf, balance=bC, clip=True).astype(np.float32)
    else:
        Crd, Cbd = Cr, Cb

    out_ycc = np.stack([Yd, Crd, Cbd], axis=-1)
    out_u8 = (clamp01(out_ycc) * 255.0 + 0.5).astype(np.uint8)
    return ycrcb_u8_to_rgb01(out_u8)


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

    # motion blur parameters
    ap.add_argument("--len", type=float, default=20.0, help="motion blur length (pixels)")
    ap.add_argument("--angle", type=float, default=15.0, help="motion blur angle (degrees)")

    # Wiener parameters
    ap.add_argument("--balance", type=float, default=3e-3,
                    help="Wiener regularization (~NSR). If you see stripes, INCREASE this.")
    ap.add_argument("--wiener_mode", default="ycrcb", choices=["rgb", "ycrcb"],
                    help="rgb=Wiener per RGB channel; ycrcb=Wiener mainly on luma (recommended)")
    ap.add_argument("--chroma_scale", type=float, default=0.0,
                    help="in ycrcb mode: chroma balance = balance * chroma_scale (0 disables chroma deblur)")

    # boundary handling
    ap.add_argument("--pad", type=int, default=None,
                    help="reflect padding size (pixels). Default: 2*len (recommended)")

    ap.add_argument("--outdir", default="results_motion_wiener")
    ap.add_argument("--max_size", type=int, default=768)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    clean = imread_rgb_float(args.input, max_size=args.max_size)
    imwrite_rgb_float(os.path.join(args.outdir, "clean.png"), clean)

    psf = motion_psf(args.len, args.angle)

    # synthesize motion blur only (no noise)
    blur = apply_motion_blur(clean, psf)
    blur_name = f"blur_len{args.len:g}_ang{args.angle:g}.png"
    imwrite_rgb_float(os.path.join(args.outdir, blur_name), blur)

    # reflect padding before deconvolution to reduce periodic stripe artifacts
    pad = int(4 * args.len) if args.pad is None else int(args.pad)
    pad = max(0, pad)

    blur_in = pad_reflect_rgb(blur, pad) if pad > 0 else blur

    if args.wiener_mode == "rgb":
        deblur_pad = wiener_deblur_rgb(blur_in, psf, balance=args.balance)
    else:
        deblur_pad = wiener_deblur_ycrcb(blur_in, psf, balance_y=args.balance, chroma_scale=args.chroma_scale)

    deblur = crop_rgb(deblur_pad, pad) if pad > 0 else deblur_pad
    imwrite_rgb_float(os.path.join(args.outdir, "wiener.png"), deblur)

    # metrics (reference is input image)
    rows = []
    m_blur = eval_metrics(clean, blur)
    rows.append(["blur", m_blur["psnr"], m_blur["ssim"]])
    print(f"{'blur':8s} PSNR={m_blur['psnr']:.3f}  SSIM={m_blur['ssim']:.5f}")

    m_wiener = eval_metrics(clean, deblur)
    rows.append(["wiener", m_wiener["psnr"], m_wiener["ssim"]])
    print(f"{'wiener':8s} PSNR={m_wiener['psnr']:.3f}  SSIM={m_wiener['ssim']:.5f}")

    save_csv(os.path.join(args.outdir, "metrics.csv"), rows)

    print("\n[saved]")
    print(" - clean.png")
    print(f" - {blur_name}")
    print(" - wiener.png")
    print(" - metrics.csv")
    print("in:", args.outdir)


if __name__ == "__main__":
    main()


# Example:
# python -m DIP.deblur.motion_wiener \
#   --input "data/self_collected/sample.jpg" \
#   --len 5 --angle 10 \
#   --balance 3e-3 \
#   --wiener_mode ycrcb --chroma_scale 0.0 \
#   --outdir results/Motion_wiener
