# ECE 253 – Digital Image Processing Course Project  
**Title:** Classical DIP vs Vision State-Space Models for Single-Image Restoration (XYScanNet)

## 1. Project Overview
This repository contains our course project for **ECE 253 (Digital Image Processing), Fall 2025**.

We study how well **modern Vision State-Space Models (VSSMs)**—specifically **XYScanNet**—restore **real-world degraded images**, compared with **classical digital image processing (DIP)** pipelines.

**Focus**
- **Primary task:** single-image **motion deblurring**, **Gaussian noise** and **JPEG compression artifacts**
- **Classical baselines:** Wiener / Richardson–Lucy deconvolution, TV regularization, guided filtering, NLM/BM3D-style denoising variants, CLAHE + sharpening, etc.
- **Learning-based model:** **XYScanNet**, a U-Net–style restoration network built with **Vision State-Space Modules**
- **Data:** public datasets(GoPro) + our **self-collected real-scene distortion dataset**

**Goal**
We want to understand when:
1) classical pipelines (with tunable priors/parameters) are sufficient,  
2) a fixed trained model generalizes, and  
3) domain gap forces adaptation via **pre/post-processing (DIP)** or **fine-tuning**.

---

## 2. Repository Structure
```text
ECE253/
├── DIP/                                           # Classical DIP baselines (deblur/denoise/deartifact)
│   ├── gaussian_noise/                            # Gaussian denoising scripts (tv/nlm/wavelet/bm3d...)
│   ├── jpeg_artifacts/                            # JPEG artifact removal pipelines
│   └── motion_deblur/                             # Wiener/RL/TV-based deblurring, etc.
├── model/                                         # XYScanNet code: training/inference/model defs
│   ├── xyscannet/                                 # network modules (VSSM blocks, U-Net backbone)
│   ├── xvscannetP
│   ├── schedulers.py
│   ├── models.py
│   ├── networks.py
│   └── configs/                                   # hyperparams / dataset configs
├── data/
│   ├── public/                                    # public datasets 
│   └── self_collected/                            # our real-scene three distortions dataset
├── results/                                       # outputs (images, metrics csv, plots) (not committed)
├── scripts/                                       # helper scripts (batch run, eval, plotting)
│   ├── predict_GoPro_test_results.py              # testing entry
│   ├── train_XYScanNet_stage1.py                  # training entry1
│   └── train_XYScanNet_stage2.py                  # training entry2
├── utills/ 
│   ├── metric_counter.py                          # helper function for evaluation
│   ├── metrics.py                                 # metrics for evaluation
│   └── visualizer.py
├── requirements.txt                             
└── README.md
```

## 3. Project Setup
### 1) virtual environment setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) XVScanNet Reproduction -- based on GoPro dataset
(1) training
```
# Stage 1: We pre-train XYScanNetPlus for 4000 epochs with patch size 252x252
python train_XYScanNet_stage1.py --job_name xyscannetp_gopro # for stage 1
# Stage 2: After 4000 epochs, we keep training XYScanNetPlus for 4000 epochs with patch size 320x320
python train_XYScanNet_stage2.py --job_name xyscannetp_gopro # for stage 2
```
(2) testing
```
python predict_GoPro_test_results.py --job_name xyscannetp_gopro
```
(3) Results
The results for deblurring and pretrained model can be found here:
```
https://drive.google.com/drive/folders/1j1x4XM_AkVE2xuJVlcew_2_8ECxBHnAS?dmr=1&ec=wgc-drive-globalnav-goto
```

### 3) Classical DIP Methods
(1) Motion Blur
```
python -m DIP.deblur.motion_wiener \
  --input "data/self_collected/sample.jpg" \
  --len 5 --angle 10 \
  --balance 3e-3 \
  --wiener_mode ycrcb --chroma_scale 0.0 \
  --outdir results/Motion_wiener
```
For evaluation:
```
python -m DIP.blur.deblur_eval \
  --indir results/Motion_wiener \
  --pattern "*.png" \
  --noise_tol 0.20 --noise_penalty 1.0 \
  --block_weight 0.05 --aux_weight 0.30 \
  --w_niqe 0.49 --w_brisque 0.49 --w_piqe 0.02 \
  --out_csv eval_metrics.csv
```
(2) Gaussian Noise
```
python -m DIP.noise.denoise_BM3D \
  --input "data/self_collected/Gaussian_sample1.jpg" \
  --sigma 25 --seed 0 \
  --bm3d_mode ycbcr --chroma_scale 0.5 --unsharp \
  --outdir results/Gaussian_noise
```
for evaluation:
```
python -m DIP.noise.noise_eval \
  --indir results/Gaussian_noise \
  --pattern "*.png" \
  --out_csv eval_metrics.csv
```
(3) JPEG Artifacts
```
python -m DIP.JPEG.dip_combined \
  --input data/self_collected/JPEG_sample.jpg \
  --jpeg_q 20 \
  --strength medium \
  --outdir results/JPEG_combo
```
for evaluation:
```
python -m DIP.JPEG.dip_eval --indir results/JPEG_combo --pattern "*.png"
```

## 4. Additional Link for Large Data
https://drive.google.com/drive/folders/1j1x4XM_AkVE2xuJVlcew_2_8ECxBHnAS?usp=sharing
