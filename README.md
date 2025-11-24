# ECE 253 – Digital Image Processing Course Project  
**Title:** Classical DIP vs Vision State-Space Models for Single-Image Restoration  

## 1. Project Overview

This repository contains the course project for **ECE 253 (Digital Image Processing), Fall 2025**.  
We investigate how well **modern Vision State-Space Models (VSSMs)**—specifically **XYScanNet**—perform on real-world degraded images compared with **classical digital image processing (DIP)** pipelines.

Our focus:

- Target task: **single-image motion deblurring** (with a few other realistic distortions).
- Baseline: classical DIP methods (e.g., Wiener / Richardson–Lucy deconvolution, sharpening, CLAHE).
- ML model: XYScanNet, a U-Net–style restoration backbone built from Vision State-Space Modules.
- Data: combination of **public clean datasets** and our **self-collected real-scene blurry dataset**.

The goal is to understand when a fixed trained model is enough and when domain gap forces us to adapt either the **input/output side (DIP)** or the **model itself (fine-tuning)**.

---

## 2. Repository Structure

```text
ECE253/
├── DIP/                        # Classical DIP baselines / scripts
├── model/                      # XYScanNet model code, training + inference
├── data/
│   ├── self_collected/         # Our real-scene blurry dataset (or symlinks)
│   └── ...                     # Additional datasets / metadata
├── README.md                   # This file
└── (more to be added...)
