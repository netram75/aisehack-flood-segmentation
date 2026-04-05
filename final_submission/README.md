# 🌊 Flood Detection & Localization — Phase 2
### ANRF AISEHack 2026 | Theme 1: Flood Prediction

<div align="center">

![Score](https://img.shields.io/badge/Score-0.1845-2ea44f?style=for-the-badge)
![Rank](https://img.shields.io/badge/Rank-%2316-028090?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-3--Class%20Segmentation-0A2342?style=for-the-badge)
![GPU](https://img.shields.io/badge/GPU-Kaggle%20T4-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-ANRF%20Open-blueviolet?style=for-the-badge)

**Team: Outliers** — Netram Faran · Ravijot Sinha

[📓 Kaggle Notebook](https://www.kaggle.com/competitions/anrfaisehack-theme-1-phase2) · [🤖 Model Checkpoint](https://www.kaggle.com/models/netram75/outliers-flood-detection-phase2) · [📄 Report](docs/final_submission_report.pdf)

</div>

---

## 📌 Overview

This repository contains our solution to **Phase 2** of the ANRF AISEHack flood detection challenge — 3-class pixel-wise segmentation of multi-sensor satellite imagery over West Bengal, India (May 29, 2024 flood event).

| Class | Label | Description |
|---|---|---|
| 🟫 No Flood | `0` | Dry land — no inundation |
| 🔵 Flood | `1` | Newly inundated areas (event-specific) |
| 💧 Water Body | `2` | Permanent or seasonal water bodies |

> **The core challenge of Phase 2:** Flood water and permanent water bodies both appear as low-backscatter regions in SAR imagery. Separating them requires domain-specific preprocessing and spectral water indices.

---

## 🏆 Results

| Submission | Score | Key Feature |
|---|---|---|
| Original baseline | 0.170 | Starting point |
| Back-to-basics + bug fixes | 0.1706 | Clean reimplementation |
| MIT-B2 Transformer | 0.1735 | Architecture experiment |
| 2-seed ensemble | 0.1799 | Probability averaging |
| **32-patch + BoundaryLoss ⭐ FINAL** | **0.1845** | **Selected submission** |
| Top team (reference) | 0.2561 | 5+ model ensemble |

---

## 🧠 Solution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT  (8 channels)                       │
│  SAR HH* │ SAR HV* │ Green │ Red │ NIR │ SWIR │ NDWI │ MNDWI│
│          * bilateral speckle filtered before normalization   │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  ResNet34       │  ← ImageNet pretrained
              │  Encoder        │     ~21.3M parameters
              └────────┬────────┘
                       │  skip connections
              ┌────────▼────────┐
              │  UNet Decoder   │  ← symmetric upsampling
              │                 │     ~2.5M parameters
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ 3-Class Softmax │  No Flood | Flood | Water Body
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Multi-Scale TTA │  3 scales × 8-way = 24 preds
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Post-processing │  Morphological close + CCA
              │  (min 50 px)    │  ≥ 16,200 m² at 18m resolution
              └─────────────────┘
```

---

## 🔧 Technical Details

### Data Preprocessing

```python
# 1. SAR Speckle Reduction — applied BEFORE normalization
hh = cv2.bilateralFilter(hh.astype(np.float32), d=5, sigmaColor=50, sigmaSpace=50)
hv = cv2.bilateralFilter(hv.astype(np.float32), d=5, sigmaColor=50, sigmaSpace=50)

# 2. Z-score normalization using dataset statistics
MEANS = [841.13, 371.62, 1734.18, 1588.31, 1742.85, 1218.56]
STDS  = [473.71, 170.36,  623.05,  612.85,  564.58,  528.09]

# 3. Spectral water indices added as channels 7 & 8
NDWI  = clip((Green - NIR)  / (|Green + NIR|  + 1e-6), -3, 3)
MNDWI = clip((Green - SWIR) / (|Green + SWIR| + 1e-6), -3, 3)
```

### Patch-Based Training — The Key Innovation

With only **59 training images**, standard full-image training overfits severely.
Our solution: extract **32 random 256×256 patches per image per epoch**.

```
59 images  ×  32 patches  =  1,888 training samples / epoch
                           =  32× data multiplication
                              without any synthetic data
```

> Increasing from 16 → 32 patches/image was the single largest score improvement: **0.1799 → 0.1845**

### Loss Function

```
L = 0.4 × L_CrossEntropy(weighted) + 0.4 × L_Dice + 0.1 × L_Boundary

  L_CE       — class weights = inverse frequency, flood class × 2× boost
  L_Dice     — region overlap loss, robust to class imbalance
  L_Boundary — Sobel edge loss on flood class predictions
               penalizes errors specifically at flood boundaries
```

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Encoder | ResNet34 (ImageNet pretrained) |
| Decoder | UNet with skip connections |
| Total parameters | ~23.8M |
| Input channels | 8 |
| Patch size | 256 × 256 |
| Patches per image | 32 |
| Epochs | 50 |
| Optimizer | Adam (lr = 1e-4) |
| LR Scheduler | CosineAnnealingLR (T_max=50, eta_min=1e-6) |
| Batch size | 8 |
| GPU | Kaggle T4 (15 GB VRAM) |
| Training time | ~90 minutes |
| Inference time | ~12 seconds/image (with TTA) |

### Augmentation Pipeline

```python
A.D4(p=1.0)                                               # 8-way flips + rotations
A.RandomBrightnessContrast(0.15, 0.15, p=0.5)            # ± 15%
A.GaussNoise(var_limit=(10.0, 40.0), p=0.35)
A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.25)
A.CoarseDropout(max_holes=6, max_height=28, max_width=28, p=0.3)
```

### Inference Pipeline

```python
# Multi-scale TTA
scales = [0.8, 1.0, 1.2]

# 8-way geometric TTA at each scale
augmentations = [
    identity, h_flip, v_flip,
    rot90, rot180, rot270,
    rot90 + h_flip, rot270 + h_flip
]
# Total: 3 scales × 8 augmentations = 24 averaged predictions

# Threshold: tuned on val set by sweeping [0.30, 0.71] in steps of 0.01

# Post-processing:
# 1. Morphological closing (5×5 ellipse kernel)
# 2. Connected component filtering (remove regions < 50 pixels)
```

---

## 📊 Key Scientific Findings

**1. Patch density is the primary lever on small satellite datasets**

Testing 16 vs 32 patches/image is a clean single-variable ablation — same model, same code. The +0.0046 score improvement confirms that patch sampling functions as free data augmentation on datasets below 100 images.

**2. Auxiliary data can fail due to geographic saturation**

The Ganges-Brahmaputra delta has **64% of pixels historically wet** per JRC Global Surface Water Explorer — making this standard auxiliary dataset useless for flood vs. water-body discrimination in this region. A finding relevant to all ML-based flood monitoring systems in South Asian delta geographies.

**3. Prediction distribution analysis is more diagnostic than training metrics**

Our baseline predicted flood at **26%** of pixels vs. the true **8.7%**. This simple distributional check revealed why training loss looked good but leaderboard scores were poor — the model confused flood with water body. The diagnostic directly drove threshold calibration and class weight strategy.

---

## ❌ What Did NOT Work

<details>
<summary>Click to expand full ablation table</summary>

| Approach | Score | Root Cause |
|---|---|---|
| ResNet50 + DeepLabV3+ | 0.1673 | Too complex for 59 samples — overfits |
| MIT-B2 Transformer encoder | 0.1735 | Transformers need substantially more data |
| 10-channel input (SAR HH/HV ratio) | 0.1641 | Numerically unstable after bilateral filter + normalization |
| Train on all 79 samples (no val split) | 0.1596 | Threshold overfits to training data |
| Max confidence pooling ensemble | 0.1587 | Picks most confident model, not most accurate |
| GSW Global Surface Water (9ch input) | 0.1121 | 64% of delta pixels historically wet — no discrimination |
| Push-to-water threshold trick | 0.1495 | Overcorrects already inflated flood predictions |
| Argmax inference vs. binary threshold | worse | RLE encodes flood only — binary threshold is correct approach |
| 3× flood weight boost (unnormalized) | worse | Model predicts flood everywhere, precision collapses |
| ElasticTransform + CoarseDropout heavy | worse | Too destructive on only 59 training images |
| EfficientNet-B4 + UNet++ | 0.1644 | Larger encoder + complex decoder compounds overfitting |
| HRNet-W22 + 4-fold CV | unstable | Window size incompatibility with 256×256 patches |

</details>

---

## 🚀 How to Reproduce

### Requirements

```bash
pip install segmentation-models-pytorch albumentations rasterio opencv-python torch numpy pandas
```

### Run

```bash
# 1. Clone repository
git clone https://github.com/netram75/outliers-flood-detection.git
cd outliers-flood-detection

# 2. Open notebook.ipynb on Kaggle
#    Add competition dataset:
#    kaggle.com/competitions/anrfaisehack-theme-1-phase2

# 3. Run all cells end to end
#    Seeds fixed at 42 — fully reproducible

# 4. Output: submission_phase2.csv (RLE-encoded flood pixels)
```

> **Self-contained:** Trains from scratch in ~90 min on T4 GPU. No pre-downloaded weights required — ResNet34 ImageNet weights are fetched automatically by segmentation-models-pytorch.

---

## 🤖 AI Tools Disclosure

We used two GenAI tools as **assistants** during development. All decisions — architectural, experimental, and submission — were made and validated by the team.

| Tool | Role | What We Did |
|---|---|---|
| **Claude Sonnet 4.6** (claude.ai) | Coding assistant | We directed it to generate boilerplate code, find bugs, and suggest options. Team evaluated and selected what to implement |
| **Grok** (xAI) | Hyperparameter reviewer | We fed our existing code to Grok and asked for a config review. Team tested suggestions on Kaggle — adopted 16→32 patches (+0.0046 gain), rejected others |

> Every AI suggestion was validated experimentally on Kaggle before adoption. Several suggestions were discarded after testing showed no improvement or a score regression.

---

## 📚 References

1. Ronneberger et al. (2015). *U-Net: CNNs for Biomedical Image Segmentation.* MICCAI.
2. He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
3. Pekel et al. (2016). *High-resolution global maps of 21st century surface water.* Nature.
4. Buslaev et al. (2020). *Albumentations: Fast and Flexible Image Augmentations.* Information.
5. Yakubovskiy (2019). *Segmentation Models PyTorch.* GitHub.

---

## 📄 License

Released under the **ANRF Open License** — see [LICENSE](LICENSE) for full terms.

---

<div align="center">
<sub>ANRF AISEHack 2026 Edition 1 · Phase 2 Finale · IIIT Hyderabad · April 5, 2026</sub>
</div>
