# FloodNet++: Flood Segmentation using SAR and Deep Learning

## Overview
This project presents a robust flood segmentation pipeline using SAR and multispectral satellite data. The approach combines domain-specific preprocessing, feature engineering, and advanced inference techniques.

## Key Features
- SAR speckle noise reduction using bilateral filtering
- NDWI and MNDWI feature augmentation
- UNet (ResNet34 encoder) adapted for 8-channel input
- Hybrid loss (Cross Entropy + Dice + Boundary Loss)
- Snapshot ensembling
- Multi-scale + 8-way TTA inference
- Threshold tuning for optimal segmentation

## Dataset
AISEHack Phase 2 dataset

## Model
- Architecture: UNet (ResNet34)
- Input: 8-channel (6 bands + NDWI + MNDWI)

## Training
- Data augmentation: D4, grid distortion, noise
- Loss: Weighted CE + Dice + Boundary Loss

## Inference
- Multi-scale prediction (0.8, 1.0, 1.2)
- 8-way Test Time Augmentation
- Snapshot ensemble

## Results
Leaderboard Score: 0.1736

## How to Run
1. Install dependencies:
   pip install segmentation-models-pytorch albumentations

2. Run notebook:
   notebooks/best-model.ipynb

## Future Work
- Improve generalization
- Better post-processing
- Ensemble optimization
