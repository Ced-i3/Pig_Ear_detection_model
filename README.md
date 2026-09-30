# Pig Ear Detection Model

YOLO26n thermal pig-ear detection model for the O.I.N.K. non-contact swine temperature screening prototype.

## Model

- Architecture: YOLO26n
- Task: Thermal pig-ear detection
- Dataset: TIRPigEar YOLO dataset
- Classes: ear
- Training epochs: 15
- Image size: 640
- Batch size: 8
- Device: NVIDIA RTX 4050 Laptop GPU
- DataLoader workers: 0

## Validation Results

- Precision: 0.892
- Recall: 0.921
- mAP50: 0.955
- mAP50-95: 0.630

## Intended O.I.N.K. Pipeline

MLX90640 thermal frame
→ thermal image
→ YOLO ear detection
→ ear ROI
→ extract MLX90640 temperatures
→ temperature statistics
→ configured screening threshold
→ NORMAL / POSSIBLE FEVER

## Important

This model detects the pig-ear region.

It does NOT diagnose African swine fever (ASF).

It does NOT determine fever by itself.

The temperature measurement and screening decision are separate stages of the O.I.N.K. pipeline.

## Current Limitation

The model was trained using the TIRPigEar thermal dataset. It still needs validation using actual MLX90640 32×24 thermal frames before being considered reliable for the O.I.N.K. hardware.

## Model File

The trained model is located at:

weights/best.pt

Do not upload the original TIRPigEar dataset to this repository.
