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

## Model Files

### PyTorch (training / research)

`weights/best.pt` — original Ultralytics checkpoint (5.4 MB).

### ONNX (deployment / website demo)

`weights/best.onnx` — exported ONNX version of the same trained model (9.3 MB).

Exported with:

- Ultralytics 8.4.166, ONNX opset 17
- Precision: FP32
- Input: static `1 × 3 × 640 × 640` (no dynamic axes)
- Simplification: off (onnxslim not required)
- No retraining was performed; the ONNX graph is a direct conversion of `best.pt`

#### ONNX input / output contract

| Item | Value |
|------|-------|
| Input name | `images` |
| Input shape | `(1, 3, 640, 640)` float32 |
| Input layout | NCHW, RGB, values scaled to `[0, 1]` |
| Input content | image letterboxed to 640×640 (aspect ratio preserved, padding value 114) |
| Output name | `output0` |
| Output shape | `(1, 5, 8400)` float32 |
| Output rows | 8400 anchors; after transpose to `(8400, 5)` each row is `[cx, cy, w, h, score]` |
| Coordinates | pixel space of the 640×640 letterboxed input |
| Class | 0 = ear (single class) |

Post-processing required on the raw output:

1. Transpose `(5, 8400)` → `(8400, 5)`.
2. Filter rows where `score > threshold` (e.g. 0.25).
3. Convert `[cx, cy, w, h]` → `[x1, y1, x2, y2]` if needed.
4. Apply NMS (IoU threshold e.g. 0.7).
5. Map boxes back to original image coordinates:
   `x_orig = (x_640 − pad_x) / scale`, `y_orig = (y_640 − pad_y) / scale`,
   where `scale = min(640 / orig_w, 640 / orig_h)` and padding is centered.

#### Minimal Python usage

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("weights/best.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name  # "images"

# img: HWC uint8 or float RGB image, already letterboxed to 640x640
x = img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0  # NCHW
output = sess.run(None, {input_name: x})[0]  # (1, 5, 8400)
pred = output[0].T  # (8400, 5) rows = [cx, cy, w, h, score]
```

#### Verification

The exported ONNX model was verified against `best.pt` (no retraining, checkpoint unchanged):

- Raw tensor parity: max absolute difference 2.14e-04 on identical inputs (within FP32 cross-backend tolerance).
- Detection parity on real TIRPigEar validation images: identical detection counts, mean IoU ≈ 0.97, maximum confidence delta 0.17 (cross-backend NMS drift on near-threshold boxes).
- Decode-format verification on 15 validation images: the raw output rows are `[cx, cy, w, h, score]` in 640×640 letterboxed pixel coordinates. The full pipeline (decode → threshold → NMS → un-letterbox to original coordinates) recovered 100% of PyTorch detections with mean IoU 0.971, and matched ultralytics' own ONNX backend counts on 15/15 images.
- Verification environment: ultralytics 8.4.166, onnxruntime 1.30.0 (CPU), Python 3.13.

#### Browser / web demo note

The ONNX file loads directly with ONNX Runtime Web (onnxruntime-web) for in-browser inference. Opset 17 and the static FP32 graph are widely supported. For lower latency, quantize to INT8 in a separate export if needed — do not replace `weights/best.onnx` until accuracy is re-verified.

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

Do not upload the original TIRPigEar dataset to this repository.

## Repository Structure

```
Pig_Ear_detection_model/
├── README.md
└── weights/
    ├── best.pt             # PyTorch checkpoint
    └── best.onnx           # ONNX deployment version
```
