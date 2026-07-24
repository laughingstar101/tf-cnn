# EMNIST Digit Recognition

A complete TensorFlow pipeline for handwritten digit recognition using EMNIST Digits. Train a CNN, evaluate per‑digit performance, and run end‑to‑end prediction on images containing multiple digits. Enables native GPU support on Windows using Tensorflow v2.10.0 (must install necessary compatible CUDA Toolkit v11.2 and cuDNN v8.1.0 libraries). Only supports Python v3.10.

## Features
- Convert EMNIST IDX files to CSV
- 3‑layer convolutional neural network with dropout
- Training with early stopping and TensorBoard
- Per‑digit accuracy breakdown on test set
- Full prediction pipeline: contour detection → ROI extraction → preprocessing → classification
- Robust prediction pipeline:
  - Colour‑based foreground segmentation (LAB distance from background)
  - Adaptive percentile threshold to handle faint digits of any colour
  - Fallback to grayscale for images where colour is not discriminative
  - Dynamic area filtering to adapt to different image sizes
  - Morphological cleaning with larger kernels to close gaps
- Debug mode to visualise each processing step

## Project Structure
├── convert_emnist.py -> IDX → CSV converter  
├── train.py -> Training with early stopping  
├── evaluate.py -> Test evaluation with per‑digit stats  
├── predict.py -> End‑to‑end prediction from image  
├── preprocess_digit.py -> Otsu threshold, centering, resize to 28×28 (applied per ROI)  
├── opencv.py -> Robust contour detection using colour distance (LAB) and fallback  
├── model.py -> CNN architecture  
├── mnist.py -> Data loading helpers  
├── data/ -> CSV files (train/test)  
├── checkpoints/ -> Saved model checkpoints  
├── graphs/ -> TensorBoard logs  
├── debug/ -> Debug images (optional)  
└── roi/ -> Extracted digit regions  

## Installation
```
pip install .
```

## Dataset Preparation
1. Download EMNIST Digits IDX files from the official site: [biometrics.nist.gov/cs_links/EMNIST/gzip.zip](https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip)
2. Place them in data/ with names like emnist-digits-train-images-idx3-ubyte.gz.
3. Convert to CSV:
```python convert_emnist.py ```

## Training
```
python train.py
```

## Key arguments:
- --batch_size 256
- --num_iter 50000
- --checkpoint_file_path checkpoints/model.ckpt
- --train_data data/emnist_digits_train.csv
- --summary_dir graphs
- --patience 20 (early stopping)
- --min_delta 0.0001

## Monitor with TensorBoard:
```
tensorboard --logdir graphs/
```

## Evaluation
```
python evaluate.py
```

## Example Output
```
==================================================
Per-Digit Classification Breakdown
==================================================
Digit    | Correct   | Wrong     | Accuracy
--------------------------------------------------
0        | 1823      | 11        | 99.40%
1        | 1853      | 7         | 99.62%
...
Overall Accuracy: 0.9909
```

## Prediction Pipeline
Process an image with multiple handwritten digits:
```
python predict.py path/to/image.png
```
Output:
```
Original:    phone_number.png
Predicted:   1234567890
```
Enable debug visualisation (saves intermediate images to debug/):
```
python predict.py path/to/image.png --debug
```
## How It Works
- Contour detection – finds and extracts each digit region.
- Preprocessing – applies Otsu thresholding, crops to bounding box, pads to square, resizes to 20×20, and centres on a 28×28 canvas.
- Classification – feeds each processed digit through the trained CNN.

## Model Architecture
|Layer|	Details|
|-|-|
|Conv1|	5×5, 32 filters, ReLU, 2×2 pool|
|Conv2| 5×5, 64 filters, ReLU, 2×2 pool|
|Conv3|	3×3, 128 filters, ReLU|
|FC1|	256 units, ReLU, dropout (0.8)|
|FC2 (output)|	10 units (logits)|

## Results
Test accuracy: 99.41% on EMNIST Digits.
|Digit    | Correct    | Wrong      | Accuracy|
|---------|------------|------------|---------|
|0        | 3984       | 16         | 99.60%|
|1        | 3987       | 13         | 99.67%|
|2        | 3976       | 24         | 99.40%|
|3        | 3979       | 21         | 99.48%|
|4        | 3984       | 16         | 99.60%|
|5        | 3979       | 21         | 99.48%|
|6        | 3975       | 25         | 99.38%|
|7        | 3980       | 20         | 99.50%|
|8        | 3955       | 45         | 98.88%|
|9        | 3965       | 35         | 99.12%|

## License
MIT License

Copyright (c) 2026 laughingstar101

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
