<div align="center">

<!-- BANNER IMAGE: Replace the src below with your project banner (e.g., a sample detection result) -->
<!-- Recommended size: 1200×400px -->
<img src="assets\image.png" alt="BDD100K Object Detection Banner" width="100%"/>

<br/>
<br/>

# Traffic Monitoring using FasterR-CNN
### Fine-tuned Faster R-CNN on Berkeley DeepDrive Dataset

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![TorchVision](https://img.shields.io/badge/TorchVision-0.17%2B-orange?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=flat-square&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## Project Overview

This project fine-tunes a **Faster R-CNN** (ResNet-50 + FPN backbone) model from TorchVision on a curated subset of the [BDD100K](<!-- ADD DATASET LINK HERE: e.g., https://bdd-data.berkeley.edu/ -->) autonomous driving dataset. The model is trained to detect **10 traffic-relevant object categories** in complex road scenes.

Training is performed exclusively on **daytime images**, and the model is then evaluated against both a standard daytime validation set and a challenging **nighttime validation set** — enabling a direct analysis of the domain gap between day and night driving conditions.

---

## Features

- **Transfer learning** from a COCO-pretrained Faster R-CNN backbone
- Custom **BDD100K dataset loader** with flexible annotation parsing
- **Mixed-precision training** via `torch.amp` for faster GPU training
- Automatic **best model checkpointing** based on epoch loss
- **Dual evaluation** on daytime and nighttime validation splits
- **mAP evaluation** using `torchmetrics` with per-class metrics
- Epoch-wise **loss curve plotting** saved automatically

---

## Dataset Structure & Source

**Source:** Berkeley DeepDrive BDD100K — [BDD100K DATASET](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)

The dataset used in this project is a **curated subset** of BDD100K:

```
mini_dataset/
├── images/
│   ├── train/          # 2,000 daytime training images
│   ├── val/            # 500 daytime validation images
│   └── night_val/      # 1,000 nighttime validation images
└── labels/
    ├── train.json      # BDD100K-format annotations for train set
    ├── val.json        # BDD100K-format annotations for val set
    └── night_val.json  # BDD100K-format annotations for night val set
```

**Detected Classes (10 + background):**

| ID | Class          
|----|----------------
| 1  | person         
| 2  | rider          
| 3  | car            
| 4  | truck          
| 5  | bus            
| 6  | train         
| 7  | motor         
| 8  | bike          
| 9  | traffic light 
| 10 | traffic sign  
---

## Model Architecture

The model is based on **Faster R-CNN with a ResNet-50 + FPN (Feature Pyramid Network)** backbone, initialized with COCO pretrained weights.

The classification head is replaced to predict the 10 BDD100K classes (+ background):

```
torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
  └── roi_heads.box_predictor → FastRCNNPredictor(in_features, num_classes=11)
```

<!-- ARCHITECTURE DIAGRAM -->
<!-- Add an architecture diagram here. Recommended: a clear flowchart showing -->
<!-- Image → Backbone (ResNet-50) → FPN → RPN → RoI Pooling → Predictor Head → Detections -->
<div align="center">
  <img src="assets\Faster-RCNN-architecture.ppm" alt="Model Architecture" width="80%"/>
  <p><em>Figure 1: Faster R-CNN with ResNet-50 + FPN Architecture</em></p>
</div>

**Training configuration:**

| Hyperparameter    | Value           |
|-------------------|-----------------|
| Optimizer         | SGD             |
| Learning Rate     | 0.01            |
| Momentum          | 0.9             |
| Weight Decay      | 0.0005          |
| Batch Size        | 4               |
| Epochs            | 30              |
| Input Resolution  | 384 × 640       |
| Mixed Precision   | AMP             |

---

## Installation

**Prerequisites:** Python 3.10+, CUDA-enabled GPU

```bash
# 1. Clone the repository
git clone https://github.com/gayathri-edathodathil/Traffic-Monitoring-using-FasterR-CNN
cd Traffic-Monitoring-using-FasterR-CNN

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics matplotlib pillow pandas

# 4. download dataset, create directories: best_model, results
```

> **Note:** Adjust the PyTorch install URL to match your CUDA version. See [pytorch.org](https://pytorch.org/get-started/locally/) for options.

---

## Usage

### Training

```bash
python fasterrcnn.py
```

By default, this trains on `mini_dataset`. The script will:
- Load the training set and initialize the model
- Train for 30 epochs with AMP and SGD
- Save the **best model checkpoint** to `best_model/best_model_mini_dataset.pth`
- Save the **loss curve** to `results/losses_mini_dataset.png`

To use a different dataset, change the `dataset` argument in `fasterrcnn.py`:

```python
if __name__ == '__main__':
    main(dataset="your_dataset_folder")
```

---

### Evaluation

```bash
python testing_module.py
```

This loads the best saved checkpoint and evaluates on all three splits:

- **Train set** — measures training fit
- **Val set** — standard daytime performance
- **Night Val set** — measures performance under domain shift

Results are printed to the console as mAP metrics (mAP@[0.5:0.95], mAP@0.5, per-class AP).

---

### Resuming Training

To resume from a checkpoint, uncomment this line in `fasterrcnn.py`:

```python
# model.load_state_dict(torch.load(f"best_model/best_model_{dataset}.pth"))
```

---

## Results

### Loss Curve

<!-- LOSS CURVE IMAGE -->
<!-- Replace with your actual loss plot saved at results/losses_mini_dataset.png -->
<div align="center">
  <img src="assets\losses_bdd100k.png" alt="Loss vs Epoch" width="65%"/>
  <p><em>Figure 2: Training Loss vs Epoch</em></p>
</div>

---

### Detection Examples

<!-- DETECTION RESULTS IMAGES -->
<!-- Add side-by-side comparisons of ground truth vs. predictions, or a grid of detections -->
<div align="center">
  <img src="assets\ca2c7c84-5902ee09.jpg.png" alt="Daytime Detections" width="48%"/>
  &nbsp;
  <img src="assets\5f4b8bdd-686f2306.jpg.png" alt="Nighttime Detections" width="48%"/>
  <p><em>Figure 3: Sample detections — Daytime (left) vs. Nighttime (right)</em></p>
</div>

---

### mAP Evaluation Table

<!-- Fill in your actual results below -->

| Metric             | Train Set | Val Set (Day) | Val Set (Night) |
|--------------------|-----------|---------------|-----------------|
| mAP @[0.5:0.95]   | 0.5681         | 0.2015             | 0.1041               |
| mAP @0.50          | 0.8404         | 0.4069             | 0.2286               |
| mAP @0.75          | 0.6576         | 0.1748             | 0.0817               |


> **Observation:** The drop in mAP between the daytime and nighttime validation sets reflects the domain gap introduced by illumination changes. The model was trained exclusively on daytime images.

---

## Project Structure

```
.
├── datasetLoader.py        # Custom PyTorch Dataset for BDD100K annotations
├── transforms.py           # Custom Resize, ToTensor, and Compose transforms
├── fasterrcnn.py           # Training pipeline with AMP and checkpointing
├── valuation.py            # Evaluator class using torchmetrics mAP
├── testing_module.py       # Evaluation script for all three data splits
│
├── mini_dataset/           # Dataset directory (not included in repo)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── night_val/
│   └── labels/
│       ├── train.json
│       ├── val.json
│       └── night_val.json
│
├── best_model/             # Saved model checkpoints
│   └── best_model_mini_dataset.pth
│
├── results/                # Output plots
│   └── losses_mini_dataset.png
│
└── assets/                 # README images (banner, architecture, samples)
    ├── banner.png
    ├── architecture.png
    ├── detections_day.png
    └── detections_night.png
```

---

## Technologies Used

| Library / Tool       | Purpose                                      |
|----------------------|----------------------------------------------|
| **PyTorch**          | Deep learning framework, training loop       |
| **TorchVision**      | Faster R-CNN model, pretrained weights       |
| **torchmetrics**     | mAP computation with per-class metrics       |
| **Pillow (PIL)**     | Image loading and preprocessing              |
| **Matplotlib**       | Loss curve visualization                     |
| **CUDA / AMP**       | GPU acceleration and mixed-precision training|
| **BDD100K**          | Autonomous driving dataset                   |

---


## Author

**Gayathri E**

- GitHub: [@gayathri-edathodathil](https://github.com/gayathri-edathodathil)
- LinkedIn: [gayathri-edathodathil](https://linkedin.com/in/gayathri-edathodathil)

---