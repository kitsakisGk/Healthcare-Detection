# CLI Scripts - Production Tools

This directory contains production-ready command-line tools for the Healthcare Pneumonia Detection system.

## 📋 Available Scripts

### 1. `download_datasets.py` - Dataset Downloader
Automatically download and organize multi-class pneumonia datasets.

**Features:**
- Downloads original Kaggle pneumonia dataset (NORMAL + BACTERIAL)
- Downloads COVID-19 radiography database
- Organizes into 4-class structure: NORMAL, BACTERIAL, VIRAL, COVID19
- Automatic train/val/test splitting
- Dataset summary generation

**Prerequisites:**
```bash
pip install kaggle
# Setup Kaggle API: https://www.kaggle.com/docs/api
```

**Usage:**
```bash
# Download all datasets
python scripts/download_datasets.py --output data

# Download only original pneumonia dataset
python scripts/download_datasets.py --output data --original-only

# Download only COVID-19 dataset
python scripts/download_datasets.py --output data --covid-only

# Skip COVID-19 download
python scripts/download_datasets.py --output data --skip-covid
```

**Output Structure:**
```
data/
├── train/
│   ├── NORMAL/
│   ├── BACTERIAL/
│   ├── VIRAL/
│   └── COVID19/
├── val/
│   └── (same structure)
├── test/
│   └── (same structure)
└── dataset_summary.txt
```

---

### 2. `train.py` - Model Training
Train individual models or ensemble on multi-class data.

**Features:**
- Train ResNet152, DenseNet201, EfficientNet-B3, or Ensemble
- Automatic mixed precision training
- Early stopping with patience
- Model checkpointing (best + periodic)
- Learning rate scheduling
- Class weighting for imbalanced data
- Training history logging (JSON + TensorBoard)

**Usage:**
```bash
# Train EfficientNet-B3
python scripts/train.py --model efficientnet --epochs 30

# Train ensemble model
python scripts/train.py --model ensemble --epochs 20 --batch-size 16

# Train with custom learning rate
python scripts/train.py --model resnet152 --lr 0.0005 --epochs 25

# Resume from checkpoint
python scripts/train.py --model efficientnet --resume models/checkpoint_epoch_10.pth

# Train on GPU
python scripts/train.py --model ensemble --device cuda
```

**Arguments:**
- `--model`: Model architecture (resnet152, densenet201, efficientnet, ensemble)
- `--config`: Path to config.yaml (default: config/config.yaml)
- `--data-dir`: Data directory (default: data)
- `--epochs`: Number of epochs (overrides config)
- `--batch-size`: Batch size (overrides config)
- `--lr`: Learning rate (overrides config)
- `--save-dir`: Checkpoint save directory (default: models)
- `--resume`: Resume from checkpoint
- `--device`: Device (cpu, cuda, auto)
- `--num-workers`: DataLoader workers (default: 4)

**Outputs:**
- `models/best_model.pth` - Best model checkpoint
- `models/checkpoint_epoch_N.pth` - Periodic checkpoints
- `models/training_history.json` - Training metrics

---

### 3. `evaluate.py` - Model Evaluation
Comprehensive evaluation on test set with metrics and visualizations.

**Features:**
- Calculate accuracy, precision, recall, F1, AUC-ROC
- Per-class and overall metrics
- Confusion matrix visualization
- Metrics summary plots
- Classification report
- JSON export of all metrics

**Usage:**
```bash
# Evaluate EfficientNet model
python scripts/evaluate.py \
    --model-type efficientnet \
    --model-path models/best_model.pth \
    --data-dir data/test

# Evaluate ensemble model
python scripts/evaluate.py \
    --model-type ensemble \
    --model-path models/ensemble_best.pth \
    --data-dir data/test \
    --save-dir reports

# Custom batch size
python scripts/evaluate.py \
    --model-type resnet152 \
    --model-path models/resnet152_best.pth \
    --data-dir data/test \
    --batch-size 64
```

**Arguments:**
- `--model-path`: Path to trained checkpoint (required)
- `--model-type`: Model architecture (required)
- `--data-dir`: Test data directory (default: data/test)
- `--config`: Path to config.yaml
- `--batch-size`: Batch size (default: 32)
- `--save-dir`: Save directory for results (default: reports)
- `--device`: Device (cpu, cuda, auto)

**Outputs:**
- `reports/{model}_evaluation.json` - All metrics in JSON
- `reports/{model}_confusion_matrix.png` - Confusion matrix heatmap
- `reports/{model}_metrics_summary.png` - Metrics bar charts
- Console output with classification report

---

### 4. `inference.py` - Prediction & Inference
Run predictions on single images or batches with full explainability.

**Features:**
- Single image or batch prediction
- Uncertainty quantification (Monte Carlo dropout)
- Grad-CAM explainability
- Quality control checks
- DICOM file support
- PDF report generation
- Risk stratification (low/medium/high)

**Usage:**
```bash
# Simple prediction
python scripts/inference.py \
    --model-path models/best_model.pth \
    --model-type efficientnet \
    --image test.jpg

# Prediction with Grad-CAM
python scripts/inference.py \
    --model-path models/best_model.pth \
    --model-type ensemble \
    --image patient_xray.jpg \
    --gradcam \
    --save-report results/

# With uncertainty quantification
python scripts/inference.py \
    --model-path models/best_model.pth \
    --model-type efficientnet \
    --image scan.jpg \
    --uncertainty \
    --gradcam

# Quality control + PDF report
python scripts/inference.py \
    --model-path models/best_model.pth \
    --model-type ensemble \
    --image patient.jpg \
    --quality-check \
    --gradcam \
    --save-report reports/

# DICOM file support
python scripts/inference.py \
    --model-path models/best_model.pth \
    --model-type ensemble \
    --image scan.dcm \
    --dicom \
    --gradcam \
    --save-report reports/

# Batch processing
python scripts/inference.py \
    --model-path models/best_model.pth \
    --model-type efficientnet \
    --directory data/test/NORMAL/ \
    --batch \
    --output batch_results.json
```

**Arguments:**
- `--model-path`: Path to trained checkpoint (required)
- `--model-type`: Model architecture (required)
- `--image`: Single image path
- `--directory`: Directory for batch processing
- `--config`: Path to config.yaml
- `--device`: Device (cpu, cuda, auto)
- `--batch`: Enable batch processing mode
- `--gradcam`: Generate Grad-CAM visualizations
- `--uncertainty`: Enable uncertainty quantification
- `--quality-check`: Perform quality control checks
- `--save-report`: Save PDF report to directory
- `--dicom`: Input is DICOM file
- `--output`: Output JSON file (default: predictions.json)

**Outputs:**
- `predictions.json` - Prediction results with all metadata
- `reports/report_{filename}.pdf` - Professional PDF report (if --save-report)
- Console output with detailed prediction breakdown

---

## 🚀 Quick Start Workflow

### Complete Pipeline Example:

```bash
# 1. Download datasets
python scripts/download_datasets.py --output data

# 2. Train ensemble model
python scripts/train.py \
    --model ensemble \
    --epochs 30 \
    --batch-size 32

# 3. Evaluate on test set
python scripts/evaluate.py \
    --model-type ensemble \
    --model-path models/best_model.pth \
    --data-dir data/test

# 4. Run inference with explainability
python scripts/inference.py \
    --model-path models/best_model.pth \
    --model-type ensemble \
    --image new_patient.jpg \
    --gradcam \
    --uncertainty \
    --quality-check \
    --save-report clinical_reports/
```

---

## 📊 Expected Results

**Target Metrics (Multi-Class):**
- Overall Accuracy: >94%
- Weighted F1-Score: >93%
- Per-Class Recall: >90% (critical for medical AI)
- AUC-ROC: >0.95

**Ensemble Model Benefits:**
- +2-3% accuracy over single models
- Better uncertainty estimates
- More robust predictions
- Reduced false negatives

---

## 🔧 Configuration

All scripts use `config/config.yaml` for default settings. You can:
- Edit config file directly
- Override with CLI arguments
- Create custom config files

See [config/config.yaml](../config/config.yaml) for all options.

---

## 📝 Notes

- All scripts support `--help` for detailed usage
- Logs are saved to `logs/` directory
- Models are saved to `models/` directory
- Reports are saved to `reports/` directory
- GPU training is automatic if CUDA is available
- Mixed precision training enabled by default (faster!)

---

## 🐛 Troubleshooting

**Out of Memory?**
```bash
# Reduce batch size
python scripts/train.py --model ensemble --batch-size 8
```

**Slow Training?**
```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# Use fewer workers
python scripts/train.py --model efficientnet --num-workers 2
```

**Kaggle Download Issues?**
```bash
# Install Kaggle CLI
pip install kaggle

# Setup API token
# 1. Go to https://www.kaggle.com/account
# 2. Create new API token
# 3. Save to ~/.kaggle/kaggle.json
```

---

**For more details, see the main [README.md](../README.md)**
