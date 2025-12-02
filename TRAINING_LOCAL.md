# Local Training Guide

Complete guide for training models on your own computer.

## Prerequisites

### Hardware Requirements

**Minimum (CPU Training)**:
- 8GB RAM
- 20GB free disk space
- **Training time**: ~10-15 hours per model

**Recommended (GPU Training)**:
- NVIDIA GPU with 6GB+ VRAM (GTX 1660, RTX 2060, or better)
- 16GB RAM
- 50GB free disk space
- **Training time**: ~2-3 hours per model

### Software Requirements

- Python 3.8 or higher
- CUDA 11.8+ (for GPU training)
- Git

## Step-by-Step Instructions

### 1. Clone Repository

```bash
git clone https://github.com/kitsakisGk/Healthcare-Detection.git
cd Healthcare-Detection
```

### 2. Set Up Environment

**Option A: Using venv (Recommended)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Option B: Using conda**
```bash
conda create -n healthcare python=3.10
conda activate healthcare
pip install -r requirements.txt
```

### 3. Download Dataset

**Automatic Download:**
```bash
python scripts/download_datasets.py
```

This will:
- Download 2 datasets from Kaggle (~2GB total)
- Organize into 4 classes: NORMAL, BACTERIAL, VIRAL, COVID-19
- Split into train/val/test sets

**Manual Download:**
1. Get Kaggle API token from https://www.kaggle.com/settings
2. Place `kaggle.json` in `~/.kaggle/`
3. Run the download script

### 4. Train Models

#### Train Individual Models

**EfficientNet-B3** (Best single model - 98.18% F1):
```bash
python scripts/train.py \
    --model efficientnet \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.0001 \
    --device cuda
```

**For CPU (slower but works)**:
```bash
python scripts/train.py \
    --model efficientnet \
    --epochs 30 \
    --batch-size 8 \
    --lr 0.0001 \
    --device cpu
```

**ResNet152**:
```bash
python scripts/train.py \
    --model resnet152 \
    --epochs 30 \
    --batch-size 32 \
    --device cuda
```

**DenseNet201**:
```bash
python scripts/train.py \
    --model densenet201 \
    --epochs 30 \
    --batch-size 32 \
    --device cuda
```

**Ensemble** (99.14% F1 - Combines all 3):
```bash
python scripts/train.py \
    --model ensemble \
    --epochs 20 \
    --batch-size 16 \
    --lr 0.00005 \
    --device cuda
```

### 5. Training Configuration

The training timeout is set to **25 minutes** by default in `config/config.yaml`:

```yaml
training:
  epochs: 30
  timeout_minutes: 25  # Will stop after 25 min if too slow
```

**For slow computers**, training will automatically stop after 25 minutes and save the best model achieved so far.

**To change timeout:**
1. Edit `config/config.yaml`
2. Change `timeout_minutes: 25` to your desired value
3. Or set to `null` for no timeout

### 6. Monitor Training

During training, you'll see:

```
Starting Training: 30 epochs
Device: cuda
Timeout: 25 minutes
============================================================

Epoch 1/30
  Train Loss: 0.5234 | Acc: 0.8234
  Val Loss: 0.4123 | Acc: 0.8567 | F1: 0.8523
  ✓ New best model! F1: 0.8523

Epoch 2/30
  Train Loss: 0.3456 | Acc: 0.8891
  Val Loss: 0.2987 | Acc: 0.9012 | F1: 0.8989
  ✓ New best model! F1: 0.8989
...
```

**Training will automatically stop if:**
- Timeout is reached (25 min default)
- Early stopping triggered (5 epochs without improvement)
- All epochs completed

### 7. Test Your Trained Model

After training, test with the web app:

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 and upload a chest X-ray to test!

## Troubleshooting

### Out of Memory Error

**GPU Memory Error:**
```bash
# Reduce batch size
python scripts/train.py --model efficientnet --batch-size 16 --device cuda
```

**CPU Memory Error:**
```bash
# Use smaller batch size
python scripts/train.py --model efficientnet --batch-size 4 --device cpu
```

### CUDA Not Available

**Install PyTorch with CUDA:**
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

**Check if CUDA is available:**
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Shows your GPU
```

### Training Too Slow

**Solutions:**
1. **Reduce batch size**: `--batch-size 8`
2. **Reduce epochs**: `--epochs 15`
3. **Use smaller model**: Try ResNet152 instead of Ensemble
4. **Increase timeout**: Edit `config/config.yaml`

### Download Dataset Fails

**Manual download:**
```bash
# Install Kaggle CLI
pip install kaggle

# Download datasets
kaggle datasets download paultimothymooney/chest-xray-pneumonia
kaggle datasets download tawsifurrahman/covid19-radiography-database

# Extract and organize manually
unzip chest-xray-pneumonia.zip
unzip covid19-radiography-database.zip
```

## Expected Training Times

| Hardware | Model | Batch Size | Time per Epoch | Total Time (30 epochs) |
|----------|-------|------------|----------------|----------------------|
| RTX 4090 | EfficientNet | 32 | 2-3 min | ~1-1.5 hours |
| RTX 3060 | EfficientNet | 32 | 4-5 min | ~2-2.5 hours |
| GTX 1660 | EfficientNet | 16 | 6-8 min | ~3-4 hours |
| CPU (i7) | EfficientNet | 4 | 30-40 min | ~15-20 hours |
| CPU (i5) | EfficientNet | 4 | 50-60 min | ~25-30 hours |

**Note**: With the 25-minute timeout, CPU training will only complete 1-2 epochs but will save the best model.

## Performance Tips

### For Faster Training:
1. **Use GPU** if available
2. **Increase batch size** (if GPU memory allows)
3. **Enable mixed precision** (automatic in config)
4. **Use fewer workers**: `--num-workers 2`

### For Better Accuracy:
1. **Train longer**: `--epochs 50`
2. **Lower learning rate**: `--lr 0.00005`
3. **Use ensemble model**
4. **Adjust class weights** in `config/config.yaml`

## Saving and Using Models

### Model Files

After training, models are saved in `models/`:
```
models/
├── best_model.pth           # Latest best model
├── checkpoint_epoch_5.pth   # Checkpoint every 5 epochs
├── checkpoint_epoch_10.pth
├── training_history.json    # Training metrics
```

### Using Trained Models

**In Streamlit App:**
```bash
# Just run - it auto-loads from models/
streamlit run app/streamlit_app.py
```

**In Python Script:**
```python
from src.models import EfficientNetB3Model
import torch

# Load model
model = EfficientNetB3Model(num_classes=4)
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make prediction
output = model(image_tensor)
prediction = torch.argmax(output, dim=1)
```

## Next Steps

1. **Evaluate your model**: `python scripts/evaluate.py`
2. **Test on new images**: Use the Streamlit app
3. **Deploy**: See [USAGE.md](USAGE.md) for deployment options
4. **Expand to more diseases**: See [EXPANDING.md](EXPANDING.md)

## Support

Having issues? Check:
- [USAGE.md](USAGE.md) - General usage guide
- [README.md](README.md) - Project overview
- GitHub Issues: https://github.com/kitsakisGk/Healthcare-Detection/issues
