# Usage Guide

Complete guide for using the Healthcare Detection system.

## Table of Contents

1. [Installation](#installation)
2. [Running the Web Application](#running-the-web-application)
3. [Training Models](#training-models)
4. [Evaluation](#evaluation)
5. [Deployment](#deployment)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for training

### Step 1: Clone Repository

```bash
git clone https://github.com/kitsakisGk/Healthcare-Detection.git
cd Healthcare-Detection
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

### Step 3: Download Pre-trained Models

Download the pre-trained models from the releases page and place them in the `models/` directory:

```
models/
├── efficientnet_b3_pneumonia.pth
├── resnet152_pneumonia.pth
├── densenet201_pneumonia.pth
└── ensemble_pneumonia.pth
```

## Running the Web Application

### Local Deployment

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Upload X-Ray Image**: Click "Browse files" or drag-and-drop
2. **View Prediction**: See the predicted class and confidence scores
3. **Analyze Grad-CAM**: View heatmap showing which regions influenced the decision
4. **Download Report**: Export results for clinical records

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- DICOM (.dcm)

## Training Models

### Option 1: Google Colab (Recommended for Beginners)

1. Open `notebooks/train_multiclass_colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells in order
4. Training time: 6-8 hours for all 4 models
5. Download trained models from Google Drive

### Option 2: Local Training (Requires GPU)

#### Step 1: Prepare Dataset

```bash
python scripts/download_datasets.py
```

This will download and organize the datasets automatically.

#### Step 2: Train Individual Models

**EfficientNet-B3** (Best single model):
```bash
python scripts/train.py \
    --model efficientnet \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.0001 \
    --device cuda
```

**ResNet152**:
```bash
python scripts/train.py \
    --model resnet152 \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.0001 \
    --device cuda
```

**DenseNet201**:
```bash
python scripts/train.py \
    --model densenet201 \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.0001 \
    --device cuda
```

**Ensemble** (Best overall):
```bash
python scripts/train.py \
    --model ensemble \
    --epochs 20 \
    --batch-size 16 \
    --lr 0.00005 \
    --device cuda
```

#### Training Parameters

- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.0001)
- `--device`: Device to use (cuda/cpu/auto)
- `--num-workers`: Number of data loading workers (default: 4)
- `--save-dir`: Directory to save models (default: models/)

## Evaluation

### Evaluate a Trained Model

```bash
python scripts/evaluate.py \
    --model-type ensemble \
    --model-path models/ensemble_pneumonia.pth \
    --data-dir data/chest_xray/test
```

### Output

The evaluation script generates:
- Confusion matrix
- Classification report (Precision, Recall, F1-Score)
- ROC curves
- Per-class performance metrics

Results are saved in `reports/` directory.

## Inference on Single Images

```bash
python scripts/inference.py \
    --image path/to/xray.jpg \
    --model-path models/ensemble_pneumonia.pth \
    --model-type ensemble
```

### Batch Inference

```bash
python scripts/inference.py \
    --image-dir path/to/xray/folder \
    --model-path models/ensemble_pneumonia.pth \
    --model-type ensemble \
    --output results.csv
```

## Deployment

### Docker Deployment

1. Build Docker image:
```bash
docker build -t healthcare-detection .
```

2. Run container:
```bash
docker run -p 8501:8501 healthcare-detection
```

### Hugging Face Spaces

The application can be deployed to Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Select "Streamlit" as the SDK
3. Upload the repository files
4. The app will deploy automatically

Live Demo: [https://huggingface.co/spaces/kitsakisG/Pneumonia-Detection](https://huggingface.co/spaces/kitsakisG/Pneumonia-Detection)

## Configuration

Edit `config/config.yaml` to customize:

- Model architectures
- Training hyperparameters
- Data augmentation settings
- Class weights
- Output directories

## Troubleshooting

### Model Not Loading in Web App

**Error**: "Ensemble model not available"

**Solution**:
- Ensure models are in `models/` directory
- Check file names match exactly:
  - `ensemble_pneumonia.pth`
  - `efficientnet_b3_pneumonia.pth`
  - `resnet152_pneumonia.pth`
  - `densenet201_pneumonia.pth`

### Out of Memory During Training

**Solution**: Reduce batch size:
```bash
python scripts/train.py --model efficientnet --batch-size 16
```

### CUDA Not Available

**Solution**: Install PyTorch with CUDA:
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Performance Tips

### For Training
- Use GPU (10-20x faster than CPU)
- Increase batch size if you have GPU memory
- Use mixed precision training for faster training

### For Inference
- Use ensemble model for best accuracy
- Use EfficientNet-B3 for faster single-model inference
- Batch multiple images together for efficiency

## Support

For issues and questions:
- GitHub Issues: https://github.com/kitsakisGk/Healthcare-Detection/issues
- Documentation: Check README.md and code comments

## License

MIT License - See LICENSE file for details
