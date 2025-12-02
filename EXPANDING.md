# Expanding to More Diseases

Guide for adding new diseases and medical conditions to the detection system.

## Overview

The current system detects **4 pneumonia classes**. This guide shows how to expand it to detect additional diseases like:
- Tuberculosis
- Lung cancer
- Cardiovascular diseases
- Diabetic retinopathy
- Brain tumors
- Skin diseases
- etc.

## Architecture Benefits

The system is designed for **easy expansion**:

✅ **Modular Code** - Add new models without changing existing code
✅ **Flexible Data Pipeline** - Works with any medical image type
✅ **Transfer Learning Ready** - Pre-trained models adapt quickly
✅ **Config-Driven** - Change classes via configuration files

## Adding New Disease Categories

### Method 1: Extend Existing Pneumonia Model

**Best for**: Adding more lung diseases (TB, cancer, etc.)

#### Step 1: Update Configuration

Edit `config/config.yaml`:

```yaml
data:
  num_classes: 7  # Changed from 4
  class_names: [
    "NORMAL",
    "BACTERIAL",
    "VIRAL",
    "COVID19",
    "TUBERCULOSIS",    # New
    "LUNG_CANCER",     # New
    "PULMONARY_EDEMA"  # New
  ]
```

#### Step 2: Organize Dataset

Create new folders in `data/`:

```
data/
├── train/
│   ├── NORMAL/
│   ├── BACTERIAL/
│   ├── VIRAL/
│   ├── COVID19/
│   ├── TUBERCULOSIS/      # New
│   ├── LUNG_CANCER/       # New
│   └── PULMONARY_EDEMA/   # New
├── val/
└── test/
```

#### Step 3: Add Data

Download datasets for new diseases:
- **Tuberculosis**: Kaggle - `nikhilpandey360/chest-xray-masks-and-labels`
- **Lung Cancer**: Kaggle - `adityamahimkar/iqothnccd-lung-cancer-dataset`

Or use your own labeled data.

#### Step 4: Retrain Models

```bash
python scripts/train.py \
    --model efficientnet \
    --epochs 30 \
    --device cuda
```

The model will automatically detect 7 classes instead of 4!

#### Step 5: Update Streamlit App

Edit `app/streamlit_app.py` - add CSS classes for new diseases:

```python
.tuberculosis {
    background-color: #fff3cd;
    color: #856404;
    border: 2px solid #ffc107;
}

.lung_cancer {
    background-color: #f8d7da;
    color: #721c24;
    border: 2px solid #f5c6cb;
}
```

### Method 2: Create Separate Disease Detector

**Best for**: Completely different organs/images (eyes, brain, skin)

#### Step 1: Create New Config

Create `config/config_retinopathy.yaml`:

```yaml
project:
  name: "diabetic-retinopathy-detection"
  version: "1.0.0"

data:
  num_classes: 5
  class_names: [
    "NO_DR",
    "MILD",
    "MODERATE",
    "SEVERE",
    "PROLIFERATIVE"
  ]
  img_size: 224
  batch_size: 32

training:
  epochs: 30
  timeout_minutes: 25
```

#### Step 2: Train Separate Model

```bash
python scripts/train.py \
    --model efficientnet \
    --config config/config_retinopathy.yaml \
    --data-dir data_retinopathy \
    --save-dir models/retinopathy
```

#### Step 3: Create Separate App (Optional)

```bash
cp app/streamlit_app.py app/retinopathy_app.py
# Edit to load retinopathy model
streamlit run app/retinopathy_app.py --server.port 8502
```

## Example: Adding Tuberculosis Detection

### Complete Example

**1. Download TB Dataset**

```bash
# Using Kaggle API
kaggle datasets download -d nikhilpandey360/chest-xray-masks-and-labels
unzip chest-xray-masks-and-labels.zip -d data_multiclass/raw_tb/
```

**2. Organize TB Images**

```python
import shutil
from pathlib import Path
import random

random.seed(42)

# Collect TB images
tb_images = list(Path('data_multiclass/raw_tb/Tuberculosis').glob('*.png'))
random.shuffle(tb_images)

# 80/10/10 split
n_train = int(len(tb_images) * 0.8)
n_val = int(len(tb_images) * 0.9)

splits = {
    'train': tb_images[:n_train],
    'val': tb_images[n_train:n_val],
    'test': tb_images[n_val:]
}

# Copy to organized structure
for split, images in splits.items():
    dest = Path(f'data_multiclass/{split}/TUBERCULOSIS')
    dest.mkdir(exist_ok=True)
    for img in images:
        shutil.copy2(img, dest / img.name)

print(f"✓ Organized {len(tb_images)} TB images")
```

**3. Update Config**

```yaml
data:
  num_classes: 5
  class_names: ["NORMAL", "BACTERIAL", "VIRAL", "COVID19", "TUBERCULOSIS"]
```

**4. Retrain**

```bash
python scripts/train.py --model ensemble --epochs 30 --device cuda
```

**Done!** Your model now detects 5 classes including TB.

## Working with Different Image Types

### Retinal Images (Diabetic Retinopathy)

```python
# In src/data/preprocessing.py
class RetinalPreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            # Retinal-specific augmentations
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomRotation(360),  # Full rotation OK for eyes
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
```

### Brain MRI Scans

```python
# Different preprocessing for MRI
class MRIPreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            # MRI-specific normalization
            transforms.Normalize([0.5], [0.5])
        ])
```

### Skin Lesions

```python
# For dermoscopic images
class SkinPreprocessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # Hair removal augmentation
            transforms.ColorJitter(saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
```

## Multi-Disease Platform

### Option 1: Single Unified Model

**Pros**: One model, simpler deployment
**Cons**: Lower accuracy per disease

```yaml
data:
  num_classes: 15
  class_names: [
    # Lung diseases
    "NORMAL_LUNG", "PNEUMONIA", "TB", "LUNG_CANCER",
    # Eye diseases
    "NORMAL_EYE", "DIABETIC_RETINOPATHY", "GLAUCOMA",
    # Brain diseases
    "NORMAL_BRAIN", "BRAIN_TUMOR", "STROKE",
    # etc...
  ]
```

### Option 2: Multiple Specialized Models (Recommended)

**Pros**: Higher accuracy, easier to maintain
**Cons**: Multiple models to deploy

```
models/
├── pneumonia/
│   ├── ensemble_pneumonia.pth
│   └── config.yaml
├── retinopathy/
│   ├── efficientnet_retinopathy.pth
│   └── config.yaml
├── brain_tumor/
│   ├── resnet_brain.pth
│   └── config.yaml
```

**Streamlit Multi-Disease App:**

```python
# In app/multi_disease_app.py
import streamlit as st

disease_type = st.sidebar.selectbox(
    "Select Disease Type",
    ["Pneumonia", "Diabetic Retinopathy", "Brain Tumor", "Skin Lesions"]
)

if disease_type == "Pneumonia":
    model = load_pneumonia_model()
elif disease_type == "Diabetic Retinopathy":
    model = load_retinopathy_model()
# etc...

prediction = model(uploaded_image)
```

## Public Datasets for Expansion

### Lung Diseases
- **Tuberculosis**: `nikhilpandey360/chest-xray-masks-and-labels`
- **Lung Cancer**: `adityamahimkar/iqothnccd-lung-cancer-dataset`
- **COVID-19**: `tawsifurrahman/covid19-radiography-database` (already using)

### Eye Diseases
- **Diabetic Retinopathy**: `tanlikesmath/diabetic-retinopathy-resized`
- **Glaucoma**: `sshikamaru/glaucoma-detection`
- **Cataract**: `jr2ngb/cataractdataset`

### Brain
- **Brain Tumor**: `masoudnickparvar/brain-tumor-mri-dataset`
- **Alzheimer's**: `tourist55/alzheimers-dataset-4-class-of-images`
- **Stroke**: `afridirahman/ischemic-stroke-dataset`

### Skin
- **Skin Cancer**: `nodoubttome/skin-cancer9-classesisic`
- **Acne**: `rutviklathiyan/acne-grading-classificationdataset`
- **Melanoma**: `hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images`

### Other
- **Breast Cancer**: `awsaf49/cbis-ddsm-breast-cancer-image-dataset`
- **Kidney Disease**: `nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone`
- **Bone Fracture**: `bmadushanirodrigo/fracture-multi-region-x-ray-data`

## Best Practices

### 1. Data Quality
- **Minimum 500 images per class** for good accuracy
- **Balanced classes** or use class weights
- **High-quality labels** - verify with medical professionals
- **Diverse sources** - different hospitals, scanners

### 2. Model Selection
- **EfficientNet-B3**: Best for most medical images
- **ResNet152**: Good for X-rays, CT scans
- **DenseNet201**: Good for MRI, complex textures
- **Ensemble**: Best accuracy but slower

### 3. Validation
- **Hold-out test set** never seen during training
- **Cross-validation** for small datasets
- **Clinical validation** with medical professionals
- **Error analysis** - understand failure cases

### 4. Deployment
- **Start with one disease** - get it working well first
- **Add gradually** - don't try everything at once
- **Monitor performance** - track accuracy over time
- **Update models** - retrain with new data regularly

## Regulatory Considerations

⚠️ **Important**: Medical AI systems require regulatory approval:

- **FDA (USA)**: Class II or III medical device
- **CE Mark (Europe)**: Medical Device Regulation (MDR)
- **PMDA (Japan)**: Medical device approval

**This system is for research and educational purposes only.**

For clinical use:
1. Validate with large clinical trials
2. Get regulatory approval
3. Implement quality management system (ISO 13485)
4. Ensure HIPAA/GDPR compliance

## Example Roadmap

### Phase 1: Pneumonia (✅ Complete)
- 4 classes: NORMAL, BACTERIAL, VIRAL, COVID-19
- 99.14% F1 score
- Production deployment

### Phase 2: Expanded Lung Diseases
- Add: Tuberculosis, Lung Cancer, Pulmonary Edema
- Target: 90%+ accuracy per class
- Timeline: 2-3 months

### Phase 3: Multi-Organ Platform
- Add: Diabetic Retinopathy, Brain Tumors
- Separate specialized models
- Multi-disease web interface
- Timeline: 4-6 months

### Phase 4: Clinical Integration
- DICOM support
- HL7 FHIR integration
- Electronic Health Records (EHR) integration
- Clinical trials
- Timeline: 6-12 months

## Support

Need help expanding?
- Check existing issues: https://github.com/kitsakisGk/Healthcare-Detection/issues
- Open a new issue with your use case
- Contributions welcome!
