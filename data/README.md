# Dataset Directory

This directory should contain your medical imaging datasets.

## Structure

```
data/
├── chest_xray/          # Pneumonia detection dataset
│   ├── train/
│   │   ├── NORMAL/
│   │   ├── BACTERIAL/
│   │   ├── VIRAL/
│   │   └── COVID19/
│   ├── val/
│   └── test/
└── [other_datasets]/    # Future disease detection datasets
```

## Pneumonia Detection Dataset

The current implementation uses two public datasets:

1. **Chest X-Ray Pneumonia Dataset**
   - Source: Kaggle - `paultimothymooney/chest-xray-pneumonia`
   - Classes: NORMAL, BACTERIAL (pneumonia)

2. **COVID-19 Radiography Database**
   - Source: Kaggle - `tawsifurrahman/covid19-radiography-database`
   - Classes: COVID-19, VIRAL (pneumonia)

## Downloading Datasets

### Option 1: Using the Download Script

```bash
python scripts/download_datasets.py
```

This will automatically download and organize both datasets.

### Option 2: Manual Download via Kaggle

1. Install Kaggle API:
   ```bash
   pip install kaggle
   ```

2. Set up Kaggle credentials:
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Place `kaggle.json` in `~/.kaggle/` directory

3. Download datasets:
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   kaggle datasets download -d tawsifurrahman/covid19-radiography-database
   ```

### Option 3: Google Colab

Use the provided Colab notebook (`notebooks/train_multiclass_colab.ipynb`) which includes automated dataset download and organization.

## Dataset Statistics

**Total Images**: ~10,848 images

| Split | NORMAL | BACTERIAL | VIRAL | COVID-19 | Total |
|-------|--------|-----------|-------|----------|-------|
| Train | 1,260  | 2,217     | 1,076 | 2,892    | 7,445 |
| Val   | 157    | 277       | 134   | 362      | 930   |
| Test  | 158    | 278       | 135   | 362      | 933   |

## Image Format

- **Format**: JPEG/PNG
- **Size**: Variable (automatically resized to 224x224 during training)
- **Channels**: Grayscale (converted to 3-channel for model compatibility)

## Data Augmentation

Training includes:
- Random horizontal flips
- Random rotations (±10°)
- Color jitter
- Normalization (ImageNet statistics)

Validation/Test:
- Resize to 224x224
- Normalization only

## License & Attribution

Please review and comply with the original dataset licenses when using this data for research or commercial purposes.
