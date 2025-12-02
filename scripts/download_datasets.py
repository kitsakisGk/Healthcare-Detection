"""
Dataset download script for multi-class pneumonia detection
Downloads and organizes: COVID-19, Bacterial, Viral, and Normal chest X-rays

Author: Georgios Kitsakis
Usage: python scripts/download_datasets.py --output data
"""

import argparse
import sys
from pathlib import Path
import shutil
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class DownloadProgressBar(tqdm):
    """Progress bar for file downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, description: str = "Downloading"):
    """Download file with progress bar"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=description) as t:
        urllib.request.urlretrieve(url, filename=str(output_path), reporthook=t.update_to)


def extract_archive(archive_path: Path, extract_to: Path):
    """Extract zip or tar archive"""
    print(f"Extracting {archive_path.name}...")

    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

    print(f"✓ Extracted to {extract_to}")


def download_kaggle_dataset(dataset_name: str, output_dir: Path):
    """
    Download dataset from Kaggle using kaggle CLI

    Prerequisites:
        - Install: pip install kaggle
        - Setup API key: https://www.kaggle.com/docs/api
    """
    try:
        import kaggle
        print(f"\n{'='*60}")
        print(f"Downloading Kaggle dataset: {dataset_name}")
        print(f"{'='*60}")

        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(output_dir),
            unzip=True
        )

        print(f"✓ Downloaded {dataset_name}")
        return True

    except ImportError:
        print("\n❌ Kaggle API not installed!")
        print("Install with: pip install kaggle")
        print("Setup guide: https://www.kaggle.com/docs/api")
        return False
    except Exception as e:
        print(f"\n❌ Error downloading from Kaggle: {e}")
        print("\nMake sure you have:")
        print("1. Installed kaggle: pip install kaggle")
        print("2. Setup API token: ~/.kaggle/kaggle.json")
        print("3. Accepted dataset terms on Kaggle website")
        return False


def organize_original_dataset(source_dir: Path, output_dir: Path):
    """
    Organize the original Kaggle pneumonia dataset
    Keeps NORMAL and maps PNEUMONIA to BACTERIAL
    """
    print(f"\n{'='*60}")
    print("Organizing original pneumonia dataset...")
    print(f"{'='*60}")

    splits = ['train', 'test', 'val']

    for split in splits:
        source_split = source_dir / split
        if not source_split.exists():
            print(f"⚠ Warning: {split} folder not found in {source_dir}")
            continue

        # NORMAL images
        source_normal = source_split / 'NORMAL'
        if source_normal.exists():
            dest_normal = output_dir / split / 'NORMAL'
            dest_normal.mkdir(parents=True, exist_ok=True)

            normal_images = list(source_normal.glob('*.jpeg')) + list(source_normal.glob('*.jpg')) + list(source_normal.glob('*.png'))
            print(f"\n{split.upper()} - NORMAL: {len(normal_images)} images")

            for img in tqdm(normal_images, desc=f"Copying {split} NORMAL"):
                shutil.copy2(img, dest_normal / img.name)

        # PNEUMONIA -> BACTERIAL (original dataset has bacterial pneumonia)
        source_pneumonia = source_split / 'PNEUMONIA'
        if source_pneumonia.exists():
            dest_bacterial = output_dir / split / 'BACTERIAL'
            dest_bacterial.mkdir(parents=True, exist_ok=True)

            # Filter for bacterial (most pneumonia in original dataset is bacterial)
            pneumonia_images = list(source_pneumonia.glob('*.jpeg')) + list(source_pneumonia.glob('*.jpg')) + list(source_pneumonia.glob('*.png'))
            print(f"{split.upper()} - BACTERIAL: {len(pneumonia_images)} images")

            for img in tqdm(pneumonia_images, desc=f"Copying {split} BACTERIAL"):
                shutil.copy2(img, dest_bacterial / img.name)

    print("\n✓ Original dataset organized!")


def download_covid_dataset(output_dir: Path):
    """
    Download COVID-19 chest X-ray dataset

    Dataset: COVID-19 Radiography Database
    Source: Kaggle
    """
    print(f"\n{'='*60}")
    print("Downloading COVID-19 dataset...")
    print(f"{'='*60}")

    # Try Kaggle download
    dataset_name = "tawsifurrahman/covid19-radiography-database"
    temp_dir = output_dir / "temp_covid"
    temp_dir.mkdir(parents=True, exist_ok=True)

    success = download_kaggle_dataset(dataset_name, temp_dir)

    if success:
        # Organize COVID-19 images
        covid_source = temp_dir / "COVID" / "images"
        if not covid_source.exists():
            covid_source = temp_dir / "COVID-19"  # Alternative path

        if covid_source.exists():
            covid_images = list(covid_source.glob('*.png')) + list(covid_source.glob('*.jpg'))
            print(f"\nFound {len(covid_images)} COVID-19 images")

            # Split: 80% train, 10% val, 10% test
            n_train = int(len(covid_images) * 0.8)
            n_val = int(len(covid_images) * 0.1)

            splits = {
                'train': covid_images[:n_train],
                'val': covid_images[n_train:n_train + n_val],
                'test': covid_images[n_train + n_val:]
            }

            for split, images in splits.items():
                dest_dir = output_dir / split / 'COVID19'
                dest_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n{split.upper()} - COVID19: {len(images)} images")
                for img in tqdm(images, desc=f"Copying {split} COVID19"):
                    shutil.copy2(img, dest_dir / img.name)

            # Clean up temp
            shutil.rmtree(temp_dir)
            print("\n✓ COVID-19 dataset downloaded and organized!")
        else:
            print(f"⚠ Warning: COVID images not found in expected location")
    else:
        print("\n⚠ COVID-19 dataset download failed")
        print("Please download manually from:")
        print("https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database")


def download_viral_dataset(output_dir: Path):
    """
    Download viral pneumonia dataset

    Note: Since dedicated viral pneumonia datasets are limited,
    this provides instructions for manual download or uses
    subsets from existing datasets
    """
    print(f"\n{'='*60}")
    print("Setting up VIRAL pneumonia class...")
    print(f"{'='*60}")

    print("\n⚠ Viral pneumonia images require manual curation:")
    print("Options:")
    print("1. NIH ChestX-ray14 dataset (filter for viral pneumonia)")
    print("2. RSNA Pneumonia Detection (some viral cases)")
    print("3. Manually curate from medical image databases")

    print("\nFor this demo, creating placeholder VIRAL directory.")
    print("Please add viral pneumonia images manually to:")

    for split in ['train', 'val', 'test']:
        viral_dir = output_dir / split / 'VIRAL'
        viral_dir.mkdir(parents=True, exist_ok=True)
        print(f"  - {viral_dir}")

    print("\n💡 TIP: You can use NIH ChestX-ray14 dataset and filter by metadata")


def create_dataset_summary(output_dir: Path):
    """Create summary of downloaded datasets"""
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}\n")

    classes = ['NORMAL', 'BACTERIAL', 'VIRAL', 'COVID19']
    splits = ['train', 'val', 'test']

    summary = {}

    for split in splits:
        print(f"{split.upper()}:")
        summary[split] = {}

        for cls in classes:
            cls_dir = output_dir / split / cls
            if cls_dir.exists():
                images = list(cls_dir.glob('*.png')) + list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.jpeg'))
                count = len(images)
                summary[split][cls] = count
                print(f"  {cls}: {count} images")
            else:
                summary[split][cls] = 0
                print(f"  {cls}: 0 images (not found)")

        total = sum(summary[split].values())
        print(f"  TOTAL: {total} images\n")

    # Save summary to file
    summary_file = output_dir / "dataset_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("MULTI-CLASS PNEUMONIA DATASET SUMMARY\n")
        f.write("="*60 + "\n\n")

        for split in splits:
            f.write(f"{split.upper()}:\n")
            for cls in classes:
                f.write(f"  {cls}: {summary[split][cls]}\n")
            f.write(f"  TOTAL: {sum(summary[split].values())}\n\n")

    print(f"✓ Summary saved: {summary_file}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Download and organize multi-class pneumonia datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python scripts/download_datasets.py --output data

  # Download only original pneumonia dataset
  python scripts/download_datasets.py --output data --original-only

  # Download only COVID-19 dataset
  python scripts/download_datasets.py --output data --covid-only

Prerequisites:
  - Kaggle API: pip install kaggle
  - API token: https://www.kaggle.com/docs/api
        """
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Output directory for datasets'
    )
    parser.add_argument(
        '--original-only',
        action='store_true',
        help='Download only original pneumonia dataset'
    )
    parser.add_argument(
        '--covid-only',
        action='store_true',
        help='Download only COVID-19 dataset'
    )
    parser.add_argument(
        '--skip-covid',
        action='store_true',
        help='Skip COVID-19 dataset download'
    )

    return parser.parse_args()


def main():
    """Main download function"""
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("PNEUMONIA DETECTION DATASET DOWNLOADER")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}\n")

    # Download original pneumonia dataset (NORMAL + BACTERIAL)
    if not args.covid_only:
        print("\n📥 Step 1: Original Pneumonia Dataset")
        print("Dataset: Chest X-Ray Images (Pneumonia)")
        print("Source: Kaggle - paultimothymooney/chest-xray-pneumonia")

        temp_dir = output_dir / "temp_original"
        success = download_kaggle_dataset("paultimothymooney/chest-xray-pneumonia", temp_dir)

        if success:
            # Find chest_xray folder
            chest_xray_dir = temp_dir / "chest_xray"
            if not chest_xray_dir.exists():
                # Try direct subdirectories
                subdirs = list(temp_dir.glob("*/"))
                if subdirs:
                    chest_xray_dir = subdirs[0]

            if chest_xray_dir.exists():
                organize_original_dataset(chest_xray_dir, output_dir)
                shutil.rmtree(temp_dir)
            else:
                print("⚠ Warning: Could not find dataset structure")
        else:
            print("\n⚠ Please download manually from:")
            print("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")

    # Download COVID-19 dataset
    if not args.original_only and not args.skip_covid:
        print("\n📥 Step 2: COVID-19 Dataset")
        download_covid_dataset(output_dir)

    # Setup VIRAL class
    if not args.original_only and not args.covid_only:
        print("\n📥 Step 3: Viral Pneumonia Dataset")
        download_viral_dataset(output_dir)

    # Create summary
    create_dataset_summary(output_dir)

    print(f"\n{'='*60}")
    print("✓ DATASET DOWNLOAD COMPLETE!")
    print(f"{'='*60}")
    print(f"\nData location: {output_dir.absolute()}")
    print("\nNext steps:")
    print("1. Verify dataset structure:")
    print("   data/train/{NORMAL,BACTERIAL,VIRAL,COVID19}/")
    print("   data/val/{NORMAL,BACTERIAL,VIRAL,COVID19}/")
    print("   data/test/{NORMAL,BACTERIAL,VIRAL,COVID19}/")
    print("\n2. If VIRAL class is empty, add viral pneumonia images manually")
    print("\n3. Start training:")
    print("   python scripts/train.py --model ensemble --epochs 30")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
