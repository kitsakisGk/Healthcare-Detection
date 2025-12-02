"""
DICOM file handling for medical imaging
CRITICAL for clinical integration
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import warnings


class DICOMHandler:
    """
    Handle DICOM medical imaging files
    Supports reading, converting, and extracting metadata
    """

    def __init__(self):
        """Initialize DICOM handler"""
        try:
            import pydicom
            self.pydicom = pydicom
            self.available = True
        except ImportError:
            warnings.warn(
                "pydicom not installed. DICOM support disabled. "
                "Install with: pip install pydicom"
            )
            self.available = False
            self.pydicom = None

    def read_dicom(self, dicom_path: Union[str, Path]) -> Optional[object]:
        """
        Read DICOM file

        Args:
            dicom_path: Path to DICOM file

        Returns:
            DICOM dataset object or None if not available
        """
        if not self.available:
            raise RuntimeError("pydicom not installed")

        dicom_path = Path(dicom_path)
        if not dicom_path.exists():
            raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

        try:
            dcm = self.pydicom.dcmread(str(dicom_path))
            return dcm
        except Exception as e:
            raise RuntimeError(f"Error reading DICOM file: {e}")

    def extract_image(
        self,
        dicom_path: Union[str, Path],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extract image array from DICOM file

        Args:
            dicom_path: Path to DICOM file
            normalize: Normalize pixel values to [0, 255]

        Returns:
            Image as numpy array
        """
        dcm = self.read_dicom(dicom_path)
        pixel_array = dcm.pixel_array

        if normalize:
            # Normalize to 0-255 range
            pixel_array = pixel_array.astype(np.float32)
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
            pixel_array = (pixel_array * 255).astype(np.uint8)

        return pixel_array

    def dicom_to_pil(
        self,
        dicom_path: Union[str, Path],
        mode: str = 'L'
    ) -> Image.Image:
        """
        Convert DICOM to PIL Image

        Args:
            dicom_path: Path to DICOM file
            mode: PIL image mode ('L' for grayscale, 'RGB' for color)

        Returns:
            PIL Image
        """
        pixel_array = self.extract_image(dicom_path, normalize=True)

        # Convert to PIL Image
        image = Image.fromarray(pixel_array)

        if mode == 'RGB' and image.mode == 'L':
            image = image.convert('RGB')

        return image

    def extract_metadata(self, dicom_path: Union[str, Path]) -> Dict[str, any]:
        """
        Extract metadata from DICOM file

        Args:
            dicom_path: Path to DICOM file

        Returns:
            Dictionary of metadata
        """
        dcm = self.read_dicom(dicom_path)

        metadata = {}

        # Common DICOM tags
        tags_to_extract = {
            'PatientID': (0x0010, 0x0020),
            'PatientName': (0x0010, 0x0010),
            'PatientBirthDate': (0x0010, 0x0030),
            'PatientSex': (0x0010, 0x0040),
            'StudyDate': (0x0008, 0x0020),
            'StudyTime': (0x0008, 0x0030),
            'Modality': (0x0008, 0x0060),
            'StudyDescription': (0x0008, 0x1030),
            'SeriesDescription': (0x0008, 0x103E),
            'InstitutionName': (0x0008, 0x0080),
            'ViewPosition': (0x0018, 0x5101),
            'Rows': (0x0028, 0x0010),
            'Columns': (0x0028, 0x0011),
            'PixelSpacing': (0x0028, 0x0030),
        }

        for name, tag in tags_to_extract.items():
            try:
                value = dcm[tag].value
                # Convert to string for easier handling
                metadata[name] = str(value) if value is not None else None
            except (KeyError, AttributeError):
                metadata[name] = None

        return metadata

    def validate_dicom(self, dicom_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate DICOM file

        Args:
            dicom_path: Path to DICOM file

        Returns:
            Tuple of (is_valid, message)
        """
        if not self.available:
            return False, "pydicom not installed"

        try:
            dcm = self.read_dicom(dicom_path)

            # Check if it's a valid medical image
            if not hasattr(dcm, 'pixel_array'):
                return False, "DICOM file has no pixel data"

            # Check modality
            modality = dcm.get('Modality', None)
            if modality and modality not in ['CR', 'DX', 'MG', 'CT', 'MR']:
                return False, f"Unsupported modality: {modality}"

            # Check image dimensions
            rows = dcm.get('Rows', 0)
            cols = dcm.get('Columns', 0)
            if rows == 0 or cols == 0:
                return False, "Invalid image dimensions"

            return True, "Valid DICOM file"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def batch_convert_dicom(
        self,
        dicom_dir: Union[str, Path],
        output_dir: Union[str, Path],
        format: str = 'png'
    ) -> Dict[str, str]:
        """
        Batch convert DICOM files to standard image format

        Args:
            dicom_dir: Directory containing DICOM files
            output_dir: Directory to save converted images
            format: Output format ('png', 'jpg')

        Returns:
            Dictionary mapping DICOM filenames to output paths
        """
        dicom_dir = Path(dicom_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        converted = {}

        # Find all DICOM files
        dicom_files = list(dicom_dir.glob('*.dcm')) + list(dicom_dir.glob('*.DCM'))

        print(f"Found {len(dicom_files)} DICOM files")

        for dcm_path in dicom_files:
            try:
                # Convert to PIL Image
                image = self.dicom_to_pil(dcm_path)

                # Save as standard format
                output_path = output_dir / f"{dcm_path.stem}.{format}"
                image.save(output_path)

                converted[dcm_path.name] = str(output_path)
                print(f"✓ Converted: {dcm_path.name}")

            except Exception as e:
                print(f"✗ Failed to convert {dcm_path.name}: {e}")

        print(f"\nConverted {len(converted)}/{len(dicom_files)} files")
        return converted
