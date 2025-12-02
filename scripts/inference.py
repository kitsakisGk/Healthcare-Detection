"""
Inference script for single image or batch predictions
Usage: python scripts/inference.py --model-path models/best_model.pth --image test.jpg
"""

import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.models import ResNet152Model, DenseNet201Model, EfficientNetB3Model, EnsembleModel
from src.inference import Predictor, ExplainabilityManager
from src.utils import QualityController, PDFReportGenerator, DICOMHandler


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run inference on chest X-ray images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image prediction
  python scripts/inference.py \\
      --model-path models/best_model.pth \\
      --model-type efficientnet \\
      --image test.jpg

  # With Grad-CAM visualization
  python scripts/inference.py \\
      --model-path models/best_model.pth \\
      --model-type ensemble \\
      --image test.jpg \\
      --gradcam \\
      --save-report results/

  # Batch prediction on directory
  python scripts/inference.py \\
      --model-path models/best_model.pth \\
      --model-type efficientnet \\
      --directory data/test/NORMAL/ \\
      --batch

  # DICOM file support
  python scripts/inference.py \\
      --model-path models/best_model.pth \\
      --model-type ensemble \\
      --image scan.dcm \\
      --dicom \\
      --gradcam
        """
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['resnet152', 'densenet201', 'efficientnet', 'ensemble'],
        help='Type of model architecture'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image file'
    )
    parser.add_argument(
        '--directory',
        type=str,
        help='Path to directory with images (for batch processing)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['cpu', 'cuda', 'auto'],
        help='Device to run on'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch processing mode'
    )
    parser.add_argument(
        '--gradcam',
        action='store_true',
        help='Generate Grad-CAM visualizations'
    )
    parser.add_argument(
        '--uncertainty',
        action='store_true',
        help='Enable uncertainty quantification (slower)'
    )
    parser.add_argument(
        '--quality-check',
        action='store_true',
        help='Perform quality control checks'
    )
    parser.add_argument(
        '--save-report',
        type=str,
        help='Save detailed PDF report to directory'
    )
    parser.add_argument(
        '--dicom',
        action='store_true',
        help='Input is DICOM file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.json',
        help='Output file for predictions (JSON)'
    )

    return parser.parse_args()


def load_model(model_type: str, model_path: str, num_classes: int, device: torch.device):
    """Load trained model"""
    print(f"Loading {model_type} model from {model_path}...")

    # Create model
    if model_type == 'resnet152':
        model = ResNet152Model(num_classes=num_classes)
    elif model_type == 'densenet201':
        model = DenseNet201Model(num_classes=num_classes)
    elif model_type == 'efficientnet':
        model = EfficientNetB3Model(num_classes=num_classes)
    elif model_type == 'ensemble':
        model = EnsembleModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model


def predict_single_image(
    image_path: Path,
    predictor: Predictor,
    explainer: ExplainabilityManager = None,
    qc: QualityController = None,
    dicom_handler: DICOMHandler = None,
    is_dicom: bool = False
):
    """
    Predict on single image with optional features

    Returns:
        Dictionary with prediction results
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*60}")

    result = {
        'image_path': str(image_path),
        'prediction': None,
        'quality_check': None,
        'explainability': None,
        'dicom_metadata': None
    }

    # Handle DICOM files
    original_image = None
    if is_dicom and dicom_handler:
        print("\n📄 DICOM Processing:")
        try:
            # Validate DICOM
            is_valid, msg = dicom_handler.validate_dicom(image_path)
            print(f"  Validation: {msg}")

            if not is_valid:
                result['error'] = f"Invalid DICOM: {msg}"
                return result

            # Extract metadata
            metadata = dicom_handler.extract_metadata(image_path)
            result['dicom_metadata'] = metadata
            print(f"  Patient ID: {metadata.get('PatientID', 'N/A')}")
            print(f"  Study Date: {metadata.get('StudyDate', 'N/A')}")
            print(f"  Modality: {metadata.get('Modality', 'N/A')}")

            # Convert to PIL Image
            original_image = dicom_handler.dicom_to_pil(image_path)

        except Exception as e:
            result['error'] = f"DICOM processing error: {str(e)}"
            return result
    else:
        original_image = Image.open(image_path)

    # Quality control check
    if qc:
        print("\n🔍 Quality Control:")
        qc_result = qc.check_image_quality(original_image)
        result['quality_check'] = qc_result

        if qc_result['overall_pass']:
            print("  ✓ Quality checks PASSED")
        else:
            print("  ✗ Quality checks FAILED")
            for check_name, check_result in qc_result['checks'].items():
                if not check_result['pass']:
                    print(f"    - {check_name}: {check_result['message']}")

        if not qc_result['overall_pass']:
            print("\n⚠ Warning: Image quality issues detected!")
            print("  Prediction may be unreliable.")

    # Make prediction
    print("\n🤖 Running Inference:")
    try:
        prediction = predictor.predict(
            original_image,
            return_probabilities=True,
            return_uncertainty=True
        )
        result['prediction'] = prediction

        # Print results
        print(f"\n  Predicted Class: {prediction['predicted_class']}")
        print(f"  Confidence: {prediction['confidence']*100:.2f}%")
        print(f"  Interpretation: {prediction['confidence_interpretation']}")
        print(f"  Risk Level: {prediction['risk_level'].upper()}")

        if 'uncertainty' in prediction:
            print(f"\n  Uncertainty Metrics:")
            print(f"    Epistemic: {prediction['uncertainty']['epistemic']:.4f}")
            print(f"    Entropy: {prediction['uncertainty']['entropy']:.4f}")

        print(f"\n  Class Probabilities:")
        for i, prob in enumerate(prediction['probabilities']):
            class_name = predictor.class_names[i]
            print(f"    {class_name}: {prob*100:.2f}%")

    except Exception as e:
        result['error'] = f"Prediction error: {str(e)}"
        return result

    # Explainability (Grad-CAM)
    if explainer:
        print("\n🎯 Generating Explainability:")
        try:
            from src.data.preprocessing import get_inference_transforms
            import numpy as np

            # Preprocess image for model
            transform = get_inference_transforms()
            image_tensor = transform(original_image)

            # Generate explanations
            explanations = explainer.explain_prediction(
                image_tensor,
                np.array(original_image),
                target_class=prediction['predicted_class_idx'],
                methods=['gradcam']
            )

            result['explainability'] = {
                'gradcam_generated': True,
                'overlay_shape': explanations['gradcam']['overlay'].shape
            }

            print("  ✓ Grad-CAM visualization generated")

        except Exception as e:
            print(f"  ✗ Explainability generation failed: {e}")
            result['explainability'] = {'error': str(e)}

    return result


def main():
    """Main inference function"""
    args = parse_args()

    # Validate arguments
    if not args.image and not args.directory:
        print("❌ Error: Must specify either --image or --directory")
        sys.exit(1)

    if args.image and args.directory:
        print("❌ Error: Cannot specify both --image and --directory")
        sys.exit(1)

    # Load config
    config = get_config(args.config)

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print("PNEUMONIA DETECTION INFERENCE")
    print(f"{'='*60}")
    print(f"Model: {args.model_type}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Load model
    model = load_model(args.model_type, args.model_path, config.num_classes, device)

    # Create predictor
    predictor = Predictor(
        model=model,
        class_names=config.class_names,
        device=device,
        uncertainty_quantification=args.uncertainty
    )

    # Optional components
    explainer = None
    if args.gradcam:
        explainer = ExplainabilityManager(model, device, enable_gradcam=True)

    qc = None
    if args.quality_check:
        qc = QualityController()

    dicom_handler = None
    if args.dicom:
        dicom_handler = DICOMHandler()

    pdf_generator = None
    if args.save_report:
        pdf_generator = PDFReportGenerator()

    # Single image prediction
    if args.image:
        image_path = Path(args.image)

        if not image_path.exists():
            print(f"❌ Error: Image not found: {image_path}")
            sys.exit(1)

        result = predict_single_image(
            image_path,
            predictor,
            explainer,
            qc,
            dicom_handler,
            args.dicom
        )

        # Save results
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n✓ Results saved to: {output_file}")

        # Generate PDF report
        if args.save_report and pdf_generator:
            report_dir = Path(args.save_report)
            report_path = report_dir / f"report_{image_path.stem}.pdf"

            try:
                pdf_generator.generate_report(
                    output_path=report_path,
                    prediction=result['prediction'],
                    original_image=image_path,
                    quality_check=result.get('quality_check'),
                    metadata=result.get('dicom_metadata')
                )
                print(f"✓ PDF report saved to: {report_path}")
            except Exception as e:
                print(f"✗ PDF generation failed: {e}")

    # Batch prediction
    elif args.directory:
        directory = Path(args.directory)

        if not directory.exists():
            print(f"❌ Error: Directory not found: {directory}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print("BATCH PROCESSING MODE")
        print(f"{'='*60}")

        # Find all images
        extensions = ['*.jpg', '*.jpeg', '*.png']
        if args.dicom:
            extensions.append('*.dcm')

        image_files = []
        for ext in extensions:
            image_files.extend(directory.glob(ext))

        print(f"Found {len(image_files)} images")

        if len(image_files) == 0:
            print("No images found!")
            sys.exit(0)

        # Process each image
        results = []
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]")
            result = predict_single_image(
                img_path,
                predictor,
                explainer if args.gradcam else None,
                qc if args.quality_check else None,
                dicom_handler if args.dicom else None,
                args.dicom
            )
            results.append(result)

        # Save batch results
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"✓ Batch processing complete!")
        print(f"  Processed: {len(results)} images")
        print(f"  Results saved to: {output_file}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
