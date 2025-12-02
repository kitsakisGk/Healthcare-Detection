"""
AI-Powered Pneumonia Detection Web Application

Author: Georgios Kitsakis
Date: 2025-10-29

Description:
Interactive web app for multi-class pneumonia detection from chest X-rays.
Classes: NORMAL, BACTERIAL, VIRAL, COVID-19
Features: Grad-CAM explainability, uncertainty quantification, risk stratification.
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="lung",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .normal {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .bacterial {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .viral {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }
    .covid {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 2px solid #bee5eb;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Device configuration
@st.cache_resource
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

# Load model
@st.cache_resource
def load_model():
    """Load the trained multi-class model (4 classes)"""
    import sys
    sys.path.insert(0, 'src')

    # Try to load Ensemble model (best overall)
    try:
        from models.ensemble import EnsembleModel
        model = EnsembleModel(num_classes=4, pretrained=False)
        model_path = 'models/ensemble_pneumonia.pth'
        model_name = 'Ensemble (ResNet152 + DenseNet201 + EfficientNet-B3)'

        # Try loading model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()
        return model, model_name
    except Exception as e:
        pass  # Silently try next option

    # Fallback to EfficientNet-B3
    try:
        from torchvision.models import efficientnet_b3
        model = efficientnet_b3(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.18, inplace=True),
            nn.Linear(512, 4)  # 4 classes
        )
        model_path = 'models/efficientnet_b3_pneumonia.pth'
        model_name = 'EfficientNet-B3'

        # Try loading model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()
        return model, model_name
    except Exception as e:
        pass  # Silently try next option

    # Last fallback: untrained ResNet50
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 4)  # 4 classes
    )
    model = model.to(device)
    model.eval()
    return model, 'ResNet50 (Untrained - Demo Only)'

# Load Grad-CAM
@st.cache_resource
def load_gradcam(_model):
    """Initialize Grad-CAM with the model"""
    # Auto-detect target layer
    try:
        # For ensemble model, use EfficientNet's features
        if hasattr(_model, 'efficientnet_b3'):
            target_layers = [_model.efficientnet_b3.backbone.features[-1]]
        elif hasattr(_model, 'layer4'):
            target_layers = [_model.layer4[-1]]
        elif hasattr(_model, 'features'):
            target_layers = [_model.features[-1]]
        elif hasattr(_model, 'backbone') and hasattr(_model.backbone, 'features'):
            target_layers = [_model.backbone.features[-1]]
        else:
            # Try to get last conv layer
            conv_layers = [m for m in _model.modules() if isinstance(m, nn.Conv2d)]
            if conv_layers:
                target_layers = [conv_layers[-1]]
            else:
                target_layers = None
    except:
        target_layers = None

    if target_layers:
        try:
            cam = GradCAM(model=_model, target_layers=target_layers)
            return cam
        except:
            return None
    return None

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model input"""
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Convert to grayscale first
    if image.mode != 'L':
        image = image.convert('L')

    # For visualization
    img_np = np.array(image.resize((224, 224)))
    img_rgb = np.stack([img_np, img_np, img_np], axis=-1) / 255.0

    # For model
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    return input_tensor, img_rgb

def predict_with_gradcam(model, cam, image):
    """Make prediction and generate Grad-CAM"""
    input_tensor, img_rgb = preprocess_image(image)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100

        # Get all class probabilities
        class_probs = [probabilities[0][i].item() * 100 for i in range(4)]

    # Generate Grad-CAM
    if cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        # Overlay heatmap
        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
    else:
        visualization = img_rgb

    return predicted_class, confidence, class_probs, visualization, img_rgb

# Main app
def main():
    st.markdown('<div class="main-header">AI-Powered Multi-Class Pneumonia Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep Learning Model with Explainable AI (Grad-CAM)</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Information")
        st.markdown("""
        This application uses deep learning models trained on chest X-ray images
        to detect pneumonia types.

        **Features:**
        - Multi-class classification (4 classes)
        - Transfer learning with ImageNet pre-training
        - Grad-CAM explainability visualizations
        - Advanced architecture (EfficientNet-B3/ResNet50)

        **Classes:**
        - NORMAL: Healthy lung tissue
        - BACTERIAL: Bacterial pneumonia
        - VIRAL: Viral pneumonia
        - COVID-19: COVID-19 pneumonia
        """)

        st.markdown("---")

        st.header("Instructions")
        st.markdown("""
        1. Upload a chest X-ray image (JPEG/PNG)
        2. Wait for analysis (2-3 seconds)
        3. View prediction and confidence scores
        4. Examine Grad-CAM heatmap

        **Grad-CAM Colors:**
        - Red/Hot: High attention
        - Blue/Cold: Low attention
        """)

        st.markdown("---")

        st.header("Disclaimer")
        st.warning("""
        This is a **research project** for educational purposes.

        **NOT for clinical use.** Always consult qualified medical
        professionals for diagnosis and treatment.
        """)

        st.markdown("---")
        st.markdown("**Author:** Georgios Kitsakis  ")
        st.markdown("**Framework:** PyTorch 2.8.0")

    # Main content
    try:
        model, model_name = load_model()
        cam = load_gradcam(model)

        if "Untrained" in model_name:
            st.info(f"🎓 **Demo Mode**: Using **{model_name}**. For trained models, use the **Interactive Training** tab to train your own model!")
        else:
            st.success(f"✅ Model loaded successfully! Using: **{model_name}** | Device: **{device.type.upper()}**")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model file exists in the models/ directory")
        return

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a chest X-ray image in JPEG or PNG format"
    )

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)

        # Create columns for layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original X-Ray")
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Analysis")
            with st.spinner("Analyzing image... Please wait..."):
                try:
                    # Make prediction
                    pred_class, confidence, class_probs, gradcam_img, original_img = predict_with_gradcam(model, cam, image)

                    # Display prediction
                    class_names = ['NORMAL', 'BACTERIAL', 'VIRAL', 'COVID-19']
                    class_styles = ['normal', 'bacterial', 'viral', 'covid']
                    prediction = class_names[pred_class]
                    style = class_styles[pred_class]

                    # Prediction box with appropriate styling
                    if pred_class == 0:  # NORMAL
                        icon = "check"
                    else:
                        icon = "warning"

                    st.markdown(
                        f'<div class="prediction-box {style}">{icon} {prediction}</div>',
                        unsafe_allow_html=True
                    )

                    # Confidence metrics
                    st.markdown("### Confidence Scores")
                    st.metric("Primary Prediction", f"{confidence:.2f}%")

                    # All class probabilities
                    st.markdown("#### All Class Probabilities")
                    for i, (class_name, prob) in enumerate(zip(class_names, class_probs)):
                        st.progress(prob / 100, text=f"{class_name}: {prob:.1f}%")

                    # Risk assessment
                    st.markdown("---")
                    st.markdown("### Risk Assessment")
                    if confidence >= 90:
                        st.success(f"High Confidence: Model is very confident in {prediction} prediction")
                    elif confidence >= 70:
                        st.info(f"Medium Confidence: Model is moderately confident in {prediction} prediction")
                    else:
                        st.warning(f"Low Confidence: Prediction uncertain. Consider additional diagnostic tests")

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    return

        # Grad-CAM visualization
        st.markdown("---")
        st.markdown("## Explainability: Grad-CAM Heatmap")

        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **What is Grad-CAM?**
        Gradient-weighted Class Activation Mapping shows which regions of the X-ray
        the AI model focuses on to make its prediction. Red/hot areas indicate high
        attention, while blue/cold areas show low attention.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Original")
            st.image(original_img, width='stretch')

        with col4:
            st.subheader("Grad-CAM Heatmap")
            st.image(gradcam_img, width='stretch')

        # Interpretation guide
        st.markdown("---")
        st.markdown("### Interpretation Guide")

        col5, col6 = st.columns(2)

        with col5:
            st.markdown("""
            **Good Signs:**
            - Heatmap focuses on lung fields
            - Attention on infiltrates/consolidations (if pneumonia)
            - More diffuse attention (if normal)
            """)

        with col6:
            st.markdown("""
            **Warning Signs:**
            - Focus on image edges or corners
            - Attention on non-anatomical features
            - May indicate spurious correlations
            """)

        # Download results
        st.markdown("---")
        st.subheader("Download Results")

        # Create downloadable figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.imshow(original_img, cmap='gray')
        ax1.set_title('Original X-Ray', fontsize=13, fontweight='bold')
        ax1.axis('off')

        ax2.imshow(gradcam_img)
        ax2.set_title('Grad-CAM Heatmap', fontsize=13, fontweight='bold')
        ax2.axis('off')

        fig.suptitle(
            f'Prediction: {prediction} ({confidence:.1f}% confidence)',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)

        st.download_button(
            label="Download Analysis Report (PNG)",
            data=buf,
            file_name=f"pneumonia_analysis_{prediction.lower()}.png",
            mime="image/png"
        )

        plt.close()

    else:
        st.info("Please upload a chest X-ray image to begin analysis")

        st.markdown("---")
        st.markdown("### Sample Images")
        st.markdown("""
        You can test the application with sample images from the test dataset:
        - Normal cases: `data_multiclass/test/NORMAL/`
        - Bacterial pneumonia: `data_multiclass/test/BACTERIAL/`
        - Viral pneumonia: `data_multiclass/test/VIRAL/`
        - COVID-19 cases: `data_multiclass/test/COVID19/`
        """)

if __name__ == "__main__":
    main()
