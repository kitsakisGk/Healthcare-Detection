"""
Interactive Model Training Interface
Allows users to upload datasets, configure hyperparameters, and train models interactively
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import zipfile
import tempfile
import shutil
from PIL import Image
import sys
import time
import json
from pathlib import Path
import numpy as np
import plotly.figure_factory as ff

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import EfficientNetB3Model, ResNet152Model, DenseNet201Model
from src.data import get_transforms, MultiClassDataset
from src.utils import generate_gradcam_overlay, get_top_k_predictions, compute_confusion_matrix
from torch.utils.data import DataLoader

# Page config
st.set_page_config(
    page_title="Interactive Training - Healthcare Detection",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.metric-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0px;
}
.success-box {
    background-color: #d4edda;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #28a745;
}
.info-box {
    background-color: #d1ecf1;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #17a2b8;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Interactive Model Training")
st.markdown("**Train your own disease detection model with custom data and hyperparameters!**")

# Initialize session state
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'confusion_matrix_data' not in st.session_state:
    st.session_state.confusion_matrix_data = None
if 'model_type_trained' not in st.session_state:
    st.session_state.model_type_trained = None
if 'test_dataloader' not in st.session_state:
    st.session_state.test_dataloader = None

# Sidebar - Configuration
st.sidebar.header("⚙️ Training Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Architecture",
    ["EfficientNet-B3 (Recommended)", "ResNet152", "DenseNet201"],
    help="EfficientNet-B3 offers best accuracy for medical images"
)

# Hyperparameters
st.sidebar.subheader("Hyperparameters")

epochs = st.sidebar.slider(
    "Number of Epochs",
    min_value=1,
    max_value=50,
    value=10,
    help="More epochs = better training but takes longer. Start with 10 for testing."
)

batch_size = st.sidebar.selectbox(
    "Batch Size",
    [4, 8, 16, 32],
    index=2,
    help="Larger batch = faster but needs more memory. Use 4-8 for CPU."
)

learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001],
    value=0.0001,
    help="How fast the model learns. 0.0001 is a good default."
)

device_type = st.sidebar.radio(
    "Training Device",
    ["Auto-detect", "CPU", "GPU (CUDA)"],
    help="GPU is 10-20x faster than CPU"
)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Dataset", "🚀 Train Model", "🧪 Test Model", "🎨 Grad-CAM"])

# TAB 1: Dataset Upload
with tab1:
    st.header("Step 1: Upload Your Dataset")

    st.markdown("""
    <div class='info-box'>
    <b>Dataset Requirements:</b><br>
    • ZIP file containing folders for each disease class<br>
    • Each folder should contain images (.jpg, .jpeg, .png)<br>
    • Minimum 50 images per class recommended<br>
    • Example structure:<br>
    <code>
    dataset.zip/<br>
    ├── NORMAL/ (100 images)<br>
    ├── DISEASE_A/ (80 images)<br>
    └── DISEASE_B/ (120 images)
    </code>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Dataset ZIP",
        type=['zip'],
        help="Upload a ZIP file with folders for each disease class"
    )

    if uploaded_file:
        with st.spinner("Extracting and analyzing dataset..."):
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            zip_path = Path(temp_dir) / "dataset.zip"

            # Save uploaded file
            with open(zip_path, 'wb') as f:
                f.write(uploaded_file.getvalue())

            # Extract
            extract_dir = Path(temp_dir) / "extracted"
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Find class folders
            class_folders = [d for d in extract_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

            if len(class_folders) == 0:
                # Check one level deeper
                for subdir in extract_dir.iterdir():
                    if subdir.is_dir():
                        class_folders = [d for d in subdir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                        if class_folders:
                            extract_dir = subdir
                            break

            if len(class_folders) < 2:
                st.error("❌ Could not find class folders! Make sure your ZIP contains folders for each disease class.")
            else:
                # Analyze dataset
                class_info = {}
                total_images = 0

                for folder in class_folders:
                    images = list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + list(folder.glob('*.png'))
                    class_info[folder.name] = len(images)
                    total_images += len(images)

                # Store in session state
                st.session_state.dataset_path = str(extract_dir)
                st.session_state.class_names = list(class_info.keys())
                st.session_state.num_classes = len(class_info)

                # Display summary
                st.success(f"✅ Dataset loaded successfully!")

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Classes", st.session_state.num_classes)
                col2.metric("Total Images", total_images)
                col3.metric("Avg per Class", int(total_images / st.session_state.num_classes))

                # Class distribution
                st.subheader("Class Distribution")

                for class_name, count in class_info.items():
                    percentage = (count / total_images) * 100
                    st.write(f"**{class_name}**: {count} images ({percentage:.1f}%)")
                    st.progress(percentage / 100)

                # Show sample images
                st.subheader("Sample Images from Dataset")
                cols = st.columns(min(4, len(class_folders)))

                for idx, (class_name, folder) in enumerate(zip(class_info.keys(), class_folders)):
                    images = list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + list(folder.glob('*.png'))
                    if images:
                        sample_img = Image.open(images[0])
                        cols[idx % 4].image(sample_img, caption=class_name, width='stretch')

# TAB 2: Training
with tab2:
    st.header("Step 2: Train Your Model")

    if 'dataset_path' not in st.session_state:
        st.warning("⚠️ Please upload a dataset first (Step 1)")
    else:
        st.markdown("""
        <div class='success-box'>
        ✅ Dataset ready! Configure hyperparameters in the sidebar and click "Start Training"
        </div>
        """, unsafe_allow_html=True)

        # Display configuration
        st.subheader("Current Configuration")
        col1, col2, col3 = st.columns(3)
        col1.metric("Model", model_type.split(" ")[0])
        col2.metric("Epochs", epochs)
        col3.metric("Batch Size", batch_size)

        col1, col2, col3 = st.columns(3)
        col1.metric("Learning Rate", learning_rate)
        col2.metric("Classes", st.session_state.num_classes)
        col3.metric("Device", device_type)

        if st.button("🚀 Start Training", type="primary", use_container_width=True):

            # Setup device
            if device_type == "Auto-detect":
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            elif device_type == "GPU (CUDA)":
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            st.info(f"🖥️ Using device: {device}")

            # Create dataset with train/test split
            dataset_path = Path(st.session_state.dataset_path)

            # Load data
            train_transform = get_transforms(224, mode='train')
            val_transform = get_transforms(224, mode='val')

            try:
                dataset = MultiClassDataset(
                    root_dir=dataset_path,
                    transform=train_transform,
                    class_names=st.session_state.class_names
                )

                # Split into train (80%) and test (20%)
                from torch.utils.data import random_split
                test_size = int(len(dataset) * 0.2)
                train_size = len(dataset) - test_size
                train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

                # Set val transform for test set
                test_dataset.dataset.transform = val_transform

                dataloader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0  # Important for Streamlit
                )

                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0
                )

                # Store test dataloader for later use
                st.session_state.test_dataloader = test_dataloader

                # Create model
                if "EfficientNet" in model_type:
                    model = EfficientNetB3Model(num_classes=st.session_state.num_classes)
                    model_type_name = "efficientnet"
                elif "ResNet" in model_type:
                    model = ResNet152Model(num_classes=st.session_state.num_classes)
                    model_type_name = "resnet"
                else:
                    model = DenseNet201Model(num_classes=st.session_state.num_classes)
                    model_type_name = "densenet"

                # Store model type for Grad-CAM
                st.session_state.model_type_trained = model_type_name

                model = model.to(device)

                # Setup training
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                # Training progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_placeholder = st.empty()

                history = {
                    'loss': [],
                    'accuracy': []
                }

                # Training loop
                for epoch in range(epochs):
                    model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0

                    for batch_idx, (images, labels) in enumerate(dataloader):
                        images, labels = images.to(device), labels.to(device)

                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                    # Calculate metrics
                    epoch_loss = running_loss / len(dataloader)
                    epoch_acc = 100. * correct / total

                    history['loss'].append(epoch_loss)
                    history['accuracy'].append(epoch_acc)

                    # Update progress
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

                    # Show live metrics
                    with metrics_placeholder.container():
                        col1, col2 = st.columns(2)
                        col1.line_chart(history['loss'], use_container_width=True)
                        col2.line_chart(history['accuracy'], use_container_width=True)

                # Training complete
                st.success(f"✅ Training Complete! Final Accuracy: {history['accuracy'][-1]:.2f}%")

                # Compute confusion matrix on test set
                with st.spinner("Computing confusion matrix..."):
                    cm_data = compute_confusion_matrix(
                        model,
                        test_dataloader,
                        st.session_state.class_names,
                        device=str(device)
                    )
                    st.session_state.confusion_matrix_data = cm_data

                    # Show quick confusion matrix preview
                    st.subheader("Confusion Matrix (Test Set)")
                    cm = np.array(cm_data['confusion_matrix'])
                    fig_cm = ff.create_annotated_heatmap(
                        z=cm,
                        x=cm_data['class_names'],
                        y=cm_data['class_names'],
                        colorscale='Blues',
                        showscale=True
                    )
                    fig_cm.update_layout(
                        title="Confusion Matrix",
                        xaxis_title="Predicted",
                        yaxis_title="Actual",
                        height=400
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                    # Show overall metrics
                    overall = cm_data['overall_metrics']
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{overall['accuracy']:.3f}")
                    col2.metric("Precision", f"{overall['precision']:.3f}")
                    col3.metric("Recall", f"{overall['recall']:.3f}")
                    col4.metric("F1 Score", f"{overall['f1_score']:.3f}")

                # Save to session state
                st.session_state.training_complete = True
                st.session_state.trained_model = model
                st.session_state.training_history = history

                # Download button
                model_path = f"trained_{model_type.split()[0].lower()}_model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_names': st.session_state.class_names,
                    'num_classes': st.session_state.num_classes,
                    'model_type': model_type_name
                }, model_path)

                with open(model_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Trained Model",
                        data=f,
                        file_name=model_path,
                        mime="application/octet-stream"
                    )

            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.exception(e)

# TAB 3: Testing
with tab3:
    st.header("Step 3: Test Your Trained Model")

    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first (Step 2)")
    else:
        st.markdown("""
        <div class='success-box'>
        ✅ Model trained! Upload an image to test predictions.
        </div>
        """, unsafe_allow_html=True)

        # Show training summary
        st.subheader("Training Summary")
        col1, col2 = st.columns(2)

        final_loss = st.session_state.training_history['loss'][-1]
        final_acc = st.session_state.training_history['accuracy'][-1]

        col1.metric("Final Loss", f"{final_loss:.4f}")
        col2.metric("Final Accuracy", f"{final_acc:.2f}%")

        # Upload test image
        st.subheader("Test Prediction")
        test_image = st.file_uploader("Upload Test Image", type=['jpg', 'jpeg', 'png'])

        if test_image:
            # Load and display image
            image = Image.open(test_image).convert('RGB')

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Test Image", width='stretch')

            with col2:
                # Preprocess
                val_transform = get_transforms(224, mode='val')
                image_tensor = val_transform(image).unsqueeze(0)

                # Get top-3 predictions
                device = next(st.session_state.trained_model.parameters()).device
                predictions = get_top_k_predictions(
                    st.session_state.trained_model,
                    image_tensor,
                    st.session_state.class_names,
                    k=min(3, st.session_state.num_classes),
                    device=str(device)
                )

                # Show top prediction
                top_pred = predictions[0]
                st.markdown(f"""
                <div class='metric-box'>
                <h2 style='text-align: center;'>Prediction: {top_pred['class']}</h2>
                <h3 style='text-align: center; color: #28a745;'>{top_pred['confidence']*100:.2f}% confidence</h3>
                </div>
                """, unsafe_allow_html=True)

                # Show top-3 predictions with colored indicators
                st.subheader("Top Predictions")
                for i, pred in enumerate(predictions, 1):
                    confidence = pred['confidence'] * 100
                    color_emoji = "🟢" if i == 1 else "🟡" if i == 2 else "🟠"
                    st.markdown(f"{color_emoji} **{i}. {pred['class']}** - {confidence:.2f}%")
                    st.progress(pred['confidence'])

# TAB 4: Grad-CAM Interpretability
with tab4:
    st.header("Step 4: Model Interpretability with Grad-CAM")

    if not st.session_state.training_complete:
        st.warning("⚠️ Please train a model first (Step 2)")
    else:
        st.markdown("""
        <div class='info-box'>
        <b>Grad-CAM (Gradient-weighted Class Activation Mapping)</b><br>
        Shows which regions of the medical image the model focuses on to make its prediction.
        This is crucial for clinical trust and understanding AI decisions.
        </div>
        """, unsafe_allow_html=True)

        # Upload image for Grad-CAM
        gradcam_image = st.file_uploader(
            "Upload Image for Grad-CAM Analysis",
            type=['jpg', 'jpeg', 'png'],
            key="gradcam_upload",
            help="Upload a medical image to visualize where the model focuses"
        )

        if gradcam_image:
            image = Image.open(gradcam_image).convert('RGB')

            with st.spinner("Generating Grad-CAM visualization..."):
                try:
                    val_transform = get_transforms(224, mode='val')

                    # Generate Grad-CAM
                    overlay, predicted_class, confidence, heatmap = generate_gradcam_overlay(
                        st.session_state.trained_model,
                        image,
                        val_transform,
                        model_type=st.session_state.model_type_trained
                    )

                    # Display results in 3 columns
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.subheader("📷 Original Image")
                        st.image(image, width='stretch')

                    with col2:
                        st.subheader("🎨 Heatmap")
                        # Convert heatmap to colored image
                        import cv2
                        heatmap_colored = cv2.applyColorMap(
                            np.uint8(255 * heatmap),
                            cv2.COLORMAP_JET
                        )
                        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                        st.image(heatmap_colored, width='stretch')

                    with col3:
                        st.subheader("🔍 Overlay")
                        st.image(overlay, width='stretch')

                    # Prediction details
                    class_name = st.session_state.class_names[predicted_class]
                    st.markdown(f"""
                    <div class='success-box'>
                    <h3>Prediction: {class_name}</h3>
                    <h4>Confidence: {confidence*100:.2f}%</h4>
                    <p><b>Red/Yellow regions</b> show where the model focused to make this prediction.</p>
                    <p>This helps doctors verify the AI is looking at the right areas (lesions, abnormalities)
                    and not artifacts or irrelevant features.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Get top-3 predictions for this image
                    st.subheader("All Predictions for This Image")
                    device = next(st.session_state.trained_model.parameters()).device
                    image_tensor = val_transform(image).unsqueeze(0)
                    predictions = get_top_k_predictions(
                        st.session_state.trained_model,
                        image_tensor,
                        st.session_state.class_names,
                        k=min(3, st.session_state.num_classes),
                        device=str(device)
                    )

                    for i, pred in enumerate(predictions, 1):
                        confidence_pct = pred['confidence'] * 100
                        color_emoji = "🟢" if i == 1 else "🟡" if i == 2 else "🟠"
                        st.markdown(f"{color_emoji} **{i}. {pred['class']}** - {confidence_pct:.2f}%")
                        st.progress(pred['confidence'])

                except Exception as e:
                    st.error(f"❌ Error generating Grad-CAM: {str(e)}")
                    st.exception(e)
                    st.info("💡 Tip: Make sure the image is a valid medical image similar to your training data")

# Educational Section
st.markdown("---")
st.subheader("📚 Understanding the Training Process")

with st.expander("What are Hyperparameters?"):
    st.markdown("""
    **Hyperparameters** are settings you configure before training:

    - **Epochs**: How many times the model sees the entire dataset
      - More epochs = better learning (but can overfit)
      - Start with 10-20 for testing

    - **Batch Size**: How many images processed at once
      - Larger = faster but needs more memory
      - Smaller = slower but uses less memory
      - Typical: 16-32 for GPU, 4-8 for CPU

    - **Learning Rate**: How big the learning steps are
      - Too high = unstable learning
      - Too low = very slow learning
      - Sweet spot: 0.0001 - 0.001

    - **Model Architecture**: The neural network design
      - EfficientNet-B3: Best accuracy, medium speed
      - ResNet152: Good accuracy, larger model
      - DenseNet201: Different approach, good for complex patterns
    """)

with st.expander("How Does Training Work?"):
    st.markdown("""
    **Training Process**:

    1. **Forward Pass**: Image → Neural Network → Prediction
    2. **Calculate Loss**: How wrong is the prediction?
    3. **Backward Pass**: Adjust network weights to reduce error
    4. **Repeat**: Do this for all images, multiple times

    **What You See**:
    - **Loss**: Lower is better (how wrong the model is)
    - **Accuracy**: Higher is better (% of correct predictions)

    **Tips**:
    - Loss should decrease over time
    - Accuracy should increase over time
    - If both plateau, training is complete
    """)

with st.expander("Why Different Results Each Time?"):
    st.markdown("""
    Training involves **randomness**:

    - Random initial weights
    - Random order of images
    - Random data augmentation

    This is **normal and healthy**! Each training run will give slightly different results.

    **To get consistent results**: Set a random seed (advanced)
    """)

with st.expander("What is Grad-CAM and Why Does It Matter?"):
    st.markdown("""
    **Grad-CAM (Gradient-weighted Class Activation Mapping)** is a powerful interpretability technique for medical AI:

    **How it works:**
    - Analyzes the gradients flowing into the last convolutional layer
    - Creates a heatmap showing which image regions influenced the prediction
    - Red/yellow = high importance, blue = low importance

    **Why it's critical for medical imaging:**
    1. **Clinical Trust**: Doctors can verify the AI is looking at actual pathology, not artifacts
    2. **Error Detection**: Catch when models focus on wrong features (e.g., hospital equipment instead of lungs)
    3. **Regulatory Requirement**: FDA and other regulators require explainability for medical AI
    4. **Education**: Helps train junior doctors by highlighting diagnostic features
    5. **Debugging**: Identifies when models learn spurious correlations

    **Example:**
    - ✅ Good: Model highlights lung opacity for pneumonia diagnosis
    - ❌ Bad: Model focuses on patient ID markers or equipment

    **In Healthcare Detection**: Use the Grad-CAM tab to verify your trained model
    focuses on medically relevant regions!
    """)

st.markdown("---")
st.caption("💡 Tip: Start with small epochs (5-10) to test, then increase for final training")
