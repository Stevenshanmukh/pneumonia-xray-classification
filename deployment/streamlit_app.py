"""
Streamlit Web App for Pneumonia Classification with Grad-CAM
"""
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import requests
from pathlib import Path

st.set_page_config(page_title="Pneumonia Classifier", page_icon="🫁", layout="wide")

CLASS_NAMES = ['Normal', 'Pneumonia']
IMAGE_SIZE = 224
MODEL_PATH = '../models/efficientnet_best.pth'
MODEL_URL = 'https://github.com/Stevenshanmukh/pneumonia-xray-classification/releases/download/v1.0/efficientnet_best.pth'

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

@st.cache_resource
def download_model():
    """Download model if not present"""
    model_dir = Path(MODEL_PATH).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model (one-time, ~46 MB)...'):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                
                with open(MODEL_PATH, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress_bar.progress(min(downloaded / total_size, 1.0))
                
                st.success('Model downloaded!')
                return True
            except Exception as e:
                st.error(f'Failed to download model: {e}')
                st.info('Check GitHub Releases: https://github.com/Stevenshanmukh/pneumonia-xray-classification/releases')
                return False
    return True

@st.cache_resource
def load_model():
    if not download_model():
        st.stop()
    
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(image):
    image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_resized).unsqueeze(0)
    return image_tensor, image_array

def generate_gradcam(model, image_tensor, predicted_class):
    """Generate Grad-CAM heatmap"""
    if not GRADCAM_AVAILABLE:
        return None
    
    try:
        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
        return grayscale_cam
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {e}")
        return None

st.title("🫁 Pediatric Pneumonia Detection")
st.markdown("### AI-Powered Chest X-Ray Analysis")
st.markdown("---")

with st.sidebar:
    st.header("About")
    st.markdown("""
    **Model:** EfficientNet-B0  
    **Accuracy:** 90.87%  
    **Sensitivity:** 97.95%  
    **AUC:** 0.9580
    
    Research tool only.
    
    ---
    
    **Features:**
    - Pneumonia classification
    - Confidence scores
    - Grad-CAM visualization
    
    First run: Model downloads automatically (~46 MB)
    """)
    
    if not GRADCAM_AVAILABLE:
        st.warning("Grad-CAM not available in cloud deployment")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload X-Ray")
    uploaded_file = st.file_uploader("Choose X-ray", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)

with col2:
    if uploaded_file:
        st.header("Results")
        with st.spinner("Analyzing..."):
            try:
                model = load_model()
                image_tensor, image_normalized = preprocess_image(image)
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, predicted_class].item()
                
                prediction = CLASS_NAMES[predicted_class]
                
                if prediction == "Pneumonia":
                    st.error(f"**Prediction: {prediction}**")
                else:
                    st.success(f"**Prediction: {prediction}**")
                
                st.metric("Confidence", f"{confidence*100:.2f}%")
                
                st.markdown("### Class Probabilities")
                for i, class_name in enumerate(CLASS_NAMES):
                    prob = probs[0, i].item()
                    st.progress(prob, text=f"{class_name}: {prob*100:.1f}%")
                
                if GRADCAM_AVAILABLE:
                    st.markdown("---")
                    st.markdown("### Grad-CAM Visualization")
                    with st.spinner("Generating heatmap..."):
                        grayscale_cam = generate_gradcam(model, image_tensor, predicted_class)
                        
                        if grayscale_cam is not None:
                            cam_image = show_cam_on_image(image_normalized, grayscale_cam, use_rgb=True)
                            
                            col_heat, col_over = st.columns(2)
                            with col_heat:
                                st.image(grayscale_cam, caption="Heatmap", use_column_width=True, clamp=True)
                            with col_over:
                                st.image(cam_image, caption="Overlay", use_column_width=True)
                            
                            st.info("Red areas indicate regions the model focused on")
                else:
                    st.info("Grad-CAM visualization not available in cloud deployment. Available in local setup.")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.info("Upload X-ray to begin")

st.markdown("---")
st.markdown("<div style='text-align:center'><p>EfficientNet-B0 | 90.87% Accuracy | Research Use Only</p></div>", unsafe_allow_html=True)
