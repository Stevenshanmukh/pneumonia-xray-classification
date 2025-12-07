"""
Streamlit Web App for Pneumonia Classification
Run with: streamlit run streamlit_app.py
"""
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Page configuration
st.set_page_config(
    page_title="Pneumonia Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CLASS_NAMES = ['Normal', 'Pneumonia']
IMAGE_SIZE = 224
MODEL_PATH = '../models/efficientnet_best.pth'

@st.cache_resource
def load_model():
    """Load the trained EfficientNet model"""
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def preprocess_image(image):
    """Preprocess image for model input"""
    # Resize
    image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    
    # For visualization (0-1 range)
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # For model input (ImageNet normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_resized).unsqueeze(0)
    
    return image_tensor, image_normalized

def generate_gradcam(model, image_tensor, predicted_class):
    """Generate Grad-CAM heatmap"""
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_class)]
    
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    return grayscale_cam

def main():
    # Header
    st.title("Pediatric Pneumonia Detection")
    st.markdown("### AI-Powered Chest X-Ray Analysis")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This application uses deep learning to detect pneumonia 
        from pediatric chest X-rays.
        
        **Model:** EfficientNet-B0  
        **Accuracy:** 90.87%  
        **Sensitivity:** 97.95%  
        **Specificity:** 79.06%
        
        **Warning:**  
        This is a research tool. Always consult 
        medical professionals for diagnosis.
        """)
        
        st.markdown("---")
        st.header("Model Performance")
        st.metric("Test Accuracy", "90.87%")
        st.metric("AUC Score", "0.9580")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a pediatric chest X-ray image"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            
            st.image(image, caption="Uploaded X-Ray", use_column_width=True)
    
    with col2:
        if uploaded_file is not None:
            st.header("Analysis Results")
            
            with st.spinner("Analyzing X-ray..."):
                # Load model
                model = load_model()
                
                # Preprocess
                image_tensor, image_normalized = preprocess_image(image_np)
                
                # Predict
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, predicted_class].item()
                
                # Display prediction
                prediction = CLASS_NAMES[predicted_class]
                
                if prediction == "Pneumonia":
                    st.error(f"**Prediction: {prediction}**")
                else:
                    st.success(f"**Prediction: {prediction}**")
                
                st.metric("Confidence", f"{confidence*100:.2f}%")
                
                # Probability bars
                st.markdown("### Class Probabilities")
                for i, class_name in enumerate(CLASS_NAMES):
                    prob = probs[0, i].item()
                    st.progress(prob, text=f"{class_name}: {prob*100:.1f}%")
                
                # Generate Grad-CAM
                st.markdown("---")
                st.markdown("### Grad-CAM Visualization")
                st.markdown("*Areas of focus for the model's decision*")
                
                with st.spinner("Generating heatmap..."):
                    grayscale_cam = generate_gradcam(model, image_tensor, predicted_class)
                    cam_image = show_cam_on_image(image_normalized, grayscale_cam, use_rgb=True)
                
                # Display Grad-CAM
                fig_col1, fig_col2 = st.columns(2)
                with fig_col1:
                    st.image(grayscale_cam, caption="Heatmap", use_column_width=True, clamp=True)
                with fig_col2:
                    st.image(cam_image, caption="Overlay", use_column_width=True)
                
                st.info("Red areas indicate regions the model focused on for this prediction")
        else:
            st.info("Please upload a chest X-ray image to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit | Model: EfficientNet-B0 | Dataset: Pediatric Chest X-Rays</p>
        <p><strong>For Research Purposes Only - Not for Clinical Use</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
