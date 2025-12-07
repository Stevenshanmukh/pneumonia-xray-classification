"""
Streamlit Web App for Pneumonia Classification
"""
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(page_title="Pneumonia Classifier", page_icon="🫁", layout="wide")

CLASS_NAMES = ['Normal', 'Pneumonia']
IMAGE_SIZE = 224
MODEL_PATH = '../models/efficientnet_best.pth'

@st.cache_resource
def load_model():
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
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
    return grayscale_cam

st.title("🫁 Pediatric Pneumonia Detection")
st.markdown("### AI-Powered Chest X-Ray Analysis")
st.markdown("---")

with st.sidebar:
    st.header("About")
    st.markdown("""
    **Model:** EfficientNet-B0  
    **Accuracy:** 90.87%  
    **Sensitivity:** 97.95%
    
    ⚠️ Research tool only.
    """)

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
            model = load_model()
            image_tensor, image_normalized = preprocess_image(image)
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted_class].item()
            
            prediction = CLASS_NAMES[predicted_class]
            if prediction == "Pneumonia":
                st.error(f"⚠️ **{prediction}**")
            else:
                st.success(f"✅ **{prediction}**")
            
            st.metric("Confidence", f"{confidence*100:.2f}%")
            
            for i, class_name in enumerate(CLASS_NAMES):
                st.progress(probs[0, i].item(), text=f"{class_name}: {probs[0, i].item()*100:.1f}%")
            
            st.markdown("### Grad-CAM")
            with st.spinner("Generating..."):
                grayscale_cam = generate_gradcam(model, image_tensor, predicted_class)
                cam_image = show_cam_on_image(image_normalized, grayscale_cam, use_rgb=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(grayscale_cam, caption="Heatmap", use_column_width=True)
            with c2:
                st.image(cam_image, caption="Overlay", use_column_width=True)
    else:
        st.info("Upload X-ray to begin")

st.markdown("---")
st.markdown("<div style='text-align:center'><p>EfficientNet-B0 | Research Use Only</p></div>", unsafe_allow_html=True)
