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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_resized).unsqueeze(0)
    return image_tensor

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
            image_tensor = preprocess_image(image)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted_class].item()
            
            prediction = CLASS_NAMES[predicted_class]
            
            if prediction == "Pneumonia":
                st.error(f"⚠️ **Prediction: {prediction}**")
            else:
                st.success(f"✅ **Prediction: {prediction}**")
            
            st.metric("Confidence", f"{confidence*100:.2f}%")
            
            st.markdown("### Class Probabilities")
            for i, class_name in enumerate(CLASS_NAMES):
                prob = probs[0, i].item()
                st.progress(prob, text=f"{class_name}: {prob*100:.1f}%")
            
            st.info("💡 Grad-CAM visualization available in local deployment")
    else:
        st.info("👆 Upload X-ray to begin")

st.markdown("---")
st.markdown("<div style='text-align:center'><p>EfficientNet-B0 | 90.87% Accuracy</p></div>", unsafe_allow_html=True)
