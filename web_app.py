import streamlit as st
import torch
import os
import numpy as np
from PIL import Image

# 1. Check if model exists, if not, train it!
if not os.path.exists("defect_model.pth"):
    st.info("First time setup: Training the AI model... please wait.")
    import train_and_test 
    train_and_test.train_model() # Make sure this function name matches yours
    st.success("Training complete!")

# 2. Now import the model and transform from your app logic
from app import model, transform
from app import model, transform 

st.set_page_config(page_title="AI Detector", layout="centered")

st.title("🔍 Defect Detector")

# File uploader at the top
uploaded_file = st.file_uploader("Upload BMP image", type=["bmp", "jpg", "png"])

if uploaded_file is not None:
    # Prepare image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns: Left for Image, Right for Results
    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("### Preview")
        # Fixed width of 300px ensures it stays small
        st.image(image, width=300)

    with col2:
        st.write("### AI Analysis")
        
        # Process for AI
        img_array = np.array(image)
        tensor = transform(img_array).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            output = model(tensor)
            _, pred = torch.max(output, 1)
            label = "DEFECT" if pred.item() == 1 else "CLEAN"
            prob = torch.nn.functional.softmax(output, dim=1)[0][pred.item()] * 100

        # Display Result
        if label == "DEFECT":
            st.error(f"**RESULT: {label}**")
        else:
            st.success(f"**RESULT: {label}**")
            
        st.metric("Confidence", f"{prob:.2f}%")