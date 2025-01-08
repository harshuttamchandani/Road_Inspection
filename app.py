import os
import shutil
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st

# YOLO Model Loading
model = YOLO('best.pt')

def bgr2rgb(image):
    return image[:, :, ::-1]

def main():
    with open("styles.css", "r") as source_style:
        st.markdown(f"<style>{source_style.read()}</style>", unsafe_allow_html=True)
        
    st.title("AI Road Inspection System")

    st.subheader('Upload Image to Predict Defects')
    upload_img_file = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])
    
    if upload_img_file is not None:
        file_bytes = np.asarray(bytearray(upload_img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        prediction = model.predict(img)
        res_plotted = prediction[0].plot()
        image_pil = Image.fromarray(res_plotted)
        st.image(image_pil, caption='Predicted Image', use_container_width=True)

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
