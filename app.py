import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Load the MobileNetV2 model once
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_model()
IMAGE_SIZE = (224, 224)

# Function to download and preprocess image
def download_and_preprocess_image(img_url):
    try:
        response = requests.get(img_url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return img, img_array
    except Exception as e:
        st.error(f"Error downloading or processing the image: {e}")
        return None, None

# Streamlit app UI
st.title("🌼 Image Classifier using MobileNetV2")
st.write("Paste an image URL below and click **Predict** to classify the object in the image.")

img_url = st.text_input("Enter Image URL")

if st.button("Predict"):
    if img_url.strip() == "":
        st.warning("Please enter a valid image URL.")
    else:
        img, processed_img = download_and_preprocess_image(img_url)
        if processed_img is not None:
            preds = model.predict(processed_img)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]
            label = decoded[1]
            confidence = decoded[2] * 100

            st.image(img, caption="Input Image", use_column_width=True)
            st.success(f"🎯 Prediction: **{label}** ({confidence:.2f}% confidence)")
