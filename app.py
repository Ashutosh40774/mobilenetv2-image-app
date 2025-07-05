import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("flower_model.h5")

model = load_model()
IMAGE_SIZE = (180, 180)
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
def preprocess_uploaded_image(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        img = img.convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)
        return img, img_array
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None, None
st.title("🌼 Flower Classifier (Custom Model)")
st.write("Upload a flower image and click **Predict** to classify it.")

uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        img, processed_img = preprocess_uploaded_image(uploaded_file)
        if processed_img is not None:
            preds = model.predict(processed_img)
            predicted_class = class_names[np.argmax(preds)]
            confidence = np.max(preds) * 100

            st.success(f"🌸 Prediction: **{predicted_class.capitalize()}** ({confidence:.2f}% confidence)")
