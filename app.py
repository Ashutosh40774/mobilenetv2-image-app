import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
st.set_page_config(
    page_title="🌼 Flower Classifier",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2909/2909834.png", width=80)
    st.title(" Flower Classifier")
    st.markdown("""
    A custom-trained deep learning model that classifies:
    - 🌼 5 flower types  
    - 🚫 Non-flower images  
    """)
    st.markdown("Made with ❤️ using **TensorFlow** and **Streamlit**")
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips', 'non_flower']
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("flower_nonflower_model.h5")

model = load_model()
IMAGE_SIZE = (180, 180)
st.markdown("## 📷 Upload a Flower Image")
uploaded_file = st.file_uploader(
    "Upload image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='📸 Uploaded Image', use_column_width=True)
    img = image.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    if st.button("🔍 Predict"):
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        if predicted_class == "non_flower":
            st.warning(f"⚠️ This doesn't appear to be a flower. ({confidence:.2f}% confidence)")
        else:
            st.success(f"🌸 **Prediction:** {predicted_class.capitalize()} ({confidence:.2f}% confidence)")
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Created by Ashutosh • Powered by TensorFlow & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
