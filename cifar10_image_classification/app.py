import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_cifar10_model():
    model = load_model("models/cifar10_cnn.h5")
    return model

model = load_cifar10_model()

# CIFAR-10 Class Labels
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚀 CIFAR-10 Image Classification")
st.write("Upload an image and let the CNN predict its class!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((32, 32))  # CIFAR-10 expects 32x32
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype("float32") / 255.0

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    # Show result
    st.write(f"### 🏷 Prediction: **{class_names[predicted_class]}**")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")
