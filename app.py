import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np


# Function to load and prepare the image
def load_and_prepare_image(uploaded_file, target_size=(256, 256)):

    # Convert the file to bytes for OpenCV to read
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Resize the image
    img_resized = cv2.resize(img, target_size)

    # Normalize the pixel values if your model expects pixel values to be in the range [0, 1]
    img_resized = img_resized / 255.0

    # Add a batch dimension
    img_batch = np.expand_dims(img_resized, axis=0)
    return img_batch


# Function to predict the class of the image


def classify_image(model, img):
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return class_index


# Load the model (adjust the path to your model's location)
MODEL_PATH = (
    "/Users/baboo/Desktop/Portfolio/ImageClassification/models/imageclassifier.h5"
)
model = load_model(MODEL_PATH)

# Streamlit app
st.title("Emotion Detector: Real-Time Image Classification of Human Emotions")
st.write(
    "This app classifies images into categories: Happy, Sad, Angry, Fear, Surprise."
)

# Upload file

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    img = load_and_prepare_image(uploaded_file)

    # Predict the class
    class_index = classify_image(model, img)

    # Define the class labels
    labels = ["Happy", "Sad", "Angry", "Fear", "Surprise"]
    prediction = labels[class_index]

    st.write(f"Prediction: {prediction}")
