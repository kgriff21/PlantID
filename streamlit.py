import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Define species classes
SPECIES_CLASSES = ["Alocasia", "Anthurium", "Calathea", "Monstera", "Pothos"]

# Load the model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("saved_model_freeze.keras")

model = load_model()

# Streamlit UI
st.title("Plant Species Classifier 🌿")
st.write("Upload an image of a plant leaf to classify its genus.  \n"
    "This model can identify the following genera: Alocasia, Anthurium, Calathea, Monstera and Pothos")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_resized = image.resize((256, 256))  # EfficientNet typically uses 224x224
    img_array = np.array(img_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict
    predictions = model.predict(img_array)[0]  # Get first batch prediction
    
    # Sort predictions
    sorted_indices = np.argsort(predictions)[::-1]  # Sort in descending order
    sorted_species = [SPECIES_CLASSES[i].replace('_images', '').replace('_', ' ').title() for i in sorted_indices]
    sorted_scores = [predictions[i] for i in sorted_indices]
    
    # Display results
    st.write("### Predictions:")
    for species, score in zip(sorted_species, sorted_scores):
        st.write(f"**{species}:** {score:.4f}")
