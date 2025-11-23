import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model
@st.cache_resource
@st.cache_resource
def load_model():
    st.info("Loading model... Please wait...")
    model_path = "../models/skin_cancer_model.h5"
    try:
        # First try loading with the standard method
        st.info("Attempting to load model with standard method...")
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully with standard method!")
        return model
    except (TypeError, ValueError) as e:
        st.warning(f"Standard loading failed: {str(e)}")
        st.info("Trying alternative loading method...")
        try:
            # Try loading just the weights
            st.info("Building model architecture...")
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                
                tf.keras.layers.Dense(7, activation='softmax')
            ])
            
            st.info("Loading model weights...")
            # Try loading weights with and without .h5 extension
            try:
                model.load_weights(model_path)
            except:
                model.load_weights(model_path + '.h5')
                
            st.success("Model loaded successfully with custom architecture!")
            return model
            
        except Exception as e:
            st.error(f"Failed to load the model. Error: {str(e)}")
            st.error("Please ensure the model file exists at: " + os.path.abspath(model_path))
            st.stop()
model = load_model()

# Class names and descriptions
CLASS_NAMES = ["Actinic Keratoses (akiec)", 
               "Basal Cell Carcinoma (bcc)", 
               "Benign Keratosis (bkl)", 
               "Dermatofibroma (df)", 
               "Melanoma (mel)", 
               "Melanocytic Nevi (nv)", 
               "Vascular Lesions (vasc)"]

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic Keratoses are rough, scaly patches caused by sun damage. They are considered precancerous.",
    "bcc": "Basal Cell Carcinoma is the most common type of skin cancer, usually appearing as a waxy bump.",
    "bkl": "Benign Keratosis are non-cancerous skin growths that can resemble warts or moles.",
    "df": "Dermatofibroma are harmless hard bumps that form in the skin's dermis layer.",
    "mel": "Melanoma is the most serious type of skin cancer that can develop in existing moles or appear as new spots.",
    "nv": "Melanocytic Nevi are common moles that are usually harmless but should be monitored for changes.",
    "vasc": "Vascular Lesions are blood vessel-related skin abnormalities that can be benign or malignant."
}

def preprocess_image(image, target_size=(64, 64)):
    """Preprocess the image for prediction."""
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    if len(img_array.shape) == 2:  # Convert grayscale to RGB
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:  # Remove alpha channel if present
        img_array = img_array[:, :, :3]
    return np.expand_dims(img_array, axis=0)

def predict(image):
    """Make prediction on the input image."""
    preprocessed_img = preprocess_image(image)
    predictions = model.predict(preprocessed_img, verbose=0)
    return predictions[0]

def plot_prediction(predictions):
    """Create an interactive bar plot of predictions."""
    fig = px.bar(
        x=CLASS_NAMES,
        y=predictions * 100,
        labels={'x': 'Class', 'y': 'Confidence (%)'},
        title="Prediction Probabilities",
        color=CLASS_NAMES,
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        showlegend=False
    )
    return fig

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {font-size: 30px; font-weight: 700; color: #1f77b4;}
        .sub-header {font-size: 18px; color: #2c3e50; margin-bottom: 20px;}
        .upload-box {border: 2px dashed #1f77b4; border-radius: 10px; padding: 20px; text-align: center; margin: 20px 0;}
        .prediction-box {border-left: 5px solid #1f77b4; padding: 15px; margin: 15px 0; background-color: #f8f9fa;}
        .stButton>button {background-color: #1f77b4; color: white; border-radius: 5px; padding: 0.5rem 1rem;}
        .stButton>button:hover {background-color: #155a8a;}
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header">🏥 Skin Cancer Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload an image of a skin lesion for analysis</div>', unsafe_allow_html=True)

    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This application uses a deep learning model to classify skin lesions into 7 different categories.
        
        **How to use:**
        1. Upload an image of a skin lesion
        2. The model will analyze the image
        3. View the prediction results and confidence levels
        
        **Note:** This tool is for educational purposes only and should not be used as a substitute for professional medical advice.
        """)
        
        st.markdown("---")
        st.markdown("### Class Descriptions")
        for class_name in CLASS_NAMES:
            with st.expander(class_name):
                st.write(CLASS_DESCRIPTIONS[class_name.split(" ")[-1][1:-1]])

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            with st.spinner('Analyzing the image...'):
                predictions = predict(image)
                predicted_class_idx = np.argmax(predictions)
                predicted_class = CLASS_NAMES[predicted_class_idx]
                confidence = predictions[predicted_class_idx] * 100
            
            with col2:
                st.markdown("### 🎯 Prediction Results")
                
                # Display prediction with confidence
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Most Likely: <span style="color: #1f77b4;">{predicted_class}</span></h3>
                    <h4>Confidence: <span style="color: #1f77b4;">{confidence:.2f}%</span></h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Show prediction distribution
                st.plotly_chart(plot_prediction(predictions), use_container_width=True)
            
            # Show detailed class probabilities
            with st.expander("View Detailed Probabilities"):
                st.write("### Detailed Probabilities")
                for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
                    st.progress(float(prob), text=f"{class_name}: {prob*100:.2f}%")
            
            # Disclaimer
            st.warning("""
            **Important:** This tool is for educational and informational purposes only. 
            It is not a substitute for professional medical advice, diagnosis, or treatment. 
            Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
            """)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        # Show sample images if no image is uploaded
        st.markdown("### 🖼️ Example Images")
        st.markdown("Try uploading an image or check out these examples:")
        
        # Create a grid of example images
        example_dir = "../data/HAM10000/HAM10000_images_part_1"
        example_images = [os.path.join(example_dir, f) for f in os.listdir(example_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:6]
        
        if example_images:
            cols = st.columns(3)
            for idx, img_path in enumerate(example_images[:6]):
                with cols[idx % 3]:
                    st.image(img_path, use_column_width=True, caption=f"Example {idx+1}")

if __name__ == "__main__":
    main()
