import tensorflow as tf
import gradio as gr
from PIL import Image
import numpy as np

# Load your pre-trained model
model_path = 'D:/ARCHIT/chatbot/basic-model/brain_tumor_model/brain_tumors_model.h5'
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Check if the image is a numpy array (Gradio default format) and convert it to a PIL image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Ensure the image has three channels (RGB)
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Resize as per the model’s input requirement
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension for model compatibility
    return image

# Classification function for the chatbot
def classify_image(image):
    processed_image = preprocess_image(image)
    try:
        prediction = model.predict(processed_image)
        # Adjust the condition based on your model’s output format
        result = "Tumor detected" if prediction[0][0] > 0.5 else "No tumor detected"
    except Exception as e:
        print("Error during prediction:", e)
        result = "An error occurred during prediction. Please check the input and model."
    return result

# Set up Gradio interface
interface = gr.Interface(
    fn=classify_image,  # Function that processes the image and returns a result
    inputs="image",     # Input type (image)
    outputs="text",     # Output type (text)
    title="Brain Tumor Detection Chatbot",
    description="Upload an MRI scan to check for the presence of a brain tumor."
)

# Launch the chatbot
interface.launch()
print("Expected model input shape:", model.input_shape)

