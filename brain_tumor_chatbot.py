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
    elif not isinstance(image, Image.Image):
        raise ValueError("Input must be a numpy array or a PIL Image.")
    
    # Ensure the image has three channels (RGB)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize((64, 64))  # Resize to match the model’s input requirement
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension for model compatibility
    return image

# Classification function for the chatbot
def classify_image(image):
    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        print("Raw prediction:", prediction)  # Output the raw prediction for debugging
        class_index = np.argmax(prediction)  # Get the index of the class with the highest probability
        
        if class_index == 1:  # Assuming index 1 means "No Tumor"
            result = "No tumor detected"
        elif class_index == 0:  # Assuming index 0 means "Benign Tumor"
            result = "Benign tumor detected"
        elif class_index == 2:  # Assuming index 2 means "Malignant Tumor"
            result = "Malignant tumor detected"
        else:  # Assuming index 3 means "Other"
            result = "possiblity of pitutary tumor"
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
# Inside the classify_image function



# Launch the chatbot
interface.launch()
print("Expected model input shape:", model.input_shape)
