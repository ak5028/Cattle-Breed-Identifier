
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import time

# Initialize Flask app
app = Flask(__name__)

# --- Model & Class Names Loading ---
MODEL_PATH = "effnetv2s_best.keras"
CLASS_NAMES_PATH = "class_names.txt"

# Load the trained model
# We use a try-except block to give a clear error if the model is missing
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except IOError:
    # A more descriptive error for the user
    print(f"FATAL: Model file not found at {MODEL_PATH}. Please ensure the model is in the correct directory.")
    # Exit if the model can't be loaded, as the app is useless without it.
    exit()

# Load class names
# Similar error handling for the class names file
try:
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except IOError:
    print(f"FATAL: Class names file not found at {CLASS_NAMES_PATH}. Please ensure it is present.")
    exit()

# --- Image Preprocessing ---
IMG_SIZE = 300 # Must match the training configuration

def preprocess_image(image_bytes):
    """
    Preprocesses the uploaded image to be compatible with the model.
    1. Decodes image bytes
    2. Resizes to IMG_SIZE
    3. Converts to NumPy array
    4. Adds a batch dimension
    5. Applies EfficientNetV2 preprocessing
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    
    # Use the specific preprocessing function for the model
    preprocessed_img = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    return preprocessed_img

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    # The template will be created in the next step
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image prediction from either a file upload or a base64 JSON payload.
    """
    started = time.time()

    try:
        img_bytes = None
        # Case 1: JSON payload from swoosh-home
        if request.is_json:
            data = request.get_json()
            if 'imageBase64' not in data:
                return jsonify({'error': 'imageBase64 not found in JSON payload'}), 400
            # The base64 string may have a prefix like "data:image/jpeg;base64,"
            base64_str = data['imageBase64'].split(',')[-1]
            img_bytes = base64.b64decode(base64_str)

        # Case 2: File upload from the brutalist UI
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected for uploading'}), 400
            img_bytes = file.read()

        # If neither case is met, return an error
        if img_bytes is None:
            return jsonify({'error': 'No image data provided'}), 400

        # Preprocess the image and get predictions
        preprocessed_img = preprocess_image(img_bytes)
        predictions = model.predict(preprocessed_img)
        scores = tf.nn.softmax(predictions[0])

        # Format response for swoosh-home (and our simple UI can adapt)
        top_k = 5 # swoosh-home default
        if request.is_json and 'topK' in request.get_json():
            top_k = request.get_json()['topK']

        # Get top k predictions
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        response_predictions = []
        for i in top_indices:
            response_predictions.append({
                'label': class_names[i],
                'confidence': float(scores[i])
            })

        latency = (time.time() - started) * 1000

        # This response format is compatible with swoosh-home's frontend
        return jsonify({
            'model': 'effnetv2s_best',
            'latencyMs': latency,
            'predictions': response_predictions
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Runs the app on localhost, port 5000
    app.run(debug=True)
