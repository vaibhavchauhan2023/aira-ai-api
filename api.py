import os
import requests
import base64
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 1. SETUP FLASK APP ---
app = Flask(__name__)
CORS(app)  # Allow our React frontend to call this

# --- 2. DEFINE SIAMESE MODEL CLASSES ---
# This is the L1 Distance layer from your notebook.
# We MUST redefine it so TensorFlow knows what it is when we load the model.
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
       
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# --- 3. DEFINE PREPROCESSING ---
# This is the preprocessing logic from your notebook
def preprocess_image(image):
    # Resize image to 100x100 pixels
    img_resized = cv2.resize(image, (100, 100))
    # Normalize pixel values to be between 0 and 1
    img_normalized = img_resized / 255.0
    # Add a batch dimension (so its shape is 1, 100, 100, 3)
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded

# --- 4. MODEL DOWNLOAD & LOAD LOGIC ---
MODEL_URL = "https://github.com/vaibhavchauhan2023/aira-ai-api/releases/download/v1.0/siamesemodel.h5"
MODEL_PATH = "siamesemodel.h5"
SIAMESE_MODEL = None

def download_model():
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        print("[SERVER] Model already exists. Skipping download.")
        return

    print(f"[SERVER] Downloading model from {MODEL_URL}...")
    try:
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("[SERVER] Model downloaded successfully.")
    except Exception as e:
        print(f"[SERVER] Error downloading model: {e}")

def load_siamese_model():
    global SIAMESE_MODEL
    try:
        # We must pass 'L1Dist' as a custom object
        SIAMESE_MODEL = load_model(
            MODEL_PATH, 
            custom_objects={'L1Dist': L1Dist}
        )
        print("[SERVER] AI Model loaded successfully!")
    except Exception as e:
        print(f"[SERVER] Error loading model: {e}")

# --- 5. THE VERIFICATION API ENDPOINT ---
@app.route('/api/verify-face', methods=['POST'])
def verify_face():
    global SIAMESE_MODEL
    if SIAMESE_MODEL is None:
        return jsonify({'success': False, 'message': 'AI Model is not loaded on server.'}), 500

    try:
        data = request.json
        image_data_url = data.get('image') # Base64 image from React
        user_id = data.get('userId')       # Student ID (e.g., 'e23cseu01183')

        if not image_data_url or not user_id:
            return jsonify({'success': False, 'message': 'Missing image or user ID.'}), 400

        # --- 1. Decode the Webcam (Test) Image ---
        image_data = base64.b64decode(image_data_url.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        webcam_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_webcam_img = preprocess_image(webcam_img)

        # --- 2. Load the Anchor Image from our database ---
        # This is why you must rename your 19 photos!
        anchor_path = os.path.join('data', 'anchor', f'{user_id}.jpg')
        
        if not os.path.exists(anchor_path):
            return jsonify({'success': False, 'message': f'No verification image on file for user {user_id}.'}), 400
            
        anchor_img = cv2.imread(anchor_path)
        processed_anchor_img = preprocess_image(anchor_img)

        # --- 3. Make the Prediction ---
        prediction = SIAMESE_MODEL.predict([processed_anchor_img, processed_webcam_img])
        similarity_score = prediction[0][0] 
        
        # --- 4. (USER) TODO: TUNE THIS THRESHOLD ---
        # This is the most important value. You must test this!
        # 0.5 is the default from the tutorial.
        # If it's too high (e.g., 0.8), it will always fail.
        # If it's too low (e.g., 0.2), it will let anyone in.
        VERIFICATION_THRESHOLD = 0.5 
        
        print(f"[SERVER] Verification for {user_id}: Score = {similarity_score} (Threshold: {VERIFICATION_THRESHOLD})")
        
        if similarity_score > VERIFICATION_THRESHOLD:
            return jsonify({'success': True, 'message': 'Face Verified.'})
        else:
            return jsonify({'success': False, 'message': 'Face Not Recognized. Please try again.'})

    except Exception as e:
        print(f"[SERVER] An error occurred during face verification: {e}")
        return jsonify({'success': False, 'message': f'An error occurred: {e}'}), 500

# --- 6. START THE APP ---
if __name__ == '__main__':
    # When we run this file, first download, then load, then start the server
    download_model()
    load_siamese_model()
    # Flask runs on port 5000 by default, which is what Render expects
    app.run(host='0.0.0.0', port=5000)