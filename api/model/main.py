from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('model/compostnet_model.h5')

# Define class labels in the correct order
labels = [
    "cardboard",
    "compost",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

@app.route('/')
def index():
    return '✅ CompostNet Server is running.'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    predictions = model.predict(img_array)[0]
    predicted_label = labels[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return jsonify({
        'prediction': predicted_label,
        'confidence': round(confidence, 4)
    })