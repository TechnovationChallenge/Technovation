from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/compostnet_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read labels from file
with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/')
def home():
    return '✅ CompostNet TFLite API is live!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))  # match training input size
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    top_idx = int(np.argmax(output_data))
    prediction = labels[top_idx]
    confidence = float(output_data[top_idx])

    return jsonify({
        'prediction': prediction,
        'confidence': round(confidence, 4)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)