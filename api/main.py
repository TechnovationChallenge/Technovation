from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Set maximum request size to 5MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="model/compostnet_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print("❌ Model load error:", e)
    interpreter = None

# Load labels
try:
    with open("model/labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
except Exception as e:
    print("❌ Labels file error:", e)
    labels = []

@app.route('/')
def home():
    return '✅ CompostNet API is live!'

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return 'OK', 200

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None or not labels:
        return jsonify({'error': 'Model or labels not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        try:
            img = Image.open(file.stream).convert('RGB')
        except Exception:
            return jsonify({'error': 'File is not a valid image'}), 400

        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        top_idx = int(np.argmax(output_data))
        prediction = labels[top_idx]
        confidence = float(output_data[top_idx])

        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({'error': 'Internal prediction error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # required for Render/Railway
    app.run(host='0.0.0.0', port=port)