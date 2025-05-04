# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import base64
# import os

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # Load TFLite model
# try:
#     interpreter = tf.lite.Interpreter(model_path="model/compostnet_model.tflite")
#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     print("✅ Model loaded.")
# except Exception as e:
#     print("❌ Model load error:", e)
#     interpreter = None

# # Load labels
# try:
#     with open("model/labels.txt", "r") as f:
#         labels = [line.strip() for line in f.readlines()]
#     print("✅ Labels loaded.")
# except Exception as e:
#     print("❌ Labels file error:", e)
#     labels = []

# @app.route('/')
# def home():
#     return '✅ CompostNet API is live!'

# @app.route('/healthcheck', methods=['GET'])
# def healthcheck():
#     return 'OK', 200

# @app.route('/predict', methods=['POST'])
# def predict():
#     if interpreter is None or not labels:
#         return jsonify({'error': 'Model or labels not loaded'}), 500

#     try:
#         data = request.get_json()
#         if not data or 'image' not in data:
#             return jsonify({'error': 'No image provided'}), 400

#         # Decode Base64 image
#         image_data = base64.b64decode(data['image'])
#         img = Image.open(io.BytesIO(image_data)).convert('RGB')
#         img = img.resize((224, 224))
#         img_array = np.array(img, dtype=np.float32) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # Run prediction
#         interpreter.set_tensor(input_details[0]['index'], img_array)
#         interpreter.invoke()
#         output_data = interpreter.get_tensor(output_details[0]['index'])[0]

#         top_idx = int(np.argmax(output_data))
#         prediction = labels[top_idx]
#         confidence = float(output_data[top_idx])

#         return jsonify({
#             'prediction': prediction,
#             'confidence': round(confidence, 4)
#         })

#     except Exception as e:
#         print("❌ Prediction error:", e)
#         return jsonify({'error': 'Internal prediction error'}), 500

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port)

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="model/compostnet_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model loaded.")
except Exception as e:
    print("❌ Model load error:", e)
    interpreter = None

# Load labels
try:
    with open("model/labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    print("✅ Labels loaded.")
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

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Read image from uploaded file
        file = request.files['file']
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run prediction
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
        print("❌ Prediction error:", e)
        return jsonify({'error': 'Internal prediction error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)