# load dependencies
import cv2
from tensorflow import keras
import tensorflow as tf
from flask_cors import CORS
from flask import Flask, request, jsonify
from io import BytesIO
import json
import base64
import requests
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# TF Serving Assets
HEADERS = {'content-type': 'application/json'}
MODEL1_API_URL = 'http://localhost:8501/v1/models/fashion_model_serving/versions/1:predict'
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Instantiate Flask App
app = Flask(__name__)
CORS(app)

# Image resizing utils


def resize_image_array(img, img_size_dims):
    img = cv2.resize(img, dsize=img_size_dims,
                     interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float32)
    return img


# Liveness test
@app.route('/apparel_classifier/api/v1/liveness', methods=['GET', 'POST'])
def liveness():
    return 'API Live!'

# Model 1 inference endpoint
@app.route('/apparel_classifier/api/v1/model1_predict', methods=['POST'])
def image_classifier_model2():
    img = np.array([keras.preprocessing.image.img_to_array(
        keras.preprocessing.image.load_img(BytesIO(base64.b64decode(request.form['b64_img'])),
                                           target_size=(28, 28)).convert('1')) / 255.])

    data = json.dumps({"signature_name": "serving_default",
                       "instances": img.tolist()})

    json_response = requests.post(MODEL1_API_URL, data=data, headers=HEADERS)
    prediction = json.loads(json_response.text)['predictions']
    prediction = np.argmax(np.array(prediction), axis=1)[0]
    prediction = CLASS_NAMES[prediction]

    return jsonify({'apparel_type': prediction})


# running REST interface, port=5000 for direct test
# use debug=True when debugging
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
