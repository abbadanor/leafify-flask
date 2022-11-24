from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os
import random
import string

species = ['Ulmus carpinifolia', 'Acer', 'Salix aurita', 'Quercus', 'Alnus incana', 'Betula pubescens', 'Salix alba var. Sericea', 'Populus tremula', 'Ulmus glabra', 'Sorbus aucuparia', 'Salix sinerea', 'Populus', 'Tilia', 'Sorbus intermedia', 'Fagus silvatica']

IMG_HEIGHT = 224
IMG_WIDTH = 224
TF_MODEL_FILE_PATH = './leaf-india.tflite' # The default path to the saved TensorFlow Lite model

UPLOAD_FOLDER = './images/'
ALLOWED_EXTENSIONS = {'png', 'jpg'}

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(path, visualize=False):
  img = tf.keras.utils.load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')

  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
  classify_lite = interpreter.get_signature_runner('serving_default')

  #score = model.predict(img_array, verbose = 0)
  score = classify_lite(conv2d_2_input=img_array)['activation_1']
  return {'species': species[np.argmax(score)], 'confidence': 100 * np.max(score)}

@app.route('/api', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            if os.path.isfile(filepath):
                os.remove(filepath)
            else:
                return
            return jsonify(
                species=prediction['species'],
                confidence=prediction['confidence']
            )
    elif request.method == 'GET':
        response = {
            'response': 'Hello world!'
        }
        return jsonify(response)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
