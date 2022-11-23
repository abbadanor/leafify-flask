from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import os

class_names = ['Ulmus minor', 'Acer', 'Salix aurita', 'Quercus', 'Alnus incana', 'Betula pubescens', 'Salix arlba var. Sericea', 'Populus tremula', 'Ulmus glabra', 'Sorbus aucuparia', 'Salix cinerea', 'Populus', 'Tilia', 'Sorbus intermedia', 'Fagus Silvatica']

img_height = 224
img_width = 224
TF_MODEL_FILE_PATH = './leaf.tflite' # The default path to the saved TensorFlow Lite model

UPLOAD_FOLDER = './images/'
ALLOWED_EXTENSIONS = {'png'}

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
        image = Image.open(image_path)
        new_image = image.resize((img_width,img_height))

        background = Image.new('RGBA', new_image.size, (255,255,255))
        alpha_composite = Image.alpha_composite(background, new_image)
        alpha_composite_rgb = alpha_composite.convert('RGB')

        np_image = np.asarray(alpha_composite_rgb)

        print(np_image.shape)

        min = np_image.min()
        max = np_image.max()

        # normalize to the range 0-1
        np_image = np_image.astype('float32')
        np_image -= min
        np_image /= (max - min)

        return np_image

def softmax_stable(x):
        return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

def predict_image(path):
        img_array = process_image(path)
        img_array = np.expand_dims(img_array, axis=0)

        interpreter = tflite.Interpreter(model_path=TF_MODEL_FILE_PATH)

        classify_lite = interpreter.get_signature_runner('serving_default')

        predictions_lite = classify_lite(sequential_1_input=img_array)['outputs']

        score_lite = softmax_stable(predictions_lite)

        return {'leaf_species': class_names[np.argmax(score_lite)], 'confidence':  100 * np.max(score_lite)}

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
            return jsonify(
                leaf_species=prediction['leaf_species'],
                confidence=prediction['confidence']
            )
    elif request.method == 'GET':
        response = {
            'response': 'Hello world!'
        }
        return jsonify(response)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
