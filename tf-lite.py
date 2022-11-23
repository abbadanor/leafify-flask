from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

class_names = ['01 - Ulmus minor', '02 - Acer', '03 - Salix aurita', '04 - Quercus', '05 - Alnus incana', '06 - Betula pubescens', '07 - Salix arlba var. Sericea', '08 - Populus tremula', '09 - Ulmus glabra', '10 - Sorbus aucuparia', '11 - Salix cinerea', '12 - Populus', '13 - Tilia', '14 - Sorbus intermedia', '15 - Fagus Silvatica']

img_height = 224
img_width = 224
TF_MODEL_FILE_PATH = './leaf.tflite' # The default path to the saved TensorFlow Lite model

def process_image(image_path):
        image = Image.open(image_path)
        new_image = image.resize((img_width,img_height))
        np_image = np.asarray(new_image)

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

        return "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
