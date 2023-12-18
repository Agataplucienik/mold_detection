import tensorflow

import numpy as np
import os
from PIL import Image
from tensorflow import keras





def preprocess_image(img):
    '''
    Preprocessing imaging using PIL.
    1. Resizing it to (224, 224) without changing aspect ratio
    2. Converting it into a numpy array
    3. Padding for zeros in case the image is not (224,224)
    '''
    size = (224, 224)
    # 1. Resizing it to (224, 224) without changing aspect ratio
    img.thumbnail(size, Image.LANCZOS)
    # 2. Converting it into a numpy array
    img_array = np.asarray(img)
    # 3. Padding for zeros in case the image is not (224,224)
    h = size[0] - img_array.shape[0] # Horizontal size (rows)
    v = size[1] - img_array.shape[1] # Vertical size (cols)
    img_array = np.pad(
        img_array, [(0, h), (0,v), (0,0)],
        mode='constant', constant_values=0)
    return img_array


def predict(data):
    path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(path, 'fruit_veg_classifier.h5')
    # new_model = tf.keras.models.load_model('C:\\Users\\agata\\Downloads\\model.keras')
    new_model = keras.models.load_model(model_path)
    # weights = new_model.get_weights()
    # img = image.load_img(data, target_size=(224, 224))  # Adjusting the target size based on model's input size
    # img_array = image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    data_ = preprocess_image(data)
    img_array = np.expand_dims(data_, axis=0)
    predictions=new_model.predict(img_array)[0]
    idx_prediction = np.where(predictions == predictions.max())[0][0]
    prediction = prediction_dictionary(idx_prediction)
    return prediction


def prediction_dictionary(idx):
    classes = ['apple',
    'banana',
    'beetroot',
    'bell pepper',
    'cabbage',
    'capsicum',
    'carrot',
    'cauliflower',
    'chilli pepper',
    'corn',
    'cucumber',
    'eggplant',
    'garlic',
    'ginger',
    'grapes',
    'jalepeno',
    'kiwi',
    'lemon',
    'lettuce',
    'mango',
    'onion',
    'orange',
    'paprika',
    'pear',
    'peas',
    'pineapple',
    'pomegranate',
    'potato',
    'raddish',
    'soy beans',
    'spinach',
    'sweetcorn',
    'sweetpotato',
    'tomato',
    'turnip',
    'watermelon']

    return classes[idx]
