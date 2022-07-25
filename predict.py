import cv2
import tensorflow as tf
import numpy as np
CATEGORIES = ['cats', 'dogs']


def image_pre(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (100, 100))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 100, 100, 1)
    return new_arr

def predict_image(img_path,):
    model = tf.keras.models.load_model('Final_model')
    prediction = model.predict([image_pre(img_path)])
    return((CATEGORIES[prediction.argmax()]))
