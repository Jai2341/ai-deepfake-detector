import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# load NEW keras model
model = tf.keras.models.load_model("deepfake_model.keras")

IMG_SIZE = 224

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img)[0][0]
    confidence = round(float(prediction) * 100, 2)

    if prediction > 0.5:
        return "FAKE", confidence
    else:
        return "REAL", 100 - confidence