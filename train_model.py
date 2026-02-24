import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# 🔥 Use absolute safe path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_model.keras")

# Load trained model (NEW FORMAT)
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img)[0][0]

    confidence = round(float(prediction) * 100, 2)

    if prediction > 0.5:
        return "FAKE", confidence
    else:
        return "REAL", round(100 - confidence, 2)