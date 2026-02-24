import os

# ⭐⭐ CRITICAL FIX — MUST BE BEFORE TENSORFLOW IMPORT ⭐⭐
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ---------------------------------
# Model path
# ---------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "deepfake_model.h5")

print("📦 Loading model from:", MODEL_PATH)

# ⭐ compile=False avoids optimizer errors
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

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