import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224

# ⭐ load model safely (old keras compatibility)
model = tf.keras.models.load_model(
    "model/deepfake_model.h5",
    compile=False
)

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img_path):
    img = preprocess(img_path)
    prediction = model.predict(img)[0][0]

    confidence = round(float(prediction) * 100, 2)

    if prediction > 0.5:
        label = "FAKE"
    else:
        label = "REAL"

    return label, confidence
