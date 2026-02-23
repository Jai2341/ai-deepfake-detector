import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model/deepfake_model.h5")

IMG_SIZE = 224

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(img_path):

    img = preprocess(img_path)

    prediction = model.predict(img)[0][0]

    # confidence %
    confidence = round(float(prediction) * 100, 2)

    if prediction > 0.5:
        label = "FAKE"
    else:
        label = "REAL"

    # ⭐ IMPORTANT → return TWO VALUES
    return label, confidence