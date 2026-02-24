import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("model/deepfake_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    pred = model.predict(img)[0][0]
    confidence = round(float(pred)*100,2)

    if pred > 0.5:
        return "FAKE", confidence
    else:
        return "REAL", 100-confidence