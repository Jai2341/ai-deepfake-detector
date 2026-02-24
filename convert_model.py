import tensorflow as tf

print("Loading old model...")
old_model = tf.keras.models.load_model("model/deepfake_model.h5", compile=False)

print("Re-saving in new Keras format...")
old_model.save("model/deepfake_model.keras")

print("✅ Model successfully converted!")