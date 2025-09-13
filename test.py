import os
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 224
MODEL_PATH = 'BoneFracture_densenet_model.keras'
CLASS_NAMES = ['Fractured', 'Not Fractured']

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception:
    print(f"Fatal: Model file not found at '{MODEL_PATH}'. Please ensure the path is correct.")
    model = None

def predict_fracture(input_image: Image.Image) -> dict:
    if model is None:
        return {"Error": "Model not loaded."}
    img = input_image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    prediction_probs = model.predict(img_batch)[0]
    return {CLASS_NAMES[i]: float(prediction_probs[i]) for i in range(len(CLASS_NAMES))}

iface = gr.Interface(
    fn=predict_fracture,
    inputs=gr.Image(type="pil", label="Upload an X-ray Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction Results"),
    title="BoneVision.ai",
    description="An AI model to detect bone fractures in X-ray images. Upload an image to see the prediction Model.",
)

iface.launch(share=False, debug=True)