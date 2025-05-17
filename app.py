
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO

IMG_SIZE = 128

# --- Preprocessing ---
def generate_ela_image(image, quality=90):
    temp_filename = 'temp_ela.jpg'
    cv2.imwrite(temp_filename, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    ela_img = cv2.imread(temp_filename)
    ela_image = cv2.absdiff(image, ela_img)
    ela_image = cv2.normalize(ela_image.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return ela_image

def generate_fft_image(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    fft_image = cv2.cvtColor(magnitude_spectrum.astype('uint8'), cv2.COLOR_GRAY2BGR)
    return fft_image.astype('float32') / 255.0

def preprocess_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_np = np.array(image.convert('RGB'))  # Ensure 3 channels
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    rgb = image_cv.astype('float32') / 255.0
    ela = generate_ela_image(image_cv)
    fft = generate_fft_image(image_cv)

    return np.expand_dims(rgb, axis=0), np.expand_dims(ela, axis=0), np.expand_dims(fft, axis=0)

# --- Grad-CAM ---
def get_grad_cam(model, inputs, class_index=0, layer_name='rgb_conv3'):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-6)
    return heatmap.numpy()

def overlay_gradcam(image, heatmap):
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_img_rgb = cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_BGR2RGB)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(original_img_rgb, 0.6, heatmap_color_rgb, 0.4, 0)
    return superimposed

# --- Load model once ---
@st.cache_resource
def load_my_model():
    return load_model(r"C:\Users\ASUS\Desktop\main_model\main_model\forensic_model.h5")

model = load_my_model()

# --- Streamlit App ---
st.title("Deepfake Image Detector")
st.markdown("Upload an image to check whether it's Real or Fake.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    rgb, ela, fft = preprocess_image(image)

    # Prediction
    prediction = model.predict([rgb, ela, fft])[0][0]
    label = 'Fake' if prediction > 0.5 else 'Real'
    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.markdown(f"### 🔍 Confidence: `{prediction:.2f}`")

    # Grad-CAM
       # Grad-CAM
    heatmap = get_grad_cam(model, [rgb, ela, fft], class_index=0, layer_name='rgb_conv3')
    gradcam_image = overlay_gradcam(rgb[0], heatmap)

    # ✅ Save Grad-CAM to disk so Flask server can serve it
    save_path = os.path.join(os.getcwd(), "gradcam.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(gradcam_image, cv2.COLOR_RGB2BGR))
    st.markdown("### 🔥 Grad-CAM Visualization")
    st.image(gradcam_image, use_column_width=True)
    st.success(f"Grad-CAM saved to: {save_path}")



