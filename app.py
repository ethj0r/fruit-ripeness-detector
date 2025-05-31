import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load model
model = tf.keras.models.load_model("models/banana_ripeness_cnn.h5")
class_names = ['matang', 'mentah', 'setengah-matang', 'terlalu-matang']
IMG_SIZE = (180, 180)

# Transformer untuk webcam
class BananaDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Ambil frame dari webcam
        resized = cv2.resize(img, IMG_SIZE)
        normalized = resized / 255.0
        input_tensor = tf.expand_dims(normalized, axis=0)

        prediction = model.predict(input_tensor)
        label_index = np.argmax(prediction)
        label = class_names[label_index]
        confidence = 100 * prediction[0][label_index]

        # Tampilkan label & confidence di atas frame
        text = f"{label} ({confidence:.1f}%)"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# Streamlit UI
st.title("🍌 Deteksi Kematangan Pisang dari Webcam")
st.caption("Arahkan pisang ke kamera. Model akan memprediksi secara real-time.")
st.write("🧪 Python version:", sys.version)

webrtc_streamer(key="banana", video_transformer_factory=BananaDetector)