import cv2
import numpy as np
import streamlit as st
from typing import Tuple
from pathlib import Path

import tensorflow as tf

from transform.data_load import normalize

def load_model():
    model_ab_path = Path("model_instances", "model", "g_AB")
    model_ba_path = Path("model_instances", "model", "g_BA")

    model_ab = tf.keras.models.load_model(model_ab_path)
    model_ba = tf.keras.models.load_model(model_ba_path)

    return model_ab, model_ba

def preprocess_image(image: np.array, size: Tuple[int, int]) -> np.array:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, size)
    normalized_image = normalize(resized_image)
    return normalized_image[np.newaxis,...]

def postprocess_prediction(prediction: np.array):
    return np.array((prediction[0] + 1) * 127.5, np.uint8)

def generate_prediction_triple(image, model_ab, model_ba):
    IMAGE_SIZE = (256, 256)
    image_input = preprocess_image(image, IMAGE_SIZE)
    prediction_ab = model_ab(image_input)
    prediction_ab = postprocess_prediction(prediction_ab)

    prediction_ba = model_ba(image_input)
    prediction_ba = postprocess_prediction(prediction_ba)

    image_viz = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), IMAGE_SIZE)
    return image_viz, prediction_ab, prediction_ba


def main():
    tabby2tortie, tortie2tabby = load_model()

    st.title("Tabby2Tortie")
    tabby_upload = st.sidebar.file_uploader("Tabby Cat", type=["jpg", "jpeg", "png"])
    tortie_upload = st.sidebar.file_uploader("Tortiseshell Cat", type=["jpg", "jpeg", "png"])
    image_col, tabby_col, tortie_col = st.columns(3)
    image_col.subheader("Image")
    tabby_col.subheader("Tabby")
    tortie_col.subheader("Tortie")

    if tabby_upload:
        tabby_image = cv2.imdecode(np.fromstring(tabby_upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        image, tortie, tabby = generate_prediction_triple(tabby_image, tabby2tortie, tortie2tabby)
        image_col.image(image)
        tabby_col.image(tabby)
        tortie_col.image(tortie)

    if tortie_upload:
        tortie_image = cv2.imdecode(np.fromstring(tortie_upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        image, tortie, tabby = generate_prediction_triple(tortie_image, tabby2tortie, tortie2tabby)
        image_col.image(image)
        tabby_col.image(tabby)
        tortie_col.image(tortie)

if __name__ == "__main__":
    main()
