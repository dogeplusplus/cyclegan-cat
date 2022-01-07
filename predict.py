import cv2
import numpy as np
import streamlit as st
from typing import Tuple
from pathlib import Path

import tensorflow as tf

from cyclegan.model import CycleGan
from transform.data_load import normalize
from model_processing.load_model import yaml2namespace


def load_model():
    model_config_path = Path("model_instances", "model", "model_config.yaml")
    train_config_path = Path("model_instances", "model", "train_config.yaml")
    model = CycleGan(yaml2namespace(str(model_config_path)), yaml2namespace(str(train_config_path)))
    return model

def preprocess_image(image: np.array, size: Tuple[int, int]) -> np.array:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, size)
    normalized_image = normalize(resized_image)
    return normalized_image[np.newaxis,...]

def postprocess_prediction(prediction: np.array):
    return np.array((prediction[0] + 1) * 127.5, np.uint8)

def generate_prediction_triple(image, model):
    IMAGE_SIZE = (256, 256)
    image_input = preprocess_image(image, IMAGE_SIZE)
    prediction_ab = model.g_AB(image_input)
    prediction_ab = postprocess_prediction(prediction_ab)

    prediction_ba = model.g_BA(image_input)
    prediction_ba = postprocess_prediction(prediction_ba)

    image_viz = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), IMAGE_SIZE)
    return image_viz, prediction_ab, prediction_ba


def main():
    model = load_model()

    st.title("Tabby2Tortie")
    tabby_upload = st.sidebar.file_uploader("Tabby Cat", type=["jpg", "jpeg", "png"])
    tortie_upload = st.sidebar.file_uploader("Tortiseshell Cat", type=["jpg", "jpeg", "png"])
    image_col, tabby_col, tortie_col = st.columns(3)
    image_col.subheader("Image")
    tabby_col.subheader("Tabby")
    tortie_col.subheader("Tortie")

    if tabby_upload:
        tabby_image = cv2.imdecode(np.fromstring(tabby_upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        image, tortie, tabby = generate_prediction_triple(tabby_image, model)
        image_col.image(image)
        tabby_col.image(tabby)
        tortie_col.image(tortie)

    if tortie_upload:
        tortie_image = cv2.imdecode(np.fromstring(tortie_upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        image, tortie, tabby = generate_prediction_triple(tortie_image, model)
        image_col.image(image)
        tabby_col.image(tabby)
        tortie_col.image(tortie)

if __name__ == "__main__":
    main()
