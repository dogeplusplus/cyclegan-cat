import os

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from cyclegan.model import CycleGan
from data_processing.data_load import normalize
from model_processing.load_model import json2namespace


def parse_arguments():
    parser = argparse.ArgumentParser(description='Predict on single image')
    parser.add_argument('--model', type=str, help='Path to model folder')
    parser.add_argument('--image', type=str, help='Path to image')

    args = parser.parse_args()
    return args

def preprocess_image(image: np.array, size: Tuple[int, int]) -> np.array:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, size)
    normalized_image = normalize(resized_image)
    return normalized_image[np.newaxis,...]

def postprocess_prediction(prediction: np.array):
    return np.array((prediction[0] + 1) * 127.5, np.uint8)

def main(args):
    IMAGE_SIZE = (256, 256)
    model = CycleGan(model_config=json2namespace(os.path.join(args.model, 'model_config.json')))
    image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    image_input = preprocess_image(image, IMAGE_SIZE)
    prediction = model.g_AB(image_input)
    prediction = postprocess_prediction(prediction)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), IMAGE_SIZE))
    ax[1].imshow(prediction)
    plt.show()

if __name__ == "__main__":
    # args = parse_arguments()
    # main(args)

    args = argparse.Namespace(model="model_instances/sam2nyx_resnet_simple_256/", image="C:\\Users\\doge\\Downloads\\sam.jpg")
    main(args)