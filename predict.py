import os

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from cyclegan.model import CycleGan
from transform.data_load import normalize
from model_processing.load_model import yaml2namespace


def parse_arguments():
    parser = argparse.ArgumentParser(description='Predict on single image')
    parser.add_argument('--model', type=str, help='Path to model folder')
    parser.add_argument('--images', type=str, help='Path to images')

    args = parser.parse_args()
    return args

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

def main(args):
    model = CycleGan(model_config=yaml2namespace(os.path.join(args.model, 'model_config.yaml')))

    image_paths = os.listdir(args.images)
    n_rows = len(image_paths)
    fig, ax = plt.subplots(n_rows, 3, figsize=(n_rows * 2, 6))
    ax[0, 0].set_title('Original Image')
    ax[0, 1].set_title('Prediction: A -> B')
    ax[0, 2].set_title('Prediction: B -> A')

    for i, path in enumerate(image_paths):
        image = cv2.imread(os.path.join(args.images, path), cv2.IMREAD_UNCHANGED)
        img_viz, pred_ab, pred_ba = generate_prediction_triple(image, model)
        ax[i, 0].imshow(img_viz)
        ax[i, 1].imshow(pred_ab)
        ax[i, 2].imshow(pred_ba)

        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
        ax[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
