import tensorflow as tf
import logging
from cyclegan.model import CycleGan
from data_processing.data_load import create_dataset
from model_processing.load_model import yaml2namespace

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    model_config = yaml2namespace("configs/cycle.yaml")
    gan = CycleGan(model_config)
    training_config = yaml2namespace("configs/training_config.yaml")
    train_ds, val_ds = create_dataset('data/cat_faces.tfrecords', 'data/scrunge_faces.tfrecords', width=training_config.image_size)
    gan.train(
        train_dataset=train_ds,
        validation_dataset=val_ds,
        training_config = training_config
    )


