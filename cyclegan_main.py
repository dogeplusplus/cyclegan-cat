import os
import logging
import bunch
from models.cyclegan import CycleGan
from data_processing.data_load import create_dataset
from model_processing.load_model import json2dict
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    IMAGE_SIZE = 512
    gan = CycleGan(bunch.Bunch(json2dict("configs/cycle.json")))

    BATCH_SIZE = 4
    EPOCHS = 50
    train_ds, val_ds = create_dataset('data/cats.tfrecords', 'data/scrunge.tfrecords', width=IMAGE_SIZE)

    gan.train(
        train_dataset=train_ds,
        validation_dataset=val_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
