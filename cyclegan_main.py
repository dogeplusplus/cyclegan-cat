import os
import logging
import bunch
from cyclegan.model import CycleGan
from data_processing.data_load import create_dataset, tfrecord_writer
from model_processing.load_model import json2dict
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    IMAGE_SIZE = 128
    gan = CycleGan(bunch.Bunch(json2dict("configs/cycle.json")))

    # tfrecord_writer('data/apple2orange/trainA', 'data/apples.tfrecords')
    # tfrecord_writer('data/apple2orange/trainB', 'data/oranges.tfrecords')
    BATCH_SIZE = 4
    EPOCHS = 100
    train_ds, val_ds = create_dataset('data/horses.tfrecords', 'data/zebras.tfrecords', width=IMAGE_SIZE)

    gan.train(
        train_dataset=train_ds,
        validation_dataset=val_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
