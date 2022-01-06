import logging
import tensorflow as tf

from pathlib import Path
from argparse import ArgumentParser

from cyclegan.model import CycleGan
from transform.data_load import create_dataset
from model_processing.load_model import yaml2namespace

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_arguments():
    parser = ArgumentParser("Train cycle GAN")
    parser.add_argument("--model_config",
                        default=Path("configs", "cycle.yaml"),
                        help="Path to model config.")
    parser.add_argument("--train_config",
                        default=Path("configs", "training_config.yaml"),
                        help="Path to training config")
    parser.add_argument(
        "--vram",
        type=int,
        default=20000,
        help="Maximum amount in VRAM to use during training (MB)")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.vram is not None:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=args.vram)
                ])
    model_config = yaml2namespace(args.model_config)
    training_config = yaml2namespace(args.train_config)
    gan = CycleGan(model_config, training_config)

    records_a = list(map(str, Path("data", "tabby_records").iterdir()))
    records_b = list(map(str, Path("data", "tortie_records").iterdir()))
    train_ds, val_ds = create_dataset(records_a=records_a,
                                      records_b=records_b,
                                      width=training_config.image_size)
    gan.train(train_dataset=train_ds, validation_dataset=val_ds)


if __name__ == "__main__":
    main()
