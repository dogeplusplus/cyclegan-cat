import tensorflow as tf

from typing import List, Tuple
from tensorflow.data import Dataset


def example2image(example: tf.train.Example) -> tf.Tensor:
    feature = {
        "image_raw": tf.io.FixedLenFeature([], dtype=tf.string),
        "height": tf.io.FixedLenFeature([], dtype=tf.int64),
        "width": tf.io.FixedLenFeature([], dtype=tf.int64),
        "depth": tf.io.FixedLenFeature([], dtype=tf.int64)
    }
    parsed = tf.io.parse_single_example(example, feature)
    image_vector = tf.image.decode_image(parsed["image_raw"], channels=3)
    image = tf.reshape(image_vector, (parsed["height"], parsed["width"], parsed["depth"]))
    return image


def apply_augmentation(dataset: Dataset, image_size: int) -> Dataset:
    def random_jitter(image):
        image = tf.image.resize(image, [image_size + 50, image_size + 50])
        image = tf.image.random_crop(image, size=[image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
        return image

    dataset = dataset.map(random_jitter)
    return dataset


def normalize(tensor: tf.Tensor) -> tf.Tensor:
    image = tf.cast(tensor, tf.float32)
    image = (image / 127.5) - 1
    return image


def create_dataset(records_a: List[str], records_b: List[str], validation_split=0.2, width=128) -> Tuple[Dataset, Dataset]:
    def apply_mappings(dataset: Dataset, image_size: int) -> Dataset:
        dataset = dataset.map(example2image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x: tf.image.resize(x, image_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        shuffle_buffer = 1000
        dataset = dataset.shuffle(shuffle_buffer)
        return dataset

    data_a = tf.data.TFRecordDataset(records_a)
    data_b = tf.data.TFRecordDataset(records_b)

    image_size = [width, width]
    data_a = apply_mappings(data_a, image_size)
    data_b = apply_mappings(data_b, image_size)

    num_samples = sum(1 for _ in data_a)
    num_validation = int(validation_split * num_samples)

    train_a = data_a.skip(num_validation)
    train_b = data_b.skip(num_validation)
    val_a = data_a.take(num_validation)
    val_b = data_b.take(num_validation)

    train_a = apply_augmentation(train_a, width)
    train_b = apply_augmentation(train_b, width)

    train_dataset = Dataset.zip((train_a, train_b))
    val_dataset = Dataset.zip((val_a, val_b))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset
