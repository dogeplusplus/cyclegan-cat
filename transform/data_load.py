import os

import cv2
import logging
import numpy as np
import tensorflow as tf

from os.path import join
from functools import partial

logger = logging.getLogger(__name__)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image2example(image: np.array) -> tf.train.Example:
    height, width, depth = image.shape
    image_bytes = cv2.imencode('.png', image)[1].tobytes()
    feature = {
        'image_raw': _byte_feature(image_bytes),
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'depth': _int64_feature(depth)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def tfrecord_writer(image_paths: str, target: str = 'images.tfrecords', image_size: int = None):
    images = os.listdir(image_paths)
    logger.info(f'Images Found: {len(images)}')
    with tf.io.TFRecordWriter(target) as writer:
        for image in images:
            img = cv2.imread(join(image_paths, image), cv2.IMREAD_COLOR)
            if image_size:
                cv2.resize(img, (image_size, image_size))
            feature = image2example(img)
            writer.write(feature.SerializeToString())


def example2image(example):
    feature = {
        'image_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        'height': tf.io.FixedLenFeature([], dtype=tf.int64),
        'width': tf.io.FixedLenFeature([], dtype=tf.int64),
        'depth': tf.io.FixedLenFeature([], dtype=tf.int64)
    }
    parsed = tf.io.parse_single_example(example, feature)
    image_vector = tf.image.decode_image(parsed['image_raw'], channels=3)
    image = tf.reshape(image_vector, (parsed['height'], parsed['width'], parsed['depth']))
    return image


def apply_augmentation(dataset, image_size):
    def random_jitter(image):
        image = tf.image.resize(image, [image_size + 50, image_size + 50],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.image.random_crop(image, size=[image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
        return image

    dataset = dataset.map(random_jitter)
    return dataset


def normalize(tensor):
    image = tf.cast(tensor, tf.float32)
    image = (image / 127.5) - 1
    return image


def create_dataset(records_a, records_b, validation_split=0.2, width=128):
    def apply_mappings(dataset, image_size):
        resize = partial(tf.image.resize, size=image_size)

        dataset = dataset.map(example2image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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

    train_dataset = tf.data.Dataset.zip((train_a, train_b))
    val_dataset = tf.data.Dataset.zip((val_a, val_b))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset
