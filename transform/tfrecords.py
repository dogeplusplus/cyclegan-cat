import cv2
import random
import numpy as np
import tensorflow as tf

from pathlib import Path

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image2example(image: np.array) -> tf.train.Example:
    height, width, depth = image.shape
    image_bytes = cv2.imencode(".png", image)[1].tobytes()
    feature = {
        "image_raw": _byte_feature(image_bytes),
        "height": _int64_feature(height),
        "width": _int64_feature(width),
        "depth": _int64_feature(depth)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def tfrecord_writer(image_paths: str, target: str, image_size: int = None, shard_size: int = 800):
    images = list(Path(image_paths).iterdir())
    random.shuffle(images)
    logger.info(f"Images Found: {len(images)}")

    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(images), shard_size):
        record_file = str(target / f"{i // shard_size:05d}.tfrecords")
        with tf.io.TFRecordWriter(record_file) as writer:
            for image in images[i*shard_size:(i+1)*shard_size]:
                img = cv2.imread(str(image), cv2.IMREAD_COLOR)
                if image_size:
                    img = cv2.resize(img, (image_size, image_size))
                feature = image2example(img)
                writer.write(feature.SerializeToString())

