import tensorflow as tf
from typing import List

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, LeakyReLU, \
    Concatenate, Conv2D, Dropout
from tensorflow.python.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization


def double_conv(filter: int, kernel_size: int,
                norm_type: str = 'instancenorm', apply_dropout: bool = False):
    layers = tf.keras.models.Sequential()
    initializer = tf.random_normal_initializer(0., 0.02)
    for _ in range(2):
        layers.add(Conv2D(filter, kernel_size, strides=1, padding='same',
                          kernel_initializer=initializer, use_bias=False))
        if norm_type.lower() == 'batchnorm':
            layers.add(BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            layers.add(InstanceNormalization())

        layers.add(LeakyReLU(0.2))
        if apply_dropout:
            layers.add(Dropout(0.5))

    return layers


def unet_generator(filters: List[int], kernel_sizes: List[int], output_channels: int,
                   norm_type: str = 'instancenorm', apply_dropout: bool = False, expansion='upsample') -> Model:
    initializer = tf.random_normal_initializer(0., 0.02)
    skips = []
    inputs = Input(shape=[None, None, 3])
    x = inputs

    down_filters = filters
    up_filters = filters[::-1][:-1]
    # Downsampling layers
    for filter, kernel_size in list(zip(down_filters, kernel_sizes))[:-1]:
        x = double_conv(filter, kernel_size, norm_type, apply_dropout)(x)
        skips.insert(0, x)
        x = MaxPooling2D()(x)

    # Bottom section
    x = double_conv(down_filters[-1], kernel_sizes[-1], norm_type, apply_dropout)(x)

    # Upsampling section
    for filter, skip, kernel_size in zip(up_filters, skips, kernel_sizes[:0:-1]):
        if expansion == 'upsample':
            x = UpSampling2D()(x)
        else:
            x = Conv2DTranspose(filter, kernel_size=kernel_size, padding='same', strides=2,
                                kernel_initializer=initializer)(x)
            if norm_type == 'instancenorm':
                x = InstanceNormalization()(x)
            else:
                x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)
        x = Concatenate()([skip, x])
        x = double_conv(filter, kernel_size, norm_type, apply_dropout)(x)

    if output_channels:
        last = Conv2D(output_channels, kernel_size=1, strides=1, padding='same', kernel_initializer=initializer)(x)
    else:
        last = x

    model = tf.keras.models.Model(inputs=inputs, outputs=last)
    return model


def unet_discriminator(filters: List[int], kernel_sizes: List[int],
                       norm_type: str = 'instancenorm', expansion='upsample') -> Model:
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    Args:
        filters: filters per convolution
        kernel_size: convolution kernel size
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

    Returns:
      Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = Input(shape=[None, None, 3])

    unet = unet_generator(filters, kernel_sizes, None, norm_type, expansion)
    x = unet(inputs)
    last = Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer, activation='sigmoid')(x)  # (bs, 30, 30, 1)

    return Model(inputs=inputs, outputs=last)
