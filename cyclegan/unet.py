import tensorflow as tf
from typing import List, Dict

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, LeakyReLU, \
    Concatenate, Conv2D, Dropout, Activation
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


def strided_unet(config: Dict) -> Model:
    filters = config['filters']
    kernel_sizes = config['kernels']
    norm_type = config['normalization']
    output_channels = config['output_channels']
    final_activation = config['final_activation']

    initializer = tf.random_normal_initializer(0., 0.02)
    skips = []
    inputs = Input(shape=[None, None, 3])
    x = inputs

    down_filters = filters
    up_filters = filters[::-1][:-1]
    for filter, kernel_size in list(zip(down_filters, kernel_sizes))[:-1]:
        x = Conv2D(filter, kernel_size, strides=2, padding='same', kernel_initializer=initializer)(x)
        if norm_type == 'instancenorm':
            x = InstanceNormalization()(x)
        else:
            x = BatchNormalization()(x)

        x = LeakyReLU(0.2)(x)
        skips.insert(0, x)

    x = Conv2D(filters[-1], kernel_sizes[-1], strides=2, padding='same', kernel_initializer=initializer)(x)

    for filter, skip, kernel_size in zip(up_filters, skips, kernel_sizes[:0:-1]):
        x = Conv2DTranspose(filter, kernel_size=kernel_size, padding='same', strides=2, kernel_initializer=initializer)(
            x)
        x = Concatenate()([skip, x])
        if norm_type == 'instancenorm':
            x = InstanceNormalization()(x)
        else:
            x = BatchNormalization()(x)

        x = LeakyReLU(0.2)(x)

    last = Conv2DTranspose(output_channels, 4, strides=2, padding='same', kernel_initializer=initializer,
                           activation=final_activation)(x)
    return Model(inputs, last)


def unet_generator(config: Dict) -> Model:
    filters = config['filters']
    kernel_sizes = config['kernels']
    expansion = config['expansion']
    norm_type = config['normalization']
    apply_dropout = config['dropout']
    output_channels = config['output_channels']
    final_activation = config['final_activation']

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

    x = Conv2D(output_channels, kernel_size=1, strides=1, padding='same')(x)
    last = Activation(final_activation)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=last)
    return model
