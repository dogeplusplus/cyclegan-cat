from typing import List, Dict

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, ReLU, Add, Activation, Conv2DTranspose, BatchNormalization, LeakyReLU, Layer
from tensorflow.python.keras.models import Model
from tensorflow.python.layers.base import InputSpec
from tensorflow_addons.layers import InstanceNormalization


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


def residual(layer, filters, initializer):
    x = ReflectionPadding2D(padding=(1, 1))(layer)
    x = Conv2D(filters, kernel_size=3, strides=1, padding='valid', kernel_initializer=initializer)(x)
    x = InstanceNormalization(center=False, scale=False)(x)
    x = ReLU()(x)

    x = ReflectionPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters, kernel_size=3, strides=1, padding='valid', kernel_initializer=initializer)(x)
    x = InstanceNormalization(center=False, scale=False)(x)
    return Add()([layer, x])


def conv7s1(layer_input, filters, final, initializer):
    x = ReflectionPadding2D(padding=(3, 3))(layer_input)
    x = Conv2D(filters, kernel_size=7, strides=1, padding='valid', kernel_initializer=initializer)(x)
    if final:
        x = Activation('tanh')(x)
    else:
        x = InstanceNormalization(center=False, scale=False)(x)
        x = Activation('relu')(x)
    return x


def downsample(layer, filters, initializer):
    x = Conv2D(filters, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(layer)
    x = InstanceNormalization(center=False, scale=False)(x)
    x = Activation('relu')(x)
    return x


def upsample(layer, filters, initializer):
    x = Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(layer)
    x = InstanceNormalization(center=False, scale=False)(x)
    x = Activation('relu')(x)
    return x


def resnet_generator(config: Dict):
    filters = config['filters']
    input = Input([None, None, 3])
    initializer = tf.random_normal_initializer(0., 0.02)
    x = input
    x = conv7s1(x, filters, False, initializer)
    x = downsample(x, filters * 2, initializer)
    x = downsample(x, filters * 4, initializer)
    x = residual(x, filters * 4, initializer)
    x = residual(x, filters * 4, initializer)
    x = residual(x, filters * 4, initializer)
    x = residual(x, filters * 4, initializer)
    x = residual(x, filters * 4, initializer)
    x = residual(x, filters * 4, initializer)
    x = residual(x, filters * 4, initializer)
    x = residual(x, filters * 4, initializer)
    x = residual(x, filters * 4, initializer)
    x = upsample(x, filters * 2, initializer)
    x = upsample(x, filters, initializer)
    x = conv7s1(x, 3, True, initializer)
    output = x

    return Model(input, output)

def simple_discriminator(config: Dict) -> Model:
    down_filters = config['filters']
    kernel_size = config['kernels']
    norm_type = config['normalization']
    input = Input([None, None, 3])

    x = input
    initializer = tf.random_normal_initializer(0., 0.02)
    for filter in down_filters:
        x = Conv2D(filter, kernel_size, strides=2, padding='same', kernel_initializer=initializer)(x)
        if norm_type == 'instancenorm':
            x = InstanceNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization(center=False, scale=False)(x)
        x = LeakyReLU(0.2)(x)

    output = Conv2D(1, kernel_size, strides=1, padding='same', kernel_initializer=initializer)(x)

    return Model(input, output)
