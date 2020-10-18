import tqdm
import logging
import tensorflow as tf

from typing import NoReturn, Dict, List
from os.path import join

from bunch import Bunch
from tensorflow import Tensor
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, ZeroPadding2D, Dropout, \
    Concatenate, Input, ReLU, UpSampling2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Conv2DTranspose, Add
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow_addons.layers import InstanceNormalization

from cyclegan.losses import calc_cycle_loss, identity_loss, discriminator_loss, generator_loss
from cyclegan.optimizers import get_optimizer
from model_processing.load_model import dict2json

logger = logging.getLogger(__name__)
MODEL_CLASS = 'CycleGan'

def accuracy(real, fake):
    predictions = tf.cast(tf.concat([real, fake], axis=0) > 0.5, tf.float32)
    labels = tf.concat([tf.ones_like(real), tf.zeros_like(fake, tf.float32)], axis=0)
    acc = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
    return acc


def downsample(filters: int, size: int, norm_type: str = 'instancenorm', apply_norm: bool = True) -> Sequential:
    """Create downsampling section for the unet

    Args:
        filters: filters in the convolution
        size: kernel size
        norm_type: type of normalization on the outputs
        apply_norm: flag to apply normalization

    Returns:
        Downsampling layers as a submodel
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = Sequential()
    for _ in range(2):
        result.add(
            Conv2D(filters, size, strides=2, padding='same',
                   kernel_initializer=initializer, use_bias=False))

        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                result.add(BatchNormalization())
            elif norm_type.lower() == 'instancenorm':
                result.add(InstanceNormalization())

        result.add(LeakyReLU())

    return result


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

        layers.add(ReLU())
        if apply_dropout:
            layers.add(Dropout(0.5))

    return layers


def unet_generator(filters: List[int], kernel_sizes: List[int], output_channels: int,
                   norm_type: str = 'instancenorm', apply_dropout: bool = False) -> Model:
    inputs = Input(shape=[None, None, 3])
    x = inputs
    skips = []

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
        x = UpSampling2D()(x)
        x = Concatenate()([skip, x])
        x = double_conv(filter, kernel_size, norm_type, apply_dropout)(x)

    initializer = tf.random_normal_initializer(0., 0.02)
    if output_channels:
        last = Conv2D(output_channels, kernel_size, strides=1, padding='same', kernel_initializer=initializer)(x)
    else:
        last = x

    model = tf.keras.models.Model(inputs=inputs, outputs=last)
    return model


def unet_discriminator(filters: List[int], kernel_sizes: List[int],
                       norm_type: str = 'instancenorm') -> Model:
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

    unet = unet_generator(filters, kernel_sizes, None)
    x = unet(inputs)

    down1 = downsample(32, 4, norm_type, False)(x)
    down2 = downsample(32, 4, norm_type)(down1)

    zero_pad1 = ZeroPadding2D()(down2)
    conv = Conv2D(
        64, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)

    if norm_type.lower() == 'batchnorm':
        norm1 = BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = LeakyReLU()(norm1)

    zero_pad2 = ZeroPadding2D()(leaky_relu)
    last = Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer, activation='sigmoid')(zero_pad2)  # (bs, 30, 30, 1)

    return Model(inputs=inputs, outputs=last)


def simple_discriminator(down_filters: List[int], kernel_size: int, norm_type: str = 'instancenorm') -> Model:
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


def resnet_generator(filters: int):
    initializer = tf.random_normal_initializer(0., 0.02)

    def residual(layer, filters):
        x = ReflectionPadding2D(padding=(1, 1))(layer)
        x = Conv2D(filters, kernel_size=3, strides=1, padding='valid', kernel_initializer=initializer)(x)
        x = InstanceNormalization(center=False, scale=False)(x)
        x = ReLU()(x)

        x = ReflectionPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, kernel_size=3, strides=1, padding='valid', kernel_initializer=initializer)(x)
        x = InstanceNormalization(center=False, scale=False)(x)
        return Add()([layer, x])

    def conv7s1(layer_input, filters, final):
        x = ReflectionPadding2D(padding=(3, 3))(layer_input)
        x = Conv2D(filters, kernel_size=7, strides=1, padding='valid', kernel_initializer=initializer)(x)
        if final:
            x = Activation('tanh')(x)
        else:
            x = InstanceNormalization(center=False, scale=False)(x)
            x = Activation('relu')(x)
        return x

    def downsample(layer, filters):
        x = Conv2D(filters, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(layer)
        x = InstanceNormalization(center=False, scale=False)(x)
        x = Activation('relu')(x)
        return x

    def upsample(layer, filters):
        x = Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(layer)
        x = InstanceNormalization(center=False, scale=False)(x)
        x = Activation('relu')(x)
        return x

    input = Input([None, None, 3])
    x = input

    x = conv7s1(x, filters, False)
    x = downsample(x, filters * 2)
    x = downsample(x, filters * 4)
    x = residual(x, filters * 4)
    x = residual(x, filters * 4)
    x = residual(x, filters * 4)
    x = residual(x, filters * 4)
    x = residual(x, filters * 4)
    x = residual(x, filters * 4)
    x = residual(x, filters * 4)
    x = residual(x, filters * 4)
    x = residual(x, filters * 4)
    x = upsample(x, filters * 2)
    x = upsample(x, filters)
    x = conv7s1(x, 3, True)
    output = x

    return Model(input, output)


class CycleGan(Model):
    def __init__(self, model_config: Bunch):
        super(CycleGan, self).__init__()
        self.model_config = model_config
        self.model_location = getattr(self.model_config, 'location')
        self.train_summaries = tf.summary.create_file_writer(
            join(self.model_location, self.model_config.name, 'train'))
        self.val_summaries = tf.summary.create_file_writer(
            join(self.model_location, self.model_config.name, 'validation')
        )
        self.loss_weights = self.model_config.loss_weights
        if self.model_config.new:
            self.build_models()
            self.model_config.new = False
            dict2json(self.model_config, join(self.model_location, self.model_config.name, 'model_config.json'))
        else:
            self.load_model()

    def build_models(self) -> NoReturn:
        # get model attributes
        generator_filters = self.model_config.generator['filters']
        discriminator_filters = self.model_config.discriminator['filters']
        gen_kernel_size = self.model_config.generator['kernels']
        disc_kernel_size = self.model_config.discriminator['kernels']

        # TODO: clean up the generator and discriminator code to make it work for all combinations
        self.g_AB = unet_generator(generator_filters, gen_kernel_size, 3)
        self.g_BA = unet_generator(generator_filters, gen_kernel_size, 3)
        # self.g_AB = resnet_generator(64)
        # self.g_BA = resnet_generator(64)
        # self.d_A = simple_discriminator([32, 32, 64, 64], 4)
        # self.d_B = simple_discriminator([32, 32, 64, 64], 4)
        self.d_A = unet_discriminator(discriminator_filters, disc_kernel_size)
        self.d_B = unet_discriminator(discriminator_filters, disc_kernel_size)


    @tf.function
    def validate_step(self, real_a: Tensor, real_b: Tensor, training: bool = False) -> Dict:
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        fake_b = self.g_AB(real_a, training=training)
        cycled_a = self.g_BA(fake_b, training=training)

        fake_a = self.g_BA(real_b, training=training)
        cycled_b = self.g_AB(fake_a, training=training)

        same_a = self.g_BA(real_a, training=training)
        same_b = self.g_AB(real_b, training=training)

        disc_real_a = self.d_A(real_a, training=training)
        disc_real_b = self.d_B(real_b, training=training)

        disc_fake_a = self.d_A(fake_a, training=training)
        disc_fake_b = self.d_B(fake_b, training=training)

        gAB_loss = generator_loss(disc_fake_b, loss_obj, self.loss_weights['generator'])
        gBA_loss = generator_loss(disc_fake_a, loss_obj, self.loss_weights['generator'])

        total_cycle_loss = calc_cycle_loss(real_a, cycled_a, self.loss_weights['cycle']) + calc_cycle_loss(real_b,
                                                                                                           cycled_b,
                                                                                                           self.loss_weights[
                                                                                                               'cycle'])

        total_gAB_loss = gAB_loss + total_cycle_loss + identity_loss(real_b, same_b,
                                                                     self.loss_weights['identity'])
        total_gBA_loss = gBA_loss + total_cycle_loss + identity_loss(real_a, same_a,
                                                                     self.loss_weights['identity'])

        da_loss = discriminator_loss(disc_real_a, disc_fake_a, loss_obj, self.loss_weights['discriminator'])
        db_loss = discriminator_loss(disc_real_b, disc_fake_b, loss_obj, self.loss_weights['discriminator'])

        da_accuracy = accuracy(disc_real_a, disc_fake_a)
        db_accuracy = accuracy(disc_real_b, disc_fake_b)

        metrics = dict(
            gAB_loss=total_gAB_loss,
            gBA_loss=total_gBA_loss,
            dA_loss=da_loss,
            dB_loss=db_loss,
            dA_acc=da_accuracy,
            dB_acc=db_accuracy
        )
        return metrics

    @tf.function
    def train_step(self, real_a: Tensor, real_b: Tensor) -> Dict:
        with tf.GradientTape(persistent=True) as tape:
            metrics = self.validate_step(real_a,
                                         real_b,
                                         training=True)

        g_AB_gradients = tape.gradient(metrics['gAB_loss'], self.g_AB.trainable_variables)
        g_BA_gradients = tape.gradient(metrics['gBA_loss'], self.g_BA.trainable_variables)

        da_gradients = tape.gradient(metrics['dA_loss'], self.d_A.trainable_variables)
        db_gradients = tape.gradient(metrics['dB_loss'], self.d_B.trainable_variables)

        self.g_AB_optimizer.apply_gradients(zip(g_AB_gradients, self.g_AB.trainable_variables))
        self.g_BA_optimizer.apply_gradients(zip(g_BA_gradients, self.g_BA.trainable_variables))

        self.d_A_optimizer.apply_gradients(zip(da_gradients, self.d_A.trainable_variables))
        self.d_B_optimizer.apply_gradients(zip(db_gradients, self.d_B.trainable_variables))
        return metrics

    def train(self, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, training_config: Bunch) -> NoReturn:
        self.g_AB_optimizer = get_optimizer(training_config.g_opt)
        self.g_BA_optimizer = get_optimizer(training_config.g_opt)
        self.d_A_optimizer = get_optimizer(training_config.d_opt)
        self.d_B_optimizer = get_optimizer(training_config.d_opt)

        batch_size = training_config.batch_size
        epochs = training_config.epochs
        save_images_every = training_config.summary['images']
        tensorboard_samples = training_config.summary['samples']
        save_model_every = training_config.summary['model']
        sample_images = []

        metric_names = [
            'dA_loss',
            'dB_loss',
            'gAB_loss',
            'gBA_loss',
            'dA_acc',
            'dB_acc',
        ]

        train_metrics_dict = {
            m: tf.keras.metrics.Mean(name=m)
            for m in metric_names
        }

        validation_metrics_dict = {
            m: tf.keras.metrics.Mean(name=m)
            for m in metric_names
        }

        val_iter = iter(validation_dataset)
        for _ in range(tensorboard_samples):
            sample_images.append(next(val_iter))

        a_samples = tf.stack([s[0] for s in sample_images])
        b_samples = tf.stack([s[1] for s in sample_images])
        with self.val_summaries.as_default():
            tf.summary.image(name='A', data=tf.add(a_samples, 1) / 2, step=0, max_outputs=tensorboard_samples)
            tf.summary.image(name='B', data=tf.add(b_samples, 1) / 2, step=0, max_outputs=tensorboard_samples)

        train_dataset = train_dataset.batch(batch_size)
        validation_dataset = validation_dataset.batch(batch_size)
        training_size = sum(1 for _ in train_dataset)
        validation_size = sum(1 for _ in validation_dataset)
        desc = 'Epoch {} training'
        val_desc = 'Epoch {} validation'
        for e in range(epochs):
            train_bar = tqdm.tqdm(train_dataset, desc=desc.format(e + 1), ncols=150, total=training_size)
            for i, (images_a, images_b) in enumerate(train_bar):
                losses = self.train_step(images_a, images_b)
                self.update_metrics(train_metrics_dict, losses)
                self.display_metrics(train_metrics_dict, train_bar)

            self.write_summaries(self.train_summaries, e, train_metrics_dict)
            if e % save_images_every == 0:
                self.write_images(e, a_samples, b_samples, tensorboard_samples)

            # TODO: probably need some checkpointing here
            if e % save_model_every == 0:
                # self.save_model()
                pass
            val_bar = tqdm.tqdm(validation_dataset, desc=val_desc.format(e + 1), ncols=150, total=validation_size)
            for i, (images_a, images_b) in enumerate(val_bar):
                losses = self.validate_step(images_a, images_b, training=False)
                self.update_metrics(validation_metrics_dict, losses)
                self.display_metrics(validation_metrics_dict, val_bar)
            self.write_summaries(self.val_summaries, e, validation_metrics_dict)

        self.save_model()

    def write_summaries(self, summaries: tf.summary.SummaryWriter, epoch: int, metrics_dict: Dict[str, Tensor]):
        """Write summaries into tensorboard

        Args:
            summaries: training or validation summaries
            epoch: epoch number
            metrics_dict: dictionary of metrics
        """
        with summaries.as_default():
            for name, metric in metrics_dict.items():
                tf.summary.scalar(name, metric.result(), step=epoch)
                metrics_dict[name].reset_states()

    def write_images(self, epoch: int, a_samples: Tensor, b_samples: Tensor, num_samples):
        """Write summaries into tensorboard

        Args:
            epoch: epoch number
            a_samples: sample images of class a for tensorboard
            b_samples: sample images of class b for tensorboard
            num_samples: number of samples of each class to display
        """
        with self.val_summaries.as_default():
            prediction_ab = self.g_AB.predict(x=a_samples, batch_size=1)
            prediction_ba = self.g_BA.predict(x=b_samples, batch_size=1)
            tf.summary.image(name='A2B_predictions', data=tf.add(prediction_ab, 1) / 2,
                             step=epoch,
                             max_outputs=num_samples)
            tf.summary.image(name='B2A_predictions', data=tf.add(prediction_ba, 1) / 2,
                             step=epoch,
                             max_outputs=num_samples)

    def update_metrics(self, metrics_dict: Dict[str, Tensor], metrics: Dict) -> NoReturn:
        """Update the metrics dictionary with values from the training step

        Args:
            metrics_dict: dictionary of metrics
            metrics: loss values from the training batch
        """
        for name in metrics_dict.keys():
            metrics_dict[name].update_state(metrics[name])

    def display_metrics(self, metrics_dict: Dict[str, Tensor], progress_bar: tqdm.tqdm) -> NoReturn:
        """Display training progress to the console

        Args:
            metrics_dict: dictionary of metrics
            progress_bar: tqdm progress bar
        """
        evaluated_metrics = {k: str(v.result().numpy())[:7] for k, v in metrics_dict.items()}
        progress_bar.set_postfix(**evaluated_metrics)

    def save_model(self):
        tf.keras.models.save_model(self.d_A, join(self.model_location, self.model_config.name, 'd_A'))
        tf.keras.models.save_model(self.d_B, join(self.model_location, self.model_config.name, 'd_B'))
        tf.keras.models.save_model(self.g_AB, join(self.model_location, self.model_config.name, 'g_AB'))
        tf.keras.models.save_model(self.g_BA, join(self.model_location, self.model_config.name, 'g_BA'))

    def load_model(self):
        model_path = join(self.model_location, self.model_config.name)
        self.d_A = tf.saved_model.load(join(model_path, 'd_A'))
        self.d_B = tf.saved_model.load(join(model_path, 'd_B'))
        self.g_AB = tf.saved_model.load(join(model_path, 'g_AB'))
        self.g_BA = tf.saved_model.load(join(model_path, 'g_BA'))
