import datetime
import logging
import tensorflow as tf

from typing import NoReturn, Tuple, Dict, List
from os.path import join
from tensorflow import Tensor
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Conv2DTranspose, BatchNormalization, ZeroPadding2D, Dropout, \
    Concatenate, Input, ReLU
from tensorflow_addons.layers import InstanceNormalization
from models.base import BaseModel

logger = logging.getLogger(__name__)
MODEL_CLASS = 'CycleGan'


def downsample(filters: int, size: int, norm_type: str = 'batchnorm', apply_norm: bool = True) -> Sequential:
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


def upsample(filters: int, size: int, norm_type: str = 'batchnorm', apply_dropout: bool = False) -> Sequential:
    """Upsamples an input.

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_dropout: If True, adds the dropout layer

    Returns:
      Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        Conv2DTranspose(filters, size, strides=2,
                        padding='same',
                        kernel_initializer=initializer,
                        use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(Dropout(0.5))

    result.add(ReLU())

    return result


def unet_generator(down_filters: List[int], up_filters: List[int], kernel_size: int, output_channels: int,
                   norm_type: str = 'batchnorm') -> Model:
    """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

    Args:
        down_filters: filters per convolution during downsampling
        up_filters: filters per convolution during upsampling
        kernel_size: convolution kernel size
        output_channels: Output channels
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

    Returns:
        Generator model
    """
    down_stack = [
        downsample(down_filters[0], kernel_size, apply_norm=False)
    ]
    down_stack.extend([downsample(filters, kernel_size) for filters in down_filters[1:]])

    up_stack = [
        upsample(filters, kernel_size) for filters in up_filters
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')  # (bs, 256, 256, 3)

    inputs = Input(shape=[None, None, 3])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    x = last(x)

    return Model(inputs=inputs, outputs=x)


def discriminator(down_filters: List[int], up_filters: List[int], kernel_size: int,
                  norm_type: str = 'batchnorm') -> Model:
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    Args:
        down_filters: filters per convolution during downsampling
        up_filters: filters per convolution during upsampling
        kernel_size: convolution kernel size
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

    Returns:
      Discriminator model
    """

    down_stack = [
        downsample(down_filters[0], kernel_size, apply_norm=False)
    ]
    down_stack.extend([downsample(filters, kernel_size) for filters in down_filters[1:]])

    up_stack = [
        upsample(filters, kernel_size) for filters in up_filters
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    unet_last = Conv2DTranspose(
        3, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')  # (bs, 256, 256, 3)

    concat = Concatenate()

    inputs = Input(shape=[None, None, 3])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = unet_last(x)

    down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = Conv2D(
        512, 4, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = LeakyReLU()(norm1)

    zero_pad2 = ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
    last = Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return Model(inputs=inputs, outputs=last)


def calc_cycle_loss(real_image: Tensor, cycled_image: Tensor, weight: int = 10) -> float:
    """Calculate the cycle loss of the generators

    Args:
        real_image: image belonging to the original class
        cycled_image: image after going from both generators
        weight: weighting to apply to the loss function

    Returns:
        loss value
    """
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return weight * loss1


def generator_loss(generated: Tensor, loss_obj: tf.keras.losses.Loss, weight: float) -> float:
    """Calculate the accuracy of the generators being able to distinguish fake examples

    Args:
        generated: fake examples from the generator
        loss_obj: loss function to apply
        weight: weight for the generator loss

    Returns:
        loss value
    """
    return weight * loss_obj(tf.ones_like(generated), generated)


def identity_loss(real_image: Tensor, same_image: Tensor, weight: int = 5) -> float:
    """Calculate the loss of the generator in preserving instances of its own class

    Args:
        real_image: real image
        same_image: image after going through the generator that converts into its own class
        weight: weighting to apply to the loss function

    Returns:
        loss value
    """
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return weight * loss


def discriminator_loss(real: Tensor, generated: Tensor, loss_obj: tf.keras.losses.Loss, weight: float) -> float:
    """Calculate the loss of the discriminator to distinguish between fake and real examples

    Args:
        real: real images
        generated: fake images
        loss_obj: loss function to apply
        weight: weighting for the discriminator loss

    Returns:
        loss value
    """
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return weight * total_disc_loss


class CycleGan(BaseModel, Model):
    def __init__(self, model_config: Dict):
        super(CycleGan, self).__init__(model_config)
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.model_location = getattr(self.model_config, 'location')
        self.summaries = tf.summary.create_file_writer(
            join(self.model_location, self.model_config.name, 'logs', current_time))
        self.build_models()
        self.create_checkpoints(join(self.model_location, 'weights'))
        self.loss_weights = self.model_config.loss_weights

    def build_models(self) -> NoReturn:
        # get model attributes
        down_filters = getattr(self.model_config, 'down_filters', [64, 128, 256, 512, 512])
        up_filters = getattr(self.model_config, 'up_filters', [512, 256, 128, 64])
        kernel_size = getattr(self.model_config, 'kernel_size', 4)

        self.g_AB = unet_generator(down_filters, up_filters, kernel_size, 3)
        self.g_BA = unet_generator(down_filters, up_filters, kernel_size, 3)

        self.d_A = discriminator(down_filters, up_filters, kernel_size)
        self.d_B = discriminator(down_filters, up_filters, kernel_size)

        self.g_AB_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.g_BA_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.d_A_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1=0.5)
        self.d_B_optimizer = tf.keras.optimizers.Adam(2e-5, beta_1=0.5)

    @tf.function
    def train_step(self, real_a: Tensor, real_b: Tensor) -> Tuple[float, ...]:
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        with tf.GradientTape(persistent=True) as tape:
            fake_b = self.g_AB(real_a, training=True)
            cycled_a = self.g_BA(fake_b, training=True)

            fake_a = self.g_BA(real_b, training=True)
            cycled_b = self.g_AB(fake_a, training=True)

            same_a = self.g_BA(real_a, training=True)
            same_b = self.g_AB(real_b, training=True)

            disc_real_a = self.d_A(real_a, training=True)
            disc_real_b = self.d_B(real_b, training=True)

            disc_fake_a = self.d_A(fake_a, training=True)
            disc_fake_b = self.d_B(fake_b, training=True)

            # TODO: add loss weighting in the generator loss, make it higher potentially
            gen_AB_loss = generator_loss(disc_fake_b, loss_obj, self.loss_weights['generator'])
            gen_BA_loss = generator_loss(disc_fake_a, loss_obj, self.loss_weights['generator'])

            total_cycle_loss = calc_cycle_loss(real_a, cycled_a, self.loss_weights['cycle']) + calc_cycle_loss(real_b,
                                                                                                               cycled_b,
                                                                                                               self.loss_weights[
                                                                                                                   'cycle'])

            total_g_AB_loss = gen_AB_loss + total_cycle_loss + identity_loss(real_b, same_b, self.loss_weights['identity'])
            total_g_BA_loss = gen_BA_loss + total_cycle_loss + identity_loss(real_a, same_a, self.loss_weights['identity'])

            disc_a_loss = discriminator_loss(disc_real_a, disc_fake_a, loss_obj, self.loss_weights['discriminator'])
            disc_b_loss = discriminator_loss(disc_real_b, disc_fake_b, loss_obj, self.loss_weights['discriminator'])

        g_AB_gradients = tape.gradient(total_g_AB_loss, self.g_AB.trainable_variables)
        g_BA_gradients = tape.gradient(total_g_BA_loss, self.g_BA.trainable_variables)

        disc_a_gradients = tape.gradient(disc_a_loss, self.d_A.trainable_variables)
        disc_b_gradients = tape.gradient(disc_b_loss, self.d_B.trainable_variables)

        self.g_AB_optimizer.apply_gradients(zip(g_AB_gradients, self.g_AB.trainable_variables))
        self.g_BA_optimizer.apply_gradients(zip(g_BA_gradients, self.g_BA.trainable_variables))

        self.d_A_optimizer.apply_gradients(zip(disc_a_gradients, self.d_A.trainable_variables))
        self.d_B_optimizer.apply_gradients(zip(disc_b_gradients, self.d_B.trainable_variables))

        # TODO: add the discriminator accuracy to tensorboard
        return (total_g_AB_loss, total_g_BA_loss, disc_a_loss, disc_b_loss)

    def create_checkpoints(self, checkpoint_path: str) -> NoReturn:

        ckpt = tf.train.Checkpoint(
            generator_AB=self.g_AB,
            generator_BA=self.g_BA,
            discriminator_a=self.d_A,
            discriminator_b=self.d_B,
            generator_AB_optimizer=self.g_AB_optimizer,
            generator_BA_optimizer=self.g_BA_optimizer,
            disc_a_optimizer=self.d_A_optimizer,
            disc_b_optimizer=self.d_B_optimizer
        )
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    def train(self, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, epochs: int,
              batch_size: int = 32) -> NoReturn:
        tensorboard_samples = 9
        sample_images = []

        metric_names = [
            'discriminator_a_loss',
            'discriminator_b_loss',
            'generator_ab_loss',
            'generator_ba_loss'
        ]

        metrics_dict = {
            m: tf.keras.metrics.Mean(name=m)
            for m in metric_names
        }
        val_iter = iter(validation_dataset)
        for _ in range(tensorboard_samples):
            sample_images.append(next(val_iter))

        a_samples = tf.stack([s[0] for s in sample_images])
        b_samples = tf.stack([s[1] for s in sample_images])

        with self.summaries.as_default():
            tf.summary.image(name='A', data=tf.add(a_samples, 1) / 2, step=0, max_outputs=tensorboard_samples)
            tf.summary.image(name='B', data=tf.add(b_samples, 1) / 2, step=0, max_outputs=tensorboard_samples)

        train_dataset = train_dataset.batch(batch_size)
        n_batches = sum(1 for _ in train_dataset)
        for e in range(epochs):
            for i, (images_a, images_b) in enumerate(train_dataset):
                losses = self.train_step(images_a, images_b)
                self.update_metrics(metrics_dict, losses)
                self.display_metrics(metrics_dict, e, i + 1, n_batches)

            if e % 20 == 0:
                self.ckpt_manager.save()

            self.write_summaries(e, metrics_dict, a_samples, b_samples, tensorboard_samples)

    def write_summaries(self, epoch: int, metrics_dict: Dict[str, Tensor], a_samples: Tensor, b_samples: Tensor,
                        num_samples: int):
        """Write summaries into tensorboard

        Args:
            epoch: epoch number
            metrics_dict: dictionary of metrics
            a_samples: sample images of class a for tensorboard
            b_samples: sample images of class b for tensorboard
            num_samples: number of samples of each class to display
        """
        with self.summaries.as_default():
            for name, metric in metrics_dict.items():
                tf.summary.scalar(name, metric.result(), step=epoch)
                metrics_dict[name].reset_states()

            prediction_ab = self.g_AB.predict(x=a_samples, batch_size=1)
            prediction_ba = self.g_BA.predict(x=b_samples, batch_size=1)
            tf.summary.image(name='A2B_predictions', data=tf.add(prediction_ab, 1) / 2,
                             step=epoch,
                             max_outputs=num_samples)
            tf.summary.image(name='B2A_predictions', data=tf.add(prediction_ba, 1) / 2,
                             step=epoch,
                             max_outputs=num_samples)

    def update_metrics(self, metrics_dict: Dict[str, Tensor], losses: Tuple[float, ...]) -> NoReturn:
        """Update the metrics dictionary with values from the training step

        Args:
            metrics_dict: dictionary of metrics
            losses: loss values from the training batch
        """
        metrics_dict['generator_ab_loss'].update_state(losses[0])
        metrics_dict['generator_ba_loss'].update_state(losses[1])
        metrics_dict['discriminator_a_loss'].update_state(losses[2])
        metrics_dict['discriminator_b_loss'].update_state(losses[3])

    def display_metrics(self, metrics_dict: Dict[str, Tensor], epoch: int, iteration: int, total_batches:
    int) -> NoReturn:
        """Display training progress to the console
        
        Args:
            metrics_dict: dictionary of metrics
            epoch: epoch number
            iteration: iteration number
            total_batches: total number of batches per epoch
        """
        l_da = metrics_dict['discriminator_a_loss'].result()
        l_db = metrics_dict['discriminator_b_loss'].result()
        l_gab = metrics_dict['generator_ab_loss'].result()
        l_gba = metrics_dict['generator_ba_loss'].result()
        logger.info(
            f'Epoch {epoch} Batch {iteration}/{total_batches} [DA: {l_da} DB: {l_db}] [GAB: {l_gab} GBA: {l_gba}]'
        )
