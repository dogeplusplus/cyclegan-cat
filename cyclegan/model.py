import tqdm
import logging
import tensorflow as tf

from typing import NoReturn, Tuple, Dict, List
from os.path import join
from tensorflow import Tensor, add
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, ZeroPadding2D, Dropout, \
    Concatenate, Input, ReLU, UpSampling2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Conv2DTranspose, Add
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow_addons.layers import InstanceNormalization
from cyclegan.base import BaseModel

from cyclegan.losses import calc_cycle_loss, identity_loss, discriminator_loss, generator_loss

logger = logging.getLogger(__name__)
MODEL_CLASS = 'CycleGan'


def accuracy(real, fake):
    predictions = tf.concat([real, fake], axis=0)
    labels = tf.concat([tf.ones_like(real, tf.float32), tf.zeros_like(fake, tf.float32)], axis=0)
    return tf.keras.metrics.binary_accuracy(predictions, labels)


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


def unet_generator(down_filters: List[int], up_filters: List[int], kernel_size: int, output_channels: int,
                   norm_type: str = 'instancenorm', apply_dropout: bool = False) -> Model:
    inputs = Input(shape=[None, None, 3])
    x = inputs
    skips = []

    # Downsampling lasyers
    for filter in down_filters[:-1]:
        x = double_conv(filter, kernel_size, norm_type, apply_dropout)(x)
        skips.insert(0, x)
        x = MaxPooling2D()(x)

    # Bottom section
    x = double_conv(down_filters[-1], kernel_size, norm_type, apply_dropout)(x)

    # Upsampling section
    for filter, skip in zip(up_filters, skips):
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


def unet_discriminator(down_filters: List[int], up_filters: List[int], kernel_size: int,
                       norm_type: str = 'instancenorm') -> Model:
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    Args:
        down_filters: filters per convolution during downsampling
        up_filters: filters per convolution during upsampling
        kernel_size: convolution kernel size
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

    Returns:
      Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = Input(shape=[None, None, 3])

    unet = unet_generator(down_filters, up_filters, kernel_size, None)
    x = unet(inputs)

    down1 = downsample(64, 4, norm_type, False)(x)
    down2 = downsample(64, 4, norm_type)(down1)

    zero_pad1 = ZeroPadding2D()(down2)
    conv = Conv2D(
        512, 4, strides=1, kernel_initializer=initializer,
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
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

def resnet_generator(filters: int):
    initializer = tf.random_normal_initializer(0., 0.02)

    def residual(layer, filters):
        x = ReflectionPadding2D(padding=(1,1))(layer)
        x = Conv2D(filters, kernel_size=3, strides=1, padding='valid', kernel_initializer=initializer)(x)
        x = InstanceNormalization(center=False, scale=False)(x)
        x = ReLU()(x)

        x = ReflectionPadding2D(padding=(1,1))(x)
        x = Conv2D(filters, kernel_size=3, strides=1, padding='valid', kernel_initializer=initializer)(x)
        x = InstanceNormalization(center=False, scale=False)(x)
        return Add()([layer, x])

    def conv7s1(layer_input, filters, final):
        x = ReflectionPadding2D(padding=(3,3))(layer_input)
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
    x = upsample(x, filters*2)
    x = upsample(x, filters)
    x = conv7s1(x, 3, True)
    output = x

    return Model(input, output)

class CycleGan(BaseModel, Model):
    def __init__(self, model_config: Dict):
        super(CycleGan, self).__init__(model_config)
        self.model_location = getattr(self.model_config, 'location')
        self.train_summaries = tf.summary.create_file_writer(
            join(self.model_location, self.model_config.name, 'train'))
        self.val_summaries = tf.summary.create_file_writer(
            join(self.model_location, self.model_config.name, 'validation')
        )
        self.build_models()
        self.create_checkpoints(join(self.model_location, self.model_config.name, 'weights'))
        self.loss_weights = self.model_config.loss_weights

    def build_models(self) -> NoReturn:
        # get model attributes
        generator_down_filters = getattr(self.model_config, 'generator_down_filters')
        generator_up_filters = getattr(self.model_config, 'generator_up_filters')
        discriminator_down_filters = getattr(self.model_config, 'discriminator_down_filters')
        discriminator_up_filters = getattr(self.model_config, 'discriminator_up_filters')
        kernel_size = getattr(self.model_config, 'kernel_size', 4)

        # TODO: clean up the generator and discriminator code to make it work for all combinations
        # self.g_AB = unet_generator(generator_down_filters, generator_up_filters, kernel_size, 3)
        # self.g_BA = unet_generator(generator_down_filters, generator_up_filters, kernel_size, 3)
        # self.d_A = unet_discriminator(discriminator_down_filters, discriminator_up_filters, kernel_size)
        # self.d_B = unet_discriminator(discriminator_down_filters, discriminator_up_filters, kernel_size)
        self.g_AB = resnet_generator(32)
        self.g_BA = resnet_generator(32)
        self.d_A = simple_discriminator([32, 32, 64, 64], 4)
        self.d_B = simple_discriminator([32, 32, 64, 64], 4)
        # self.d_A = unet_discriminator(discriminator_down_filters, discriminator_up_filters, kernel_size)
        # self.d_B = unet_discriminator(discriminator_down_filters, discriminator_up_filters, kernel_size)

        self.g_AB_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.g_BA_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.d_A_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.d_B_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    @tf.function
    def validate_step(self, real_a: Tensor, real_b: Tensor, training: bool = False) -> Tuple[float, ...]:
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

        return (total_gAB_loss, total_gBA_loss, da_loss, db_loss, da_accuracy, db_accuracy)

    @tf.function
    def train_step(self, real_a: Tensor, real_b: Tensor) -> Tuple[float, ...]:
        with tf.GradientTape(persistent=True) as tape:
            total_gAB_loss, total_gBA_loss, da_loss, db_loss, da_accuracy, db_accuracy = self.validate_step(real_a,
                                                                                                            real_b,
                                                                                                            training=True)

        g_AB_gradients = tape.gradient(total_gAB_loss, self.g_AB.trainable_variables)
        g_BA_gradients = tape.gradient(total_gBA_loss, self.g_BA.trainable_variables)

        da_gradients = tape.gradient(da_loss, self.d_A.trainable_variables)
        db_gradients = tape.gradient(db_loss, self.d_B.trainable_variables)

        self.g_AB_optimizer.apply_gradients(zip(g_AB_gradients, self.g_AB.trainable_variables))
        self.g_BA_optimizer.apply_gradients(zip(g_BA_gradients, self.g_BA.trainable_variables))

        self.d_A_optimizer.apply_gradients(zip(da_gradients, self.d_A.trainable_variables))
        self.d_B_optimizer.apply_gradients(zip(db_gradients, self.d_B.trainable_variables))

        return (total_gAB_loss, total_gBA_loss, da_loss, db_loss, da_accuracy, db_accuracy)

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
        tensorboard_samples = 8
        sample_images = []

        metric_names = [
            'dA_loss',
            'dB_loss',
            'gAB_loss',
            'gBA_loss',
            'dA_accuracy',
            'dB_accuracy',
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
        for e in range(epochs):
            train_bar = tqdm.tqdm(train_dataset, desc=desc.format(e + 1), ncols=200, total=training_size)
            for i, (images_a, images_b) in enumerate(train_bar):
                losses = self.train_step(images_a, images_b)
                self.update_metrics(train_metrics_dict, losses)
                self.display_metrics(train_metrics_dict, train_bar)

            self.write_summaries(self.train_summaries, e, train_metrics_dict)
            if e % 5 == 0:
                self.write_images(e, a_samples, b_samples, tensorboard_samples)

            if e % 20 == 0:
                self.ckpt_manager.save()
            # TODO: complete validation step code
            val_bar = tqdm.tqdm(validation_dataset, desc=desc.format(e + 1), ncols=200, total=validation_size)
            for i, (images_a, images_b) in enumerate(val_bar):
                losses = self.validate_step(images_a, images_b, training=False)
                self.update_metrics(validation_metrics_dict, losses)
                self.display_metrics(validation_metrics_dict, val_bar)
            self.write_summaries(self.val_summaries, e, validation_metrics_dict)

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

    def update_metrics(self, metrics_dict: Dict[str, Tensor], losses: Tuple[float, ...]) -> NoReturn:
        """Update the metrics dictionary with values from the training step

        Args:
            metrics_dict: dictionary of metrics
            losses: loss values from the training batch
        """
        metrics_dict['gAB_loss'].update_state(losses[0])
        metrics_dict['gBA_loss'].update_state(losses[1])
        metrics_dict['dA_loss'].update_state(losses[2])
        metrics_dict['dB_loss'].update_state(losses[3])
        metrics_dict['dA_accuracy'].update_state(losses[4])
        metrics_dict['dB_accuracy'].update_state(losses[5])

    def display_metrics(self, metrics_dict: Dict[str, Tensor], progress_bar: tqdm.tqdm) -> NoReturn:
        """Display training progress to the console
        
        Args:
            metrics_dict: dictionary of metrics
            progress_bar: tqdm progress bar
        """
        evaluated_metrics = {k: str(v.result().numpy())[:7] for k, v in metrics_dict.items()}
        progress_bar.set_postfix(**evaluated_metrics)
