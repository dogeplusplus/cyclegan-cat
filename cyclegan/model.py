import tqdm
import logging
import numpy as np
import tensorflow as tf

from bunch import Bunch
from typing import Dict
from os.path import join
from tensorflow import Tensor
from tensorflow.keras.models import Model

from cyclegan.optimizers import get_optimizer
from model_processing.load_model import namespace2yaml
from cyclegan.unet import unet_generator, strided_unet
from cyclegan.resnet import simple_discriminator, resnet_generator
from cyclegan.losses import calc_cycle_loss, identity_loss, discriminator_loss, generator_loss, get_loss_obj

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_model(config: Dict) -> Model:
    chosen_type = config["type"]

    MODEL_FUNCTION = [
        simple_discriminator,
        resnet_generator,
        unet_generator,
        strided_unet,
    ]
    model_type_map = {model.__name__: model for model in MODEL_FUNCTION}
    return model_type_map[chosen_type](config)


def accuracy(real, fake):
    predictions = tf.cast(tf.concat([real, fake], axis=0) > 0.5, tf.float32)
    labels = tf.concat([tf.ones_like(real), tf.zeros_like(fake, tf.float32)], axis=0)
    acc = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
    return acc


class CycleGan(Model):
    def __init__(self, model_config: Bunch, train_config: Bunch):
        super(CycleGan, self).__init__()
        self.model_config = model_config
        self.model_folder = join(self.model_config.location, self.model_config.name)
        self.train_summaries = tf.summary.create_file_writer(
            join(self.model_folder, "train"))
        self.val_summaries = tf.summary.create_file_writer(
            join(self.model_folder, "validation")
        )
        self.train_config = train_config
        self.g_AB_optimizer = get_optimizer(self.train_config.g_opt)
        self.g_BA_optimizer = get_optimizer(self.train_config.g_opt)
        self.d_A_optimizer = get_optimizer(self.train_config.d_opt)
        self.d_B_optimizer = get_optimizer(self.train_config.d_opt)

        self.loss_weights = self.model_config.loss_weights
        if self.model_config.new:
            self.build_models()
            self.model_config.new = False
            namespace2yaml(join(self.model_folder, "model_config.yaml"), self.model_config)
        else:
            self.load_optimizer(self.g_AB, self.g_AB_optimizer, join(self.model_folder, "g_AB_optimizer.npy"))
            self.load_optimizer(self.g_BA, self.g_BA_optimizer, join(self.model_folder, "g_BA_optimizer.npy"))
            self.load_optimizer(self.d_A, self.d_A_optimizer, join(self.model_folder, "d_A_optimizer.npy"))
            self.load_optimizer(self.d_B, self.d_B_optimizer, join(self.model_folder, "d_B_optimizer.npy"))
            self.load_model()

    def build_models(self):
        gen_config = self.model_config.generator
        disc_config = self.model_config.discriminator

        self.g_AB = create_model(gen_config)
        self.g_BA = create_model(gen_config)
        self.d_A = create_model(disc_config)
        self.d_B = create_model(disc_config)

        self.loss_obj = get_loss_obj(self.model_config.loss)

    @tf.function
    def validate_step(self, real_a: Tensor, real_b: Tensor, training: bool = False) -> Dict:
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

        gAB_loss = generator_loss(disc_fake_b, self.loss_obj, self.loss_weights["generator"])
        gBA_loss = generator_loss(disc_fake_a, self.loss_obj, self.loss_weights["generator"])

        total_cycle_loss = calc_cycle_loss(
            real_a, cycled_a, self.loss_weights["cycle"]) + calc_cycle_loss(
                real_b, cycled_b, self.loss_weights["cycle"])

        total_gAB_loss = gAB_loss + total_cycle_loss + identity_loss(real_b, same_b,
                                                                     self.loss_weights["identity"])
        total_gBA_loss = gBA_loss + total_cycle_loss + identity_loss(real_a, same_a,
                                                                     self.loss_weights["identity"])

        da_loss = discriminator_loss(disc_real_a, disc_fake_a, self.loss_obj, self.loss_weights["discriminator"])
        db_loss = discriminator_loss(disc_real_b, disc_fake_b, self.loss_obj, self.loss_weights["discriminator"])

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

        g_AB_gradients = tape.gradient(metrics["gAB_loss"], self.g_AB.trainable_variables)
        g_BA_gradients = tape.gradient(metrics["gBA_loss"], self.g_BA.trainable_variables)

        da_gradients = tape.gradient(metrics["dA_loss"], self.d_A.trainable_variables)
        db_gradients = tape.gradient(metrics["dB_loss"], self.d_B.trainable_variables)

        self.g_AB_optimizer.apply_gradients(zip(g_AB_gradients, self.g_AB.trainable_variables))
        self.g_BA_optimizer.apply_gradients(zip(g_BA_gradients, self.g_BA.trainable_variables))

        self.d_A_optimizer.apply_gradients(zip(da_gradients, self.d_A.trainable_variables))
        self.d_B_optimizer.apply_gradients(zip(db_gradients, self.d_B.trainable_variables))
        return metrics

    def train(self, train_dataset: tf.data.Dataset,
              validation_dataset: tf.data.Dataset):

        batch_size = self.train_config.batch_size
        epochs = self.train_config.epochs
        save_images_every = self.train_config.summary["images"]
        tensorboard_samples = self.train_config.summary["samples"]
        save_model_every = self.train_config.summary["model"]
        sample_images = []

        metric_names = [
            "dA_loss",
            "dB_loss",
            "gAB_loss",
            "gBA_loss",
            "dA_acc",
            "dB_acc",
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
            tf.summary.image(name="A", data=tf.add(a_samples, 1) / 2, step=0, max_outputs=tensorboard_samples)
            tf.summary.image(name="B", data=tf.add(b_samples, 1) / 2, step=0, max_outputs=tensorboard_samples)

        train_dataset = train_dataset.batch(batch_size)
        validation_dataset = validation_dataset.batch(batch_size)
        training_size = sum(1 for _ in train_dataset)
        validation_size = sum(1 for _ in validation_dataset)
        desc = "Epoch {} training"
        val_desc = "Epoch {} validation"
        for e in range(epochs):
            train_bar = tqdm.tqdm(train_dataset, desc=desc.format(e + 1), ncols=200, total=training_size)
            for (images_a, images_b) in train_bar:
                losses = self.train_step(images_a, images_b)
                self.update_metrics(train_metrics_dict, losses)
                self.display_metrics(train_metrics_dict, train_bar)

            self.write_summaries(self.train_summaries, e, train_metrics_dict)
            if e % save_images_every == 0:
                self.write_images(e, a_samples, b_samples, tensorboard_samples)

            val_bar = tqdm.tqdm(validation_dataset, desc=val_desc.format(e + 1), ncols=200, total=validation_size)
            for (images_a, images_b) in val_bar:
                losses = self.validate_step(images_a, images_b, training=False)
                self.update_metrics(validation_metrics_dict, losses)
                self.display_metrics(validation_metrics_dict, val_bar)
            self.write_summaries(self.val_summaries, e, validation_metrics_dict)

            if e % save_model_every == 0:
                self.save_model()

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

    def write_images(self, epoch: int, a_samples: Tensor, b_samples: Tensor, num_samples: int):
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
            tf.summary.image(name="A2B_predictions", data=tf.add(prediction_ab, 1) / 2,
                             step=epoch,
                             max_outputs=num_samples)
            tf.summary.image(name="B2A_predictions", data=tf.add(prediction_ba, 1) / 2,
                             step=epoch,
                             max_outputs=num_samples)

    def update_metrics(self, metrics_dict: Dict[str, Tensor], metrics: Dict):
        """Update the metrics dictionary with values from the training step

        Args:
            metrics_dict: dictionary of metrics
            metrics: loss values from the training batch
        """
        for name in metrics_dict.keys():
            metrics_dict[name].update_state(metrics[name])

    def display_metrics(self, metrics_dict: Dict[str, Tensor], progress_bar: tqdm.tqdm):
        """Display training progress to the console

        Args:
            metrics_dict: dictionary of metrics
            progress_bar: tqdm progress bar
        """
        evaluated_metrics = {k: str(v.result().numpy())[:7] for k, v in metrics_dict.items()}
        progress_bar.set_postfix(**evaluated_metrics)

    def save_model(self):
        save = lambda x: tf.keras.models.save_model(getattr(self, x), join(self.model_folder, x))
        save("d_A")
        save("d_B")
        save("g_AB")
        save("g_BA")

        save_optimizer = lambda optimizer: np.save(join(self.model_folder, f"{optimizer}.npy"),
                getattr(self, optimizer).get_weights())
        save_optimizer("g_AB_optimizer")
        save_optimizer("g_BA_optimizer")
        save_optimizer("d_A_optimizer")
        save_optimizer("d_B_optimizer")

    def load_model(self):
        load = lambda x: tf.saved_model.load(join(self.model_folder, x))
        self.d_A = load("d_A")
        self.d_B = load("d_B")
        self.g_AB = load("g_AB")
        self.g_BA = load("g_BA")

    def load_optimizer(self, model, optimizer, optimizer_path):
        weights = np.load(optimizer_path, allow_pickle=True)
        grad_vars = model.trainable_variables
        zero_grads = [tf.zeros_like(w) for w in grad_vars]

        optimizer.apply_gradients(zip(zero_grads, grad_vars))
        optimizer.set_weights(weights)





