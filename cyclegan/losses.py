import tensorflow as tf
from tensorflow import Tensor


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


def get_loss_obj(loss: str):
    """Return the type of loss function based on the name

    Args:
        loss: abbreviation of loss function to use

    Returns:
        Loss object to be passed into the cycle gan losses.
    """
    LOSS_OBJ_MAPS = {
        "mse": tf.keras.losses.MeanSquaredError(),
        "mae": tf.keras.losses.MeanAbsoluteError(),
        "bce": tf.keras.losses.BinaryCrossentropy(from_logits=True)
    }
    return LOSS_OBJ_MAPS[loss]
