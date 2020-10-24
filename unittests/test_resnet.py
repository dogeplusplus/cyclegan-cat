import pytest
import numpy as np

from cyclegan.resnet import resnet_generator, ReflectionPadding2D, simple_discriminator

@pytest.fixture
def config():
    model_config = dict(filters=16)
    return model_config

@pytest.fixture
def mock_image():
    return np.ones((1, 128, 128, 3))

def test_resnet(config, mock_image):
    model = resnet_generator(config)
    prediction = model(mock_image)

    assert prediction.shape == mock_image.shape


def test_reflection_padding():
    x = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2]
    ])[np.newaxis, ..., np.newaxis]
    actual = ReflectionPadding2D()(x).numpy()

    expected = np.array([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1]
    ])[np.newaxis, ..., np.newaxis]

    assert np.array_equal(expected, actual)


def test_simple_discriminator(mock_image):
    model_config = dict(
        filters=[8, 16, 32],
        kernels=[4, 4, 4],
        normalization='instancenorm'
    )
    model = simple_discriminator(model_config)
    prediction = model(mock_image).numpy()
    assert prediction.shape == (1, 16, 16, 1)
