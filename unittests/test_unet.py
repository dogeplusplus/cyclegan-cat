import pytest
import numpy as np

from cyclegan.unet import unet_generator, strided_unet

@pytest.fixture
def config():
    model_config = dict(
        type='strided_unet',
        filters=[8, 8, 8],
        kernels=[4, 4, 4],
        output_channels=3,
        expansion='upsample',
        normalization='instancenorm',
        dropout=False,
        final_activation='tanh'
    )
    return model_config

@pytest.fixture
def mock_image():
    image = np.ones((1, 128, 128, 3))
    return image

def test_pooled_unet(config, mock_image):
    model = unet_generator(config)
    prediction = model(mock_image)

    assert mock_image.shape == prediction.shape


def test_strided_unet(config, mock_image):
    model = strided_unet(config)
    prediction = model(mock_image)

    assert mock_image.shape == prediction.shape
