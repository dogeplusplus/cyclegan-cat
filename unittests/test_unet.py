from copy import deepcopy

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


def test_incomplete_unet_model_config(config):
    # Check that unet generator cannot be initialized with an incomplete config
    MANDATORY_FIELDS = [
        'filters',
        'kernels',
        'expansion',
        'normalization',
        'dropout',
        'output_channels',
        'final_activation'
    ]
    for field in MANDATORY_FIELDS:
        custom_config = deepcopy(config)
        del custom_config[field]
        with pytest.raises(KeyError):
            unet_generator(custom_config)


def test_incomplete_strided_model_config(config):
    # Check that strided unet cannot be initialized with an incomplete config
    MANDATORY_FIELDS = [
        'filters',
        'kernels',
        'normalization',
        'output_channels',
        'final_activation'
    ]
    for field in MANDATORY_FIELDS:
        custom_config = deepcopy(config)
        del custom_config[field]
        with pytest.raises(KeyError):
            strided_unet(custom_config)
