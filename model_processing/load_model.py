import json
from typing import Dict, NoReturn
from importlib import import_module

from cyclegan.base import BaseModel


def json2dict(json_path: str) -> Dict:
    """Read the json into a dictionary

    Args:
        json_path: path to the json

    Returns:
        dictionary of the json
    """
    with open(json_path, 'r') as f:
        contents = json.loads(f.read())

    return contents

def dict2json(dict: Dict, json_path: str) -> NoReturn:
    """Save dictionary as json

    Args:
        dict: dictionary to serialize
        json_path: path to the json
    """
    with open(json_path, 'w') as f:
        f.write(json.dumps(dict))

def import_model_class(model_type: str) -> BaseModel:
    """Load the model module dynamically from string

    Args:
        model_type: name of the model module

    Returns:
        instance of model loaded with config
    """
    model_module = import_module(f'models.{model_type}')
    model_class_name = getattr(model_module, 'MODEL_CLASS')
    model_class = getattr(model_module, model_class_name)
    return model_class

def construct_model(model_config: Dict) -> BaseModel:
    model_type = getattr(model_config, 'type')
    model_class = import_model_class(model_type)

    model_instance = model_class(model_config)

    return model_instance
