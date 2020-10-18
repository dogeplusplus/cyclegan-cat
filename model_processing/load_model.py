import json
import yaml
from typing import Dict, NoReturn
from importlib import import_module

from bunch import Bunch
from tensorflow.keras.models import Model

def yaml2namespace(yaml_path: str) -> Bunch:
    """Load the yaml file and convert it into a namespace

    Args:
        yaml_path: path to the yaml file

    Returns:
        Namespace config for the model
    """
    with open(yaml_path, 'r') as f:
        model_config_dict = yaml.load(f, yaml.FullLoader)

    model_config = Bunch(model_config_dict)
    return model_config

def namespace2yaml(yaml_path: str, namespace: Bunch) -> NoReturn:
    """Save the YAML file on disk

    Args:
        yaml_path: path to save the yaml file
        namespace: YAML to serialize
    """
    with open(yaml_path, 'w') as f:
        yaml.dump(namespace, f)

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

def json2namespace(json_path: str) -> Bunch:
    """Read the json into a namespace

    Args:
        json_path: path to the json

    Returns:
        dictionary of the json
    """
    dictionary = json2dict(json_path)
    return Bunch(dictionary)


def import_model_class(model_type: str) -> Model:
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

def construct_model(model_config: Dict) -> Model:
    model_type = getattr(model_config, 'type')
    model_class = import_model_class(model_type)

    model_instance = model_class(model_config)

    return model_instance

if __name__ == "__main__":
    x = yaml2namespace("configs/cycle.yaml")
    print(x)