from typing import Dict
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Optimizer
from adabelief_tf.AdaBelief_tf import AdaBeliefOptimizer

def get_optimizer(optimizer_config: Dict) -> Optimizer:
    """Return the correct optimizer
    Args:
        name: name of the optimizer
    Returns:
        Tensorflow optimizer object.
    """
    learning_rate = optimizer_config['learning_rate']
    name = optimizer_config['name']
    if name == 'adam':
        optimizer = Adam(learning_rate=learning_rate, beta_1=optimizer_config['beta_1'])
    elif name == 'rmsprop':
        optimizer =  RMSprop(learning_rate=learning_rate)
    elif name == 'sgd':
        optimizer =  SGD(learning_rate=learning_rate)
    elif name == 'adabelief':
        optimizer = AdaBeliefOptimizer(learning_rate)
    return optimizer
