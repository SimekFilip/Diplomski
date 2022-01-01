import typing as t

import numpy as np


def n_step_labeling(
    return_step: int,
    prices: t.Sequence,
    threshold: float,
    allow_long_position: bool,
    allow_short_position: bool,
):
    """
    TODO add documentation
    :param return_step:
    :param prices:
    :param threshold:
    :param allow_long_position
    :param allow_short_position
    :return:
    """
    prices = np.asarray(prices)
    returns = (prices[return_step:] - prices[:-return_step]) / prices[:-return_step]
    labels = np.zeros_like(prices,  dtype=int)
    if allow_long_position:
        labels[np.where(returns > threshold)[0]]  = 1
    
    if allow_short_position:
        labels[np.where(returns < -threshold)[0]] = -1
        
    i = 0
    while i < len(labels):
        if labels[i] != 0:
            labels[i:i+return_step] = labels[i]
            i+=return_step
        else:
            i+=1


    return labels