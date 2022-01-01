from enum import Enum

import numpy as np


class Positions(int, Enum):
    OUT = 0
    LONG = 1
    SHORT = -1
    UNKNOWN = -2


def threshold_returns(
    returns: np.ndarray,
    threshold: float,
    allow_shorting: bool
):
    thresholded_returns = np.zeros(len(returns), dtype=int)
    thresholded_returns[returns > threshold] = Positions.LONG
    if allow_shorting:
        thresholded_returns[returns < -threshold] = Positions.SHORT
    return thresholded_returns


def threshold_returns_to_labels(
    thresholded_returns: np.ndarray,
    return_step: int
):
    labels = np.zeros(len(thresholded_returns) + return_step, dtype=np.int)
    for i, label in enumerate(thresholded_returns):
        if label != 0:
            labels[i: i + return_step] = label
    return labels
