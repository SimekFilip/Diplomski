import typing as t

import numpy as np
from finla._core import threshold_returns, threshold_returns_to_labels


def ewm_labeling(
    prices: t.Sequence,
    return_step: int,
    alpha: float,
    lookahead_window: int,
    threshold: float,
    allow_shorting: bool
):
    kernel = np.arange(lookahead_window)
    kernel = alpha ** kernel
    kernel = kernel / np.sum(kernel)
    prices = np.asarray(prices)
    returns = (prices[return_step:] - prices[:-return_step]) / prices[:-return_step]
    ewm_returns = np.correlate(returns, kernel, "valid")

    return threshold_returns_to_labels(
        threshold_returns(ewm_returns, threshold, allow_shorting),
        return_step
    )


if __name__ == '__main__':
    x = np.arange(20)
    y = np.ones(3)
    print(np.correlate(x, y, "valid"))
    print(0.5 ** np.arange(10))
