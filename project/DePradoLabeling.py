import pandas as pd
import numpy as np
from tqdm import tqdm


def get_Daily_Volatility(close, span0=20):
    df0 = close.pct_change()
    df0 = df0.ewm(span=span0).std()
    df0.dropna(inplace=True)
    return df0


def triple_barrier_labeling(
        price: pd.DataFrame,
        volatility_span: float = 20,
        upper_barrier_scaler: float = 2,
        bottom_barrier_scaler: float = 2,
        n_step: int = 5
):
    labels = []

    dv = get_Daily_Volatility(price, volatility_span)
    price = price.loc[dv.index]

    for i in tqdm(range(len(price) - n_step)):
        starting_price = price.iloc[i]
        upper_barrier_price = starting_price * (1 + upper_barrier_scaler * dv.iloc[i])
        bottom_barrier_price = starting_price * (1 - bottom_barrier_scaler * dv.iloc[i])
        look_ahead_price_window = price.iloc[i:i + n_step]

        upper_barrier_hit = (look_ahead_price_window >= upper_barrier_price).any()
        bottom_barrier_hit = (look_ahead_price_window <= bottom_barrier_price).any()

        if upper_barrier_hit and bottom_barrier_hit:
            argwhereupper = np.argwhere(look_ahead_price_window.values >= upper_barrier_price)[0]
            argwherebottom = np.argwhere(look_ahead_price_window.values <= bottom_barrier_price)[0]
            if argwherebottom < argwhereupper:
                labels.append(1)
            else:
                labels.append(0)
        elif upper_barrier_hit:
            labels.append(1)
        elif bottom_barrier_hit:
            labels.append(0)
        else:
            rtn = (look_ahead_price_window.values[-1] - starting_price) / starting_price
            labels.append(1 if rtn >= 0 else 0)

    return pd.Series(labels, index=dv.index[:-n_step])
