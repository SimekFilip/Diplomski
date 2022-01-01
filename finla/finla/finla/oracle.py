import ctypes as c
import typing as t

import numpy as np
from finla._core import Positions


def _get_lib_name():
    import sys

    lib_name = "liboracle_trading"
    if sys.platform.startswith("linux"):
        lib_name += ".so"
    elif sys.platform.startswith("darwin"):
        lib_name += ".dylib"
    else:
        raise Exception("Unsupported operating system: " + sys.platform)

    return lib_name


def _load_local_lib():
    import os
    basedir = os.path.abspath(os.path.dirname(__file__))
    libpath = os.path.join(basedir, _get_lib_name())
    return c.CDLL(libpath)


_LIB = None


def oracle_labeling(
    current_prices: t.Sequence,
    entry_prices: t.Sequence,
    long_entry_fee: float,
    long_exit_fee: float,
    short_entry_fee: float = float("inf"),
    short_exit_fee: float = float("inf"),
    start_positions: t.Optional[t.Sequence[Positions]] = None,
    allow_immediate_position_switch: bool = False,
    last_position: Positions = Positions.OUT
):
    global _LIB
    if _LIB is None:
        _LIB = _load_local_lib()

    if start_positions is None:
        start_positions = [Positions.OUT]

    n = len(current_prices)
    out = (c.c_short * len(current_prices))()

    _LIB.optimal_trading_positions(
        c.c_double(long_entry_fee), c.c_double(long_exit_fee),
        c.c_double(short_entry_fee), c.c_double(short_exit_fee),
        c.c_bool(allow_immediate_position_switch),

        len(start_positions), np.ctypeslib.as_ctypes(np.asarray(start_positions)),
        last_position.value,

        len(current_prices),
        np.ctypeslib.as_ctypes(np.asarray(current_prices)),
        np.ctypeslib.as_ctypes(np.asarray(entry_prices)),

        out
    )
    out = np.ctypeslib.as_array(out, shape=(n,))
    return out
