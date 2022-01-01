#ifndef OPT_TRADING_H
#define OPT_TRADING_H

#include "stdbool.h"

typedef short pos;
#define SHORT_POS ((short)-1)
#define OUT_POS   ((short) 0)
#define LONG_POS  ((short) 1)

int optimal_trading_positions(
    double long_entry_fee, double long_exit_fee,
    double short_entry_fee, double short_exit_fee,
    bool allow_immediate_position_switch,

    size_t n_start_positions, const pos *start_positions,
    pos last_position,

    size_t n_prices,
    const double *current_prices,
    const double *enter_prices,

    pos *output
);

#endif /* OPT_TRADING_H */