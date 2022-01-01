
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "oracle_trading.h"

// Cardinalities
#define N_POSITIONS     (3)
#define N_TRANSITIONS   (9)

// Transitions
#define OUT_TO_OUT      (0)
#define LONG_TO_OUT     (1)
#define SHORT_TO_OUT    (2)
#define OUT_TO_LONG     (3)
#define LONG_TO_LONG    (4)
#define SHORT_TO_LONG   (5)
#define OUT_TO_SHORT    (6)
#define LONG_TO_SHORT   (7)
#define SHORT_TO_SHORT  (8)

// Positions
#define OUT_IDX             (0)   // 0000000
#define LONG_IDX            (1)   // 0000001
#define SHORT_IDX           (2)   // 0000010

#define max(x, y)            (x >= y ? x : y)
#define max3(x, y, z)        (x >= y ? max(x, z) : max(y, z))

#define NEG_INFINITY (-INFINITY)

#define MEMORY_ERROR             (-1)
#define ILLEGAL_POSITIONS_PASSED (-2)
#define ILLEGAL_PRICE            (-3)

#define REQUIRED_PRECISION (1E-7)

// ********************** //
// Function declarations  //
// ********************** //

static inline void rev_argsort3(const double *arr, short *out);

static inline unsigned short determine_transition(pos from, pos to);

static inline unsigned short position_to_position_idx(pos pos);

static int initialize_equity(
    size_t n_start_positions,
    const pos *start_positions,

    double *equity
);

static int initialize_transitions(
    double long_entry_fee, double long_exit_fee,
    double short_entry_fee, double short_exit_fee,
    bool allow_immediate_position_switch,
    size_t n_start_positions,
    size_t n_prices,
    const double *prices,
    const double *enter_prices,

    double *transitions
);

static void forward_pass(
    size_t n_prices,
    size_t n_start_positions,
    const double *transitions,

    double *equity
);

static void backward_pass(
    pos last_position,
    size_t n_prices,
    size_t n_start_positions,
    const double *equity,
    const double *transitions,
    pos *output
);

static void fill_start_positions(size_t n_start_position, const pos *start_positions, pos *output);

static void convert_position_idx_to_position_labels(size_t n_prices, pos *output);

// *************************************************//
// Implementation of the optimal trading algorithm  //
// *************************************************//

int optimal_trading_positions(
    const double long_entry_fee, const double long_exit_fee,
    const double short_entry_fee, const double short_exit_fee,
    const bool allow_immediate_position_switch,

    const size_t n_start_positions, const pos *start_positions,
    const pos last_position,

    const size_t n_prices,
    const double *prices,
    const double *enter_prices,

    pos *output // this gets modified
) {
  // Initialize equity array.
  // The elements of the equity array are organized as follows:
  //    `close_0_eq, long_0_eq, short_0_eq, close_1_eq, ....`
  // The reason for this kind of organization is that we use the values in
  // this order thus this provides us with lower cache miss rate and better
  // performance.
  double *equity = malloc(sizeof(double) * N_POSITIONS * (n_prices - n_start_positions + 1));
  if (equity == NULL) {
    return MEMORY_ERROR;
  }
  int equity_init_res = initialize_equity(n_start_positions, start_positions, equity);
  if (equity_init_res != 0) {
    free(equity);
    return equity_init_res;
  }

  // Initialize transitions array.
  double *transitions = malloc(sizeof(double) * N_TRANSITIONS * (n_prices - n_start_positions));
  if (transitions == NULL) {
    free(equity);
    return MEMORY_ERROR;
  }
  int transition_init_res = initialize_transitions(
      long_entry_fee, long_exit_fee,
      short_entry_fee, short_exit_fee,
      allow_immediate_position_switch,
      n_start_positions,
      n_prices, prices, enter_prices,

      transitions
  );
  if (transition_init_res != 0) {
    free(equity);
    free(transitions);
    return transition_init_res;
  }

  forward_pass(n_prices, n_start_positions, transitions, equity);
  backward_pass(last_position, n_prices, n_start_positions, equity, transitions, output);
  fill_start_positions(n_start_positions, start_positions, output);
  convert_position_idx_to_position_labels(n_prices, output);

  // Free resources
//  printf("Before equity free.\n");
  free(equity);
  //printf("Before transitions free\n");
  free(transitions);
//  printf("After transitions free\n");

  return 0;
}

static void fill_start_positions(
    const size_t n_start_positions,
    const pos *start_positions,
    pos *output
) {
  size_t i;
  for (i = 0; i < n_start_positions; i++) {
    *output++ = *start_positions++;
  }
//  printf("fill_start_pos ok\n");
}

static void forward_pass(
    const size_t n_prices, const size_t n_start_positions,
    const double *transitions,

    double *equity
) {
  double *last_equity_pointer = equity + (N_POSITIONS * (n_prices - n_start_positions + 1));

  double *equity_prev_out_pt = equity; // + ((n_start_positions - 1) * N_POSITIONS);
  double *equity_prev_long_pt = equity_prev_out_pt + 1;
  double *equity_prev_short_pt = equity_prev_long_pt + 1;
  double *current_equity = equity_prev_short_pt + 1;

  const double *transition_pt = transitions;

  size_t i, j = 0;
  double o, l, s;

  for (i = n_start_positions; i < n_prices; i++, j++) {
    // We are starting from `n_start_positions` since `equity` states have already been
    // set to proper values in initialization state.

    assert (current_equity < last_equity_pointer);

    // Update OUT state equity and move `current_equity` and `transition_pt` pointers.
    o = *equity_prev_out_pt + *transition_pt++;
    l = *equity_prev_long_pt + *transition_pt++;
    s = *equity_prev_short_pt + *transition_pt++;
    *current_equity++ = max3(o, l, s);

    // Update LONG state equity and move `current_equity` and `transition_pt` pointers.
    o = *equity_prev_out_pt + *transition_pt++;
    l = *equity_prev_long_pt + *transition_pt++;
    s = *equity_prev_short_pt + *transition_pt++;
    *current_equity++ = max3(o, l, s);

    // Update SHORT state equity and move `current_equity` and `transition_pt` pointers.
    o = *equity_prev_out_pt + *transition_pt++;
    l = *equity_prev_long_pt + *transition_pt++;
    s = *equity_prev_short_pt + *transition_pt++;
    *current_equity++ = max3(o, l, s);

    // Update previous equity pointers to point.
    equity_prev_out_pt = equity_prev_short_pt + 1;
    equity_prev_long_pt = equity_prev_out_pt + 1;
    equity_prev_short_pt = equity_prev_long_pt + 1;
  }
}

static void backward_pass(
    const pos last_position,
    const size_t n_prices,
    const size_t n_start_positions,

    const double *equity,
    const double *transitions,

    pos *output
) {
  pos *initial_output_pt = &output[0];

  // Output will be 0, 1, 2 and at the end we will convert it to the -1, 0, 1 by converting all 2 -> -1.
  pos *output_pt = &output[n_prices - 1];

  // Find out what the last position is.
  const double *equity_pt = &equity[N_POSITIONS * (n_prices - n_start_positions + 1) - 1]; // points to short_equity
  double current_state_equity;

  // Set last output and equity pointer.
  unsigned short last_pos_idx = position_to_position_idx(last_position);
  *output_pt-- = (short) last_pos_idx;
  current_state_equity = *(equity_pt + last_pos_idx - 2);

  // Points to short
  // -1 points to long
  // -2 points to out
  // -3 points to previous short
  equity_pt -= 3;

  // Stack allocate temp arrays
  double equity_diff[3], prev_equity[3];
  short arg_arr[3]; // Can be either 0, 1, 2
  size_t i;

  // `transitions` array has the size: (n_prices - 1) * N_TRANSITIONS
  // however since we are not including the start positions then it has (n_start_positions - 1) values less.
  // Therefore, the whole size of transitions array is (n_prices - 1 - (n_start_positions - 1)) * N_TRANSITIONS
  // Since we want transitions_pt to point to the last out->out transition, we need to subtract N_TRANSITIONS
  // from the length of the transitions array. Hence, the address we are looking for is at:
  //    `(n_prices - 1 - (n_start_positions - 1)) * N_TRANSITIONS - N_TRANSITIONS`, which can be further simplified into:
  // == `(n_prices - 1 - (n_start_positions - 1) - 1) * N_TRANSITIONS'
  // == `(n_prices - n_start_positions - 1) * N_TRANSITIONS`
  const double *transitions_pt = &transitions[(n_prices - n_start_positions - 1) * N_TRANSITIONS];
  pos current_position;
  for (i = n_start_positions; i < n_prices; i++) {
    assert(initial_output_pt <= output_pt);
    current_position = *(output_pt + 1);

    // Copy values into an array and decrement value of the pointer.
    // In last iteration this will point to the memory address that is outside the
    // allocated memory, but we will not use (dereference) that memory address and
    // therefore this should be fine.
    // @formatter:off
    prev_equity[SHORT_IDX] = *equity_pt--;
    prev_equity[LONG_IDX]  = *equity_pt--;
    prev_equity[OUT_IDX]   = *equity_pt--;

    // Compute difference between current equity and previous equity + transaction cost
    equity_diff[OUT_IDX]   = current_state_equity - *(transitions_pt + determine_transition(OUT_IDX, current_position))   - prev_equity[OUT_IDX];
    equity_diff[LONG_IDX]  = current_state_equity - *(transitions_pt + determine_transition(LONG_IDX, current_position))  - prev_equity[LONG_IDX];
    equity_diff[SHORT_IDX] = current_state_equity - *(transitions_pt + determine_transition(SHORT_IDX, current_position)) - prev_equity[SHORT_IDX];

    transitions_pt -= N_TRANSITIONS; // Update transition pointer
    // @formatter:on

    // Differences that equate to 0.0 are possible paths of the algorithm.
    // And next position will be the one which has the highest previous equity and difference 0.0
    rev_argsort3(&prev_equity[0], &arg_arr[0]);
    if (fabs(equity_diff[arg_arr[0]]) <= REQUIRED_PRECISION) {
      *output_pt-- = arg_arr[0];
      current_state_equity = *(equity_pt + arg_arr[0] + 1);
      continue;
    }

    if (fabs(equity_diff[arg_arr[1]]) <= REQUIRED_PRECISION) {
      *output_pt-- = arg_arr[1];
      current_state_equity = *(equity_pt + arg_arr[1] + 1);
      continue;
    }

    if (fabs(equity_diff[arg_arr[2]]) <= REQUIRED_PRECISION) {
      *output_pt-- = arg_arr[2];
      current_state_equity = *(equity_pt + arg_arr[2] + 1);
      continue;
    }

    // unreachable unless some error occurred, hence we want to stop execution here because this is a bug.
    fprintf(stderr, "Reached logically unreachable state. This indicates a bug is present.\n");
    exit(-1);
  }
}

static void convert_position_idx_to_position_labels(
    const size_t n_prices,
    pos *output
) {
  // Since output array contains values {0, 1, 2}, and we want it to contain
  // values {-1, 0, 1} for short, out, long positions we need to convert
  // 2 -> -1.
  // Previous method (`fill_start_positions`) might input -1 values, but
  // since we are only converting 2->-1, that is not a problem.
  pos *output_pt = output;
  size_t i;
  for (i = 0; i < n_prices; i++) {
    if (*output_pt == SHORT_IDX) {
      *output_pt = SHORT_POS;
    }
    output_pt++;
  }
}

static inline void rev_argsort3(const double *arr, short *out) {
  if (arr[0] >= arr[1]) {
    if (arr[0] >= arr[2]) {
      out[0] = 0;
      if (arr[1] >= arr[2]) {
        // 0, 1, 2
        out[1] = 1;
        out[2] = 2;
      } else {
        // 0, 2, 1
        out[1] = 2;
        out[2] = 1;
      }
    } else {
      // 2, 0, 1
      out[0] = 2;
      out[1] = 0;
      out[2] = 1;
    }
  } else {
    if (arr[1] >= arr[2]) {
      out[0] = 1;
      if (arr[0] >= arr[2]) {
        // 1, 0, 2
        out[1] = 0;
        out[2] = 2;
      } else {
        // 1, 2, 0
        out[1] = 2;
        out[2] = 0;
      }
    } else {
      // 2, 1, 0
      out[0] = 2;
      out[1] = 1;
      out[2] = 0;
    }
  }
}

static inline unsigned short determine_transition(pos from, pos to) {
  // TODO(fredi): this can maybe be more efficiently implemented using bit arithmetics.
  if (from == OUT_IDX) {
    switch (to) {
      case OUT_IDX: return OUT_TO_OUT;
      case LONG_IDX: return OUT_TO_LONG;
      default: return OUT_TO_SHORT;
    }
  }
  if (from == LONG_IDX) {
    switch (to) {
      case OUT_IDX:return LONG_TO_OUT;
      case LONG_IDX:return LONG_TO_LONG;
      default:return LONG_TO_SHORT;
    }
  }

  switch (to) {
    case OUT_IDX:return SHORT_TO_OUT;
    case LONG_IDX:return SHORT_TO_LONG;
    default:return SHORT_TO_SHORT;
  }
}

static inline unsigned short position_to_position_idx(const pos pos) {
  switch (pos) {
    case OUT_POS: return OUT_IDX;
    case LONG_POS: return LONG_IDX;
    case SHORT_POS: return SHORT_IDX;
    default: {
      fprintf(stderr, "Unknown position value: %i", pos);
      exit(-1);
    }
  }
}

static int initialize_equity(
    size_t n_start_positions, const pos *start_positions,
    double *equity
) {
  // First n_start_positions * N_POSITIONS elements of equity array must be
  // filled with value that will be used to determine which positions should
  // be taken to obtain final results in backtracking part of the algorithm.
  // Backtracking is equivalent to iterating backwards in equity array and
  // determining from which previous equity state a transition was made. This
  // implies that `equity[i-1] + transition[i] == equity[i]` and therefore,
  // the initialization of equity and transition arrays are dependent on each
  // other. Hence, for all timestamps and all positions in `start_position` we
  // set equity to be equal to `initial_equity`, and for all other positions
  // we set the `equity` to `-INFINITY`. On the other hand, all `transitions`
  // which are made between to start position are set to `0.0`, and all others
  // are set to `-INFINITY`.

  switch (start_positions[n_start_positions - 1]) {
    case SHORT_POS: {
      *equity++ = NEG_INFINITY;      // CLOSE_IDX
      *equity++ = NEG_INFINITY;      // LONG_IDX
      *equity++ = 0;                 // SHORT_IDX
      break;
    }
    case OUT_POS: {
      *equity++ = 0;                 // CLOSE_IDX
      *equity++ = NEG_INFINITY;      // LONG_IDX
      *equity++ = NEG_INFINITY;      // SHORT_IDX
      break;
    }
    case LONG_POS: {
      *equity++ = NEG_INFINITY;      // CLOSE_IDX
      *equity++ = 0;                 // LONG_IDX
      *equity++ = NEG_INFINITY;      // SHORT_IDX
      break;
    }
    default: {
      return ILLEGAL_POSITIONS_PASSED;
    }

  }
  return 0;
}

static int initialize_transitions(
    double long_entry_fee, double long_exit_fee,
    double short_entry_fee, double short_exit_fee,
    bool allow_immediate_position_switch,

    size_t n_start_positions,

    size_t n_prices,
    const double *prices,
    const double *enter_prices,

    double *transitions
) {
  // NOTE: since we are initializing equity array, we do not need to check whether `start_positions`
  // is valid or not (e.g. values in range {-1, 0, 1}), but we do need to check if prices are valid, i.e.
  // not equal to nan and in range `p \in (0.0, inf)` (where `()` braces are exclusive).
  double *last_transition_element = &transitions[N_TRANSITIONS * (n_prices - n_start_positions)];

  size_t i = n_start_positions;
  double *element = &transitions[0];

#define invalid(x) ((bool)isnan(x) || ((x) <= 0.0))

  if (invalid(prices[i - 1]) || invalid(enter_prices[i - 1])) {
    return ILLEGAL_PRICE;
  }
  // Now all other transitions are updated with the corresponding values.
  double price_diff, current_price, current_entry_price;
  for (; i < n_prices; i++) {
    current_price = prices[i];
    current_entry_price = enter_prices[i];
    if (invalid(current_price) || invalid(current_entry_price)) {
      return ILLEGAL_PRICE;
    }
    price_diff = current_price - prices[i - 1];

    // Update elements in order from out->out to short->short. See index definitions in the top of the file.
    // @formatter:off

    assert(last_transition_element > element);
    *element++ = 0.0;
    *element++ = -1.0 * current_entry_price * long_exit_fee;
    *element++ = -1.0 * current_entry_price * short_exit_fee;
    *element++ = -1.0 * current_entry_price * long_entry_fee;
    *element++ = price_diff;
    *element++ = -1.0 * current_entry_price * (allow_immediate_position_switch ? short_exit_fee * long_entry_fee : INFINITY);
    *element++ = -1.0 * current_entry_price * short_entry_fee;
    *element++ = -1.0 * current_entry_price * (allow_immediate_position_switch ? long_exit_fee + short_entry_fee : INFINITY);
    *element++ = -1.0 * price_diff;
    // @formatter:on
    assert(last_transition_element >= element);
  }
  return 0;
}