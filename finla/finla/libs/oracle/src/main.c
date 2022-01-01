#include <stdio.h>
#include <stdlib.h>

#include "oracle_trading.h"

int main(int argc, char* argv[]) {
  const static int n_prices = 300;
  double prices[] = {
      228.885, 228.905, 228.929, 228.920, 228.914, 228.933, 228.930, 228.940, 228.921, 228.927,
      228.960, 228.984, 229.010, 229.000, 229.014, 228.993, 229.016, 229.013, 229.016, 229.000,
      229.324, 229.320, 229.325, 229.357, 229.366, 229.361, 229.325, 229.325, 229.290, 229.261,
      229.242, 229.238, 229.318, 229.311, 229.278, 229.307, 229.251, 229.221, 229.244, 229.273,
      229.280, 229.254, 229.282, 229.287, 229.244, 229.218, 229.203, 229.157, 229.220, 229.215,
      229.319, 229.312, 229.349, 229.392, 229.354, 229.332, 229.369, 229.320, 229.310, 229.292,
      229.260, 229.278, 229.270, 229.255, 229.280, 229.290, 229.301, 229.251, 229.236, 229.243,
      229.204, 229.216, 229.228, 229.225, 229.234, 229.263, 229.266, 229.253, 229.310, 229.323,
      229.308, 229.311, 229.345, 229.318, 229.316, 229.294, 229.323, 229.304, 229.308, 229.301,
      228.922, 228.895, 228.890, 228.863, 228.863, 228.825, 228.779, 228.829, 228.844, 228.880,
      228.885, 228.905, 228.929, 228.920, 228.914, 228.933, 228.930, 228.940, 228.921, 228.927,
      228.960, 228.984, 229.010, 229.000, 229.014, 228.993, 229.016, 229.013, 229.016, 229.000,
      229.324, 229.320, 229.325, 229.357, 229.366, 229.361, 229.325, 229.325, 229.290, 229.261,
      229.242, 229.238, 229.318, 229.311, 229.278, 229.307, 229.251, 229.221, 229.244, 229.273,
      229.280, 229.254, 229.282, 229.287, 229.244, 229.218, 229.203, 229.157, 229.220, 229.215,
      229.319, 229.312, 229.349, 229.392, 229.354, 229.332, 229.369, 229.320, 229.310, 229.292,
      229.260, 229.278, 229.270, 229.255, 229.280, 229.290, 229.301, 229.251, 229.236, 229.243,
      229.204, 229.216, 229.228, 229.225, 229.234, 229.263, 229.266, 229.253, 229.310, 229.323,
      229.308, 229.311, 229.345, 229.318, 229.316, 229.294, 229.323, 229.304, 229.308, 229.301,
      228.922, 228.895, 228.890, 228.863, 228.863, 228.825, 228.779, 228.829, 228.844, 228.880,
      228.885, 228.905, 228.929, 228.920, 228.914, 228.933, 228.930, 228.940, 228.921, 228.927,
      228.960, 228.984, 229.010, 229.000, 229.014, 228.993, 229.016, 229.013, 229.016, 229.000,
      229.324, 229.320, 229.325, 229.357, 229.366, 229.361, 229.325, 229.325, 229.290, 229.261,
      229.242, 229.238, 229.318, 229.311, 229.278, 229.307, 229.251, 229.221, 229.244, 229.273,
      229.280, 229.254, 229.282, 229.287, 229.244, 229.218, 229.203, 229.157, 229.220, 229.215,
      229.319, 229.312, 229.349, 229.392, 229.354, 229.332, 229.369, 229.320, 229.310, 229.292,
      229.260, 229.278, 229.270, 229.255, 229.280, 229.290, 229.301, 229.251, 229.236, 229.243,
      229.204, 229.216, 229.228, 229.225, 229.234, 229.263, 229.266, 229.253, 229.310, 229.323,
      229.308, 229.311, 229.345, 229.318, 229.316, 229.294, 229.323, 229.304, 229.308, 229.301,
      229.308, 229.311, 229.345, 229.318, 229.316, 229.294, 229.323, 229.304, 229.308, 229.301,
  };
  pos *output = malloc(sizeof (pos) * n_prices);
  int j;

  pos start_pos[] = {0, 1};
  for (j = 0; j < 1; j++) {
    if ((j % 10000) == 0) {
      printf("%d\n", j);
    }
    optimal_trading_positions(
        0.00013,                  // Long entry fee
        0.00013,                   // Long exit fee
        0.00013,                 // Short entry fee
        0.00013,                  // Short exit fee
        false,      // Allow immediate position switch
        sizeof (start_pos)/sizeof (pos), &*start_pos,
        0,                         // last_position
        n_prices, &*prices, &*prices,          // N prices & prices
        &output[0]                      // output
    );
  }
  printf("[");
  int i;
  for (i = 0; i < n_prices; i++) {
    if (i == n_prices - 1) {
      printf("%d", *(output+i));
    } else {
      printf("%d, ", *(output+i));
    }
  }
  printf("]\n");
  return 0;
}