#include <limits>

// TYPE defined in test file.
const auto identity = [] {
  if (TYPE == HALF) {
    // taken from std::numeric_limits<half>::lowest() in HalfFloat.hpp
    return -float(0xFBFF);
  } else {
    return -std::numeric_limits<float>::infinity();
  }
}();

const float expected0[3][1] = {
  {84.7325},
  {identity},
  {81.4111},
};
const float expected1[2][2] = {
  {identity, identity},
  {53.3526, 68.6753},
};
const float expected2[2][3] = {
  {39.1661, 81.2709, 7.4539},
  {57.5866, 17.5915, 65.4395},
};
const float expected3[4][4] = {
  {41.8107, 63.0969, 93.9388, 56.4574},
  {39.2027, 80.4818, 14.3622, 55.7751},
  {36.1396, 34.4204, 38.3141, 12.4978},
  {identity, identity, identity, identity},
};
const float expected4[2][5] = {
  {72.5981, 95.4094, 74.2324, 99.4235, 33.221},
  {86.8387, 98.5069, 98.9627, 68.2653, 44.7801},
};
const float expected5[4][6] = {
  {71.6859, 61.8182, 72.8035, 92.3896, 60.4493, 78.7454},
  {52.3738, 89.9144, 81.0255, 76.1394, 58.2379, 80.1575},
  {56.5042, 30.3654, 62.111, 52.3512, 30.9963, 31.3828},
  {66.6318, 85.5215, 50.8832, 88.3615, 64.0248, 23.6679},
};
const float expected6[1][7] = {
  {99.7273, 48.9835, 78.0362, 91.4062, 79.2411, 71.0992, 90.0964},
};
const float expected7[2][8] = {
  {53.1545, 35.7668, 74.6866, 97.3461, 50.1677, 59.6524, 35.6037, 63.7421},
  {96.6975, 75.5588, 30.0425, 29.803, 95.1839, 24.8316, 62.1086, 14.6577},
};
