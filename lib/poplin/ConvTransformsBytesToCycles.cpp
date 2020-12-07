// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// This file is automatically generated and updated by
// transforms_regression_model.py Do not modify this by hand.
#include "ConvTransformsBytesToCycles.hpp"

namespace poplin {

static int64_t expr_0__true__1_02_1(uint64_t bytes) {
  // 0.16131858850211 * X + 1034.5485750855264
  return (165 * bytes) / 1024 + 1035;
}

static int64_t expr_0__true__10_012_1(uint64_t bytes) {
  // 0.1639239965619107 * X + 2894.9521225800127
  return (168 * bytes) / 1024 + 2895;
}

static int64_t expr_0__true_0__01_1(uint64_t bytes) {
  // 0.0566005819273164 * X + 6295.885680299053
  return (58 * bytes) / 1024 + 6296;
}

static int64_t expr_0__true_1__02_1(uint64_t bytes) {
  // 0.06324917397367653 * X + 5690.7065794219425
  return (65 * bytes) / 1024 + 5691;
}

static int64_t expr_0__false____1(uint64_t bytes) {
  // 0.2612975154089416 * X + 1429.2451559198314
  return (268 * bytes) / 1024 + 1429;
}

static int64_t expr_0__false__0_01_1(uint64_t bytes) {
  // 0.09929605810680764 * X + 4100.407815491073
  return (102 * bytes) / 1024 + 4100;
}

static int64_t expr_0__false__1_02_1(uint64_t bytes) {
  // 0.10232028169747623 * X + 4048.837325709473
  return (105 * bytes) / 1024 + 4049;
}

static int64_t expr_0__false_0__01_1(uint64_t bytes) {
  // 0.10663852336346362 * X + 2138.861383993155
  return (109 * bytes) / 1024 + 2139;
}

static int64_t expr_0__false_0_1_012_1(uint64_t bytes) {
  // 0.10674033190217408 * X + 6793.8183316543855
  return (109 * bytes) / 1024 + 6794;
}

static int64_t expr_0__false_1__02_1(uint64_t bytes) {
  // 0.1428449048691192 * X + 2277.7711332884533
  return (146 * bytes) / 1024 + 2278;
}

static int64_t expr_0__false_1_0_012_1(uint64_t bytes) {
  // 0.11168879058907852 * X + 6282.130091418119
  return (114 * bytes) / 1024 + 6282;
}

static int64_t expr_0__false_10__012_1(uint64_t bytes) {
  // 0.1792069445771783 * X + 2408.590298959275
  return (184 * bytes) / 1024 + 2409;
}

static int64_t expr_0__true__0_01_1(uint64_t bytes) {
  // 0.1310503324713616 * X + 1114.3199960801705
  return (134 * bytes) / 1024 + 1114;
}

static int64_t expr_0__true_0_1_012_1(uint64_t bytes) {
  // 0.04902340387887395 * X + 8456.060523181239
  return (50 * bytes) / 1024 + 8456;
}

static int64_t expr_0__true_1_0_012_1(uint64_t bytes) {
  // 0.059397023513990066 * X + 7614.354185124411
  return (61 * bytes) / 1024 + 7614;
}

static int64_t expr_0__true_1_1_02_1(uint64_t bytes) {
  // 0.08299042463091227 * X + 5858.344322282217
  return (85 * bytes) / 1024 + 5858;
}

static int64_t expr_0__false__0_012_1(uint64_t bytes) {
  // 0.14158724444135454 * X + 709.4228166096108
  return (145 * bytes) / 1024 + 709;
}

static int64_t expr_0__false__1_012_1(uint64_t bytes) {
  // 0.14210391597880873 * X + 578.7802878078508
  return (146 * bytes) / 1024 + 579;
}

static int64_t expr_0__false_0_0_012_1(uint64_t bytes) {
  // 0.14466996610560628 * X + 994.0670704400205
  return (148 * bytes) / 1024 + 994;
}

static int64_t expr_0__false_1_1_012_1(uint64_t bytes) {
  // 0.1460389160827102 * X + 358.8442124786863
  return (150 * bytes) / 1024 + 359;
}

static int64_t expr_0__false_10_0_012_1(uint64_t bytes) {
  // 0.07725958586294454 * X + 8375.314794840635
  return (79 * bytes) / 1024 + 8375;
}

static int64_t expr_0__false_10_1_012_1(uint64_t bytes) {
  // 0.08055819101466608 * X + 7639.79013717999
  return (82 * bytes) / 1024 + 7640;
}

static int64_t expr_0__true____1(uint64_t bytes) {
  // 0.23913439195931308 * X + 1037.0065140585498
  return (245 * bytes) / 1024 + 1037;
}

static int64_t expr_0__true_0_0_01_1(uint64_t bytes) {
  // 0.08307900765855315 * X + 5497.042337324464
  return (85 * bytes) / 1024 + 5497;
}

static int64_t expr_0__true_0_10_012_1(uint64_t bytes) {
  // 0.09319769858877557 * X + 8120.255382435592
  return (95 * bytes) / 1024 + 8120;
}

static int64_t expr_0__true_1_10_012_1(uint64_t bytes) {
  // 0.08948021727687139 * X + 9251.900597740194
  return (92 * bytes) / 1024 + 9252;
}

static int64_t expr_0__true_10__012_1(uint64_t bytes) {
  // 0.7322289840827623 * X + 3134.3594568798126
  return (750 * bytes) / 1024 + 3134;
}

static int64_t expr_0__true_10_0_012_1(uint64_t bytes) {
  // 0.8620488107816132 * X + 1420.6637433713634
  return (883 * bytes) / 1024 + 1421;
}

static int64_t expr_0__true_10_1_012_1(uint64_t bytes) {
  // 0.866449309097467 * X + 1452.6036555679163
  return (887 * bytes) / 1024 + 1453;
}

static int64_t expr_0__true_10_10_012_1(uint64_t bytes) {
  // 0.4650148689242883 * X + 2511.0268270863735
  return (476 * bytes) / 1024 + 2511;
}

static int64_t expr_0__false__10_012_1(uint64_t bytes) {
  // 0.3069541200340428 * X + 1536.03718683563
  return (314 * bytes) / 1024 + 1536;
}

static int64_t expr_0__false_0_0_01_1(uint64_t bytes) {
  // 0.044212883713528724 * X + 7437.057379377116
  return (45 * bytes) / 1024 + 7437;
}

static int64_t expr_0__false_0_10_012_1(uint64_t bytes) {
  // 0.5693513384644383 * X + 1309.3533423880338
  return (583 * bytes) / 1024 + 1309;
}

static int64_t expr_0__false_1_1_02_1(uint64_t bytes) {
  // 0.049116693071021886 * X + 8033.177566461407
  return (50 * bytes) / 1024 + 8033;
}

static int64_t expr_0__false_1_10_012_1(uint64_t bytes) {
  // 0.39318812276640014 * X + 2097.3525607258093
  return (403 * bytes) / 1024 + 2097;
}

static int64_t expr_0__false_10_10_012_1(uint64_t bytes) {
  // 0.4795068276427026 * X + 1686.704623157971
  return (491 * bytes) / 1024 + 1687;
}

static int64_t expr_0__false____4(uint64_t bytes) {
  // 1.0434580224174705 * X + 2247.794906181517
  return (1069 * bytes) / 1024 + 2248;
}

static int64_t expr_0__true____4(uint64_t bytes) {
  // 0.5268344126978621 * X + 24086.36929689567
  return (539 * bytes) / 1024 + 24086;
}

static int64_t expr_0__true____8(uint64_t bytes) {
  // 0.7419377901492245 * X + 17970.317403335743
  return (760 * bytes) / 1024 + 17970;
}

static int64_t expr_0__true____2(uint64_t bytes) {
  // 1.3591182125736223 * X + 3215.2008725063
  return (1392 * bytes) / 1024 + 3215;
}

static int64_t expr_0__false____2(uint64_t bytes) {
  // 3.133338224799464 * X + 2544.543625114949
  return (3209 * bytes) / 1024 + 2545;
}

const ConvTransformsMap conversionTable = {
    {{0, {}, true, {}, {1}, {0, 2}, 1}, &expr_0__true__1_02_1},
    {{0, {}, true, {}, {1, 0}, {0, 1, 2}, 1}, &expr_0__true__10_012_1},
    {{0, {}, true, {0}, {}, {0, 1}, 1}, &expr_0__true_0__01_1},
    {{0, {}, true, {1}, {}, {0, 2}, 1}, &expr_0__true_1__02_1},
    {{0, {}, false, {}, {}, {}, 1}, &expr_0__false____1},
    {{0, {}, false, {}, {0}, {0, 1}, 1}, &expr_0__false__0_01_1},
    {{0, {}, false, {}, {1}, {0, 2}, 1}, &expr_0__false__1_02_1},
    {{0, {}, false, {0}, {}, {0, 1}, 1}, &expr_0__false_0__01_1},
    {{0, {}, false, {0}, {1}, {0, 1, 2}, 1}, &expr_0__false_0_1_012_1},
    {{0, {}, false, {1}, {}, {0, 2}, 1}, &expr_0__false_1__02_1},
    {{0, {}, false, {1}, {0}, {0, 1, 2}, 1}, &expr_0__false_1_0_012_1},
    {{0, {}, false, {1, 0}, {}, {0, 1, 2}, 1}, &expr_0__false_10__012_1},
    {{0, {}, true, {}, {0}, {0, 1}, 1}, &expr_0__true__0_01_1},
    {{0, {}, true, {0}, {1}, {0, 1, 2}, 1}, &expr_0__true_0_1_012_1},
    {{0, {}, true, {1}, {0}, {0, 1, 2}, 1}, &expr_0__true_1_0_012_1},
    {{0, {}, true, {1}, {1}, {0, 2}, 1}, &expr_0__true_1_1_02_1},
    {{0, {}, false, {}, {0}, {0, 1, 2}, 1}, &expr_0__false__0_012_1},
    {{0, {}, false, {}, {1}, {0, 1, 2}, 1}, &expr_0__false__1_012_1},
    {{0, {}, false, {0}, {0}, {0, 1, 2}, 1}, &expr_0__false_0_0_012_1},
    {{0, {}, false, {1}, {1}, {0, 1, 2}, 1}, &expr_0__false_1_1_012_1},
    {{0, {}, false, {1, 0}, {0}, {0, 1, 2}, 1}, &expr_0__false_10_0_012_1},
    {{0, {}, false, {1, 0}, {1}, {0, 1, 2}, 1}, &expr_0__false_10_1_012_1},
    {{0, {}, true, {}, {}, {}, 1}, &expr_0__true____1},
    {{0, {}, true, {0}, {0}, {0, 1}, 1}, &expr_0__true_0_0_01_1},
    {{0, {}, true, {0}, {1, 0}, {0, 1, 2}, 1}, &expr_0__true_0_10_012_1},
    {{0, {}, true, {1}, {1, 0}, {0, 1, 2}, 1}, &expr_0__true_1_10_012_1},
    {{0, {}, true, {1, 0}, {}, {0, 1, 2}, 1}, &expr_0__true_10__012_1},
    {{0, {}, true, {1, 0}, {0}, {0, 1, 2}, 1}, &expr_0__true_10_0_012_1},
    {{0, {}, true, {1, 0}, {1}, {0, 1, 2}, 1}, &expr_0__true_10_1_012_1},
    {{0, {}, true, {1, 0}, {1, 0}, {0, 1, 2}, 1}, &expr_0__true_10_10_012_1},
    {{0, {}, false, {}, {1, 0}, {0, 1, 2}, 1}, &expr_0__false__10_012_1},
    {{0, {}, false, {0}, {0}, {0, 1}, 1}, &expr_0__false_0_0_01_1},
    {{0, {}, false, {0}, {1, 0}, {0, 1, 2}, 1}, &expr_0__false_0_10_012_1},
    {{0, {}, false, {1}, {1}, {0, 2}, 1}, &expr_0__false_1_1_02_1},
    {{0, {}, false, {1}, {1, 0}, {0, 1, 2}, 1}, &expr_0__false_1_10_012_1},
    {{0, {}, false, {1, 0}, {1, 0}, {0, 1, 2}, 1}, &expr_0__false_10_10_012_1},
    {{0, {}, false, {}, {}, {}, 4}, &expr_0__false____4},
    {{0, {}, true, {}, {}, {}, 4}, &expr_0__true____4},
    {{0, {}, true, {}, {}, {}, 8}, &expr_0__true____8},
    {{0, {}, true, {}, {}, {}, 2}, &expr_0__true____2},
    {{0, {}, false, {}, {}, {}, 2}, &expr_0__false____2},
};

} // namespace poplin