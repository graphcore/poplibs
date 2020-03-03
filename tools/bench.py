#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd, All rights reserved.

"""
A tool to run a benchmark and ensure its cycle count and memory usage is within
a given limit.
"""

import argparse
import json
import subprocess
import tempfile
import collections
import math
import sys

# The maximum allowed relative difference in memory/cycles before an exception
RELATIVE_TOLERANCE = 0.01

Expected = collections.namedtuple(
    "Expected", ["cycles", "total_memory", "max_tile_mem"]
)
NONE = Expected(
    cycles=sys.maxsize, total_memory=sys.maxsize, max_tile_mem=sys.maxsize
)

# each result here matches with a benchmark defined in the CMakeLists.txt
# inside poplibs/test.
# Please keep in alphabetical order!
# fmt: off
EXPECTED_RESULTS = {
   "alexnet_tr_bs4_layer11": Expected(cycles=86_611, total_memory=66_798_161, max_tile_mem=138_584),
    "alexnet_tr_bs4_layer1": Expected(cycles=159_063, total_memory=129_746_428, max_tile_mem=174_104),
    "alexnet_tr_bs4_layer4": Expected(cycles=147_725, total_memory=86_567_896, max_tile_mem=138_872),
    "alexnet_tr_bs4_layer7": Expected(cycles=82_459, total_memory=64_606_911, max_tile_mem=138_840),
    "alexnet_tr_bs4_layer9": Expected(cycles=106_761, total_memory=80_446_083, max_tile_mem=141_912),
    "bert_ffn1_128x1024x4096": Expected(cycles=115_139, total_memory=108_103_524, max_tile_mem=147_480),
    "bert_ffn1_128x768x3072": Expected(cycles=76_291, total_memory=64_967_844, max_tile_mem=141_464),
    "bert_ffn1_384x1024x4096": Expected(cycles=270_302, total_memory=193_423_152, max_tile_mem=177_016),
    "bert_ffn1_384x768x3072": Expected(cycles=172_585, total_memory=121_496_024, max_tile_mem=146_648),
    "bert_ffn1_512x1024x4096": Expected(cycles=350_268, total_memory=212_121_358, max_tile_mem=175_596),
    "bert_ffn1_512x768x3072": Expected(cycles=215_588, total_memory=152_397_944, max_tile_mem=151_832),
    "bert_ffn2_128x3072x768": Expected(cycles=80_095, total_memory=64_667_572, max_tile_mem=141_464),
    "bert_ffn2_128x4096x1024": Expected(cycles=120_771, total_memory=116_064_384, max_tile_mem=147_480),
    "bert_ffn2_384x3072x768": Expected(cycles=173_164, total_memory=121_398_484, max_tile_mem=146_648),
    "bert_ffn2_384x4096x1024": Expected(cycles=277_603, total_memory=193_404_032, max_tile_mem=174_970),
    "bert_ffn2_512x3072x768": Expected(cycles=217_385, total_memory=146_391_930, max_tile_mem=163_864),
    "bert_ffn2_512x4096x1024": Expected(cycles=362_068, total_memory=212_035_840, max_tile_mem=212_690),
    "bert_grouped_12x128x64x128": Expected(cycles=14_602, total_memory=11_038_055, max_tile_mem=132_472),
    "bert_grouped_12x384x64x384": Expected(cycles=38_721, total_memory=36_791_438, max_tile_mem=137_240),
    "bert_grouped_12x512x64x512": Expected(cycles=58_002, total_memory=59_945_778, max_tile_mem=142_040),
    "bert_grouped_16x128x64x128": Expected(cycles=16_367, total_memory=13_138_447, max_tile_mem=133_144),
    "bert_grouped_16x384x64x384": Expected(cycles=47_634, total_memory=47_174_156, max_tile_mem=139_288),
    "bert_grouped_16x512x64x512": Expected(cycles=76_643, total_memory=73_313_104, max_tile_mem=147_512),
    "bert_kqv_128x1024x3072": Expected(cycles=91_541, total_memory=81_836_516, max_tile_mem=141_464),
    "bert_kqv_128x768x2304": Expected(cycles=62_000, total_memory=51_169_922, max_tile_mem=138_904),
    "bert_kqv_384x1024x3072": Expected(cycles=211_757, total_memory=196_283_184, max_tile_mem=179_352),
    "bert_kqv_384x768x2304": Expected(cycles=139_069, total_memory=96_956_788, max_tile_mem=145_880),
    "bert_kqv_512x1024x3072": Expected(cycles=267_123, total_memory=166_894_580, max_tile_mem=157_336),
    "bert_kqv_512x768x2304": Expected(cycles=175_727, total_memory=130_720_424, max_tile_mem=155_672),
    "bert_proj_128x1024x1024": Expected(cycles=45_499, total_memory=53_551_802, max_tile_mem=135_192),
    "bert_proj_128x768x768": Expected(cycles=30_644, total_memory=32_154_235, max_tile_mem=135_192),
    "bert_proj_384x1024x1024": Expected(cycles=95_394, total_memory=77_975_714, max_tile_mem=142_040),
    "bert_proj_384x768x768": Expected(cycles=61_749, total_memory=49_827_500, max_tile_mem=139_288),
    "bert_proj_512x1024x1024": Expected(cycles=117_975, total_memory=98_548_422, max_tile_mem=142_040),
    "bert_proj_512x768x768": Expected(cycles=75_871, total_memory=63_094_296, max_tile_mem=142_040),
    "bert_reduce_0": Expected(cycles=1_670, total_memory=3_998_792, max_tile_mem=4_100),
    "bert_reduce_1": Expected(cycles=1_882, total_memory=4_107_848, max_tile_mem=4_140),
    "bert_reduce_2": Expected(cycles=2_800, total_memory=3_849_224, max_tile_mem=5_784),
    "bert_reduce_3": Expected(cycles=2_800, total_memory=3_849_224, max_tile_mem=5_784),
    "bert_reduce_4": Expected(cycles=4_615, total_memory=5_048_512, max_tile_mem=16_632),
    "conv_5x200_1_in_100_out_bs1440": Expected(cycles=325_711, total_memory=191_284_727, max_tile_mem=211_576),
    "embedding_small": Expected(cycles=175_841, total_memory=71_097_744, max_tile_mem=87_544),
    "embedding_vlarge": Expected(cycles=148_071, total_memory=59_240_206, max_tile_mem=77_080),
    "fc_layer_1440x100x200": Expected(cycles=27_905, total_memory=18_613_662, max_tile_mem=134_296),
    "fc_layer_1440x200x400": Expected(cycles=43_674, total_memory=44_357_646, max_tile_mem=135_896),
    "fc_layer_16x1324x100": Expected(cycles=12_077, total_memory=6_388_054, max_tile_mem=131_608),
    "fc_layer_1_1000_1000_float": Expected(cycles=34_079, total_memory=12_535_060, max_tile_mem=32_952),
    "fc_layer_1_1000_1000_half": Expected(cycles=12_909, total_memory=8_390_260, max_tile_mem=32_872),
    "fc_layer_1_1000_5_float": Expected(cycles=9_995, total_memory=4_408_406, max_tile_mem=131_128),
    "fc_layer_1_1000_5_half": Expected(cycles=12_290, total_memory=4_383_737, max_tile_mem=131_128),
    "fc_layer_4_1000_1000_float": Expected(cycles=20_147, total_memory=14_989_366, max_tile_mem=134_936),
    "fc_layer_4_1000_1000_half": Expected(cycles=25_250, total_memory=16_332_681, max_tile_mem=134_488),
    "fc_layer_4_1000_5_float": Expected(cycles=8_585, total_memory=4_547_273, max_tile_mem=131_128),
    "fc_layer_4_1000_5_half": Expected(cycles=12_061, total_memory=4_489_229, max_tile_mem=131_128),
    "fc_layer_80_1324_100": Expected(cycles=17_362, total_memory=10_865_321, max_tile_mem=132_536),
    "gemm_1000x256x10000": Expected(cycles=122_949, total_memory=198_315_130, max_tile_mem=197_464),
    "gemm_1000x256x20000": Expected(cycles=375_460, total_memory=210_906_304, max_tile_mem=192_136),
    "gemm_1000x256x30000": Expected(cycles=515_365, total_memory=302_106_712, max_tile_mem=314_840),
    "gemm_1000x512x10000": Expected(cycles=320_089, total_memory=181_641_078, max_tile_mem=182_040),
    "gemm_1000x512x20000": Expected(cycles=687_510, total_memory=272_785_372, max_tile_mem=299_032),
    "gemm_1000x512x30000": Expected(cycles=985_036, total_memory=302_629_318, max_tile_mem=317_208),
    "gemm_1000x64x10000": Expected(cycles=35_778, total_memory=113_349_552, max_tile_mem=164_888),
    "gemm_1000x64x20000": Expected(cycles=64_765, total_memory=183_574_998, max_tile_mem=198_072),
    "gemm_1000x64x30000": Expected(cycles=93_793, total_memory=255_029_088, max_tile_mem=281_864),
    "gemm_200x256x10000": Expected(cycles=39_366, total_memory=66_983_752, max_tile_mem=144_376),
    "gemm_200x256x20000": Expected(cycles=65_117, total_memory=109_207_404, max_tile_mem=144_376),
    "gemm_200x256x30000": Expected(cycles=84_391, total_memory=148_213_052, max_tile_mem=150_936),
    "gemm_200x512x10000": Expected(cycles=65_021, total_memory=111_050_420, max_tile_mem=157_816),
    "gemm_200x512x20000": Expected(cycles=112_394, total_memory=179_066_712, max_tile_mem=190_424),
    "gemm_200x512x30000": Expected(cycles=175_729, total_memory=195_851_232, max_tile_mem=193_082),
    "gemm_200x64x10000": Expected(cycles=12_143, total_memory=38_246_344, max_tile_mem=138_008),
    "gemm_200x64x20000": Expected(cycles=19_951, total_memory=51_793_160, max_tile_mem=144_376),
    "gemm_200x64x30000": Expected(cycles=26_257, total_memory=71_511_032, max_tile_mem=150_936),
    "gemm_600x256x10000": Expected(cycles=86_573, total_memory=148_968_004, max_tile_mem=183_704),
    "gemm_600x256x20000": Expected(cycles=156_942, total_memory=207_795_820, max_tile_mem=206_104),
    "gemm_600x256x30000": Expected(cycles=265_361, total_memory=209_873_556, max_tile_mem=191_640),
    "gemm_600x512x10000": Expected(cycles=173_712, total_memory=210_338_116, max_tile_mem=211_256),
    "gemm_600x512x20000": Expected(cycles=346_967, total_memory=221_196_936, max_tile_mem=210_344),
    "gemm_600x512x30000": Expected(cycles=548_520, total_memory=250_339_564, max_tile_mem=241_384),
    "gemm_600x64x10000": Expected(cycles=23_642, total_memory=94_383_592, max_tile_mem=151_128),
    "gemm_600x64x20000": Expected(cycles=43_612, total_memory=180_304_576, max_tile_mem=203_864),
    "gemm_600x64x30000": Expected(cycles=62_932, total_memory=165_912_786, max_tile_mem=194_584),
    "inception_tr_bs4_i10_a1x1": Expected(cycles=54_261, total_memory=48_615_482, max_tile_mem=139_864),
    "inception_tr_bs4_i10_b1x1": Expected(cycles=60_421, total_memory=51_804_013, max_tile_mem=136_600),
    "inception_tr_bs4_i10_c1x1": Expected(cycles=68_029, total_memory=58_333_064, max_tile_mem=143_384),
    "inception_tr_bs4_i10_d1x1": Expected(cycles=39_357, total_memory=35_533_457, max_tile_mem=136_600),
    "inception_tr_bs4_i1_a1x1": Expected(cycles=27_942, total_memory=36_734_490, max_tile_mem=134_232),
    "inception_tr_bs4_i1_b1x1": Expected(cycles=24_977, total_memory=31_226_372, max_tile_mem=134_232),
    "inception_tr_bs4_i1_b5x5": Expected(cycles=86_295, total_memory=59_735_324, max_tile_mem=140_696),
    "inception_tr_bs4_i1_c3x3a": Expected(cycles=65_628, total_memory=54_752_044, max_tile_mem=138_008),
    "inception_tr_bs4_i1_c3x3b": Expected(cycles=91_498, total_memory=73_430_112, max_tile_mem=141_176),
    "inception_tr_bs4_i1_db1x1": Expected(cycles=20_766, total_memory=26_467_964, max_tile_mem=134_232),
    "inception_tr_bs4_i2_a1x1": Expected(cycles=33_716, total_memory=42_786_254, max_tile_mem=135_256),
    "inception_tr_bs4_i2_b1x1": Expected(cycles=28_985, total_memory=37_766_351, max_tile_mem=135_256),
    "inception_tr_bs4_i3_a1x1": Expected(cycles=35_807, total_memory=38_008_539, max_tile_mem=137_368),
    "inception_tr_bs4_i3_b1x1": Expected(cycles=30_844, total_memory=40_174_504, max_tile_mem=135_832),
    "inception_tr_bs4_i4_a3x3": Expected(cycles=239_477, total_memory=162_133_183, max_tile_mem=165_656),
    "inception_tr_bs4_i4_b3x3b": Expected(cycles=51_061, total_memory=51_123_591, max_tile_mem=141_464),
    "inception_tr_bs4_i5_a1x1": Expected(cycles=52_927, total_memory=65_076_678, max_tile_mem=138_520),
    "inception_tr_bs4_i5_b1x1": Expected(cycles=40_144, total_memory=49_014_509, max_tile_mem=136_024),
    "inception_tr_bs4_i5_b1x7": Expected(cycles=45_460, total_memory=43_501_655, max_tile_mem=136_856),
    "inception_tr_bs4_i5_b7x1": Expected(cycles=59_351, total_memory=56_976_937, max_tile_mem=139_736),
    "inception_tr_bs4_i5_c1x7b": Expected(cycles=58_684, total_memory=56_220_237, max_tile_mem=139_736),
    "inception_tr_bs4_i5_c7x1a": Expected(cycles=46_754, total_memory=43_416_779, max_tile_mem=136_856),
    "inception_tr_bs4_i6_b1x1": Expected(cycles=46_140, total_memory=56_889_615, max_tile_mem=138_520),
    "inception_tr_bs4_i6_b1x7": Expected(cycles=58_577, total_memory=55_796_435, max_tile_mem=140_056),
    "inception_tr_bs4_i6_b7x1": Expected(cycles=62_814, total_memory=43_597_556, max_tile_mem=138_712),
    "inception_tr_bs4_i6_c1x7b": Expected(cycles=63_830, total_memory=65_102_034, max_tile_mem=141_848),
    "inception_tr_bs4_i6_c7x1a": Expected(cycles=63_091, total_memory=49_487_331, max_tile_mem=138_264),
    "inception_tr_bs4_i6_c7x1b": Expected(cycles=55_681, total_memory=37_685_471, max_tile_mem=137_624),
    "inception_tr_bs4_i7_b1x7": Expected(cycles=80_255, total_memory=63_750_979, max_tile_mem=144_056),
    "inception_tr_bs4_i7_b7x1": Expected(cycles=70_511, total_memory=46_245_873, max_tile_mem=140_888),
    "inception_tr_bs4_i8_a3x3": Expected(cycles=63_718, total_memory=61_240_781, max_tile_mem=140_344),
    "inception_tr_bs4_i8_b3x3": Expected(cycles=42_000, total_memory=38_083_486, max_tile_mem=140_344),
    "inception_tr_bs4_i9_a1x1": Expected(cycles=39_394, total_memory=44_359_237, max_tile_mem=136_600),
    "inception_tr_bs4_i9_ba1x3": Expected(cycles=51_728, total_memory=37_709_253, max_tile_mem=137_240),
    "inception_tr_bs4_i9_bb3x1": Expected(cycles=38_458, total_memory=29_737_697, max_tile_mem=135_192),
    "inception_tr_bs4_i9_c1x1": Expected(cycles=48_796, total_memory=43_012_613, max_tile_mem=136_856),
    "inception_tr_bs4_i9_c3x3": Expected(cycles=95_618, total_memory=87_849_492, max_tile_mem=140_696),
    "inception_tr_bs4_i9_d1x1": Expected(cycles=29_462, total_memory=34_240_485, max_tile_mem=134_520),
    "inception_tr_bs4_layer10": Expected(cycles=407_526, total_memory=195_605_453, max_tile_mem=206_392),
    "inception_tr_bs4_layer1": Expected(cycles=66_050, total_memory=74_752_065, max_tile_mem=140_696),
    "inception_tr_bs4_layer3": Expected(cycles=154_386, total_memory=141_745_250, max_tile_mem=140_696),
    "inception_tr_bs4_layer5": Expected(cycles=264_191, total_memory=229_006_684, max_tile_mem=206_474),
    "inception_tr_bs4_layer8": Expected(cycles=105_882, total_memory=117_753_523, max_tile_mem=153_976),
    "inception_tr_bs1_pool1": Expected(cycles=17_706, total_memory=18_465_372, max_tile_mem=22_248),
    "inception_tr_bs1_pool2": Expected(cycles=26_142, total_memory=14_458_452, max_tile_mem=20_568),
	  "inception_tr_bs1_i1_dmaxpool": Expected(cycles=18_493, total_memory=8_663_444, max_tile_mem=18_056),
    "inception_tr_bs1_i2_dmaxpool": Expected(cycles=14_962, total_memory=11_191_284, max_tile_mem=18_632),
    "inception_tr_bs1_i3_dmaxpool": Expected(cycles=33_476, total_memory=11_912_606, max_tile_mem=18_904),
    "inception_tr_bs1_i4_cmaxpool": Expected(cycles=30_281, total_memory=7_989_140, max_tile_mem=17_976),
    "inception_tr_bs1_i5_dmaxpool": Expected(cycles=25_710, total_memory=9_574_597, max_tile_mem=18_056),
    "inception_tr_bs1_i6_dmaxpool": Expected(cycles=15_704, total_memory=6_544_809, max_tile_mem=17_448),
    "inception_tr_bs1_i9_dmax_pool": Expected(cycles=50_486, total_memory=8_949_874, max_tile_mem=17_160),
	  "inception_tr_bs1_i10_dmax_pool": Expected(cycles=38_124, total_memory=8_176_421, max_tile_mem=17_304),
    "mobilenet_conv1_1": Expected(cycles=46_821, total_memory=48_289_709, max_tile_mem=136_920),
    "mobilenet_conv_pw_1_1": Expected(cycles=40_220, total_memory=65_299_381, max_tile_mem=141_720),
    "mobilenet_conv_pw_12_1": Expected(cycles=40_171, total_memory=46_706_609, max_tile_mem=137_368),
    "mobilenet_conv_pw_13_1": Expected(cycles=63_198, total_memory=56_455_647, max_tile_mem=137_368),
    "mobilenet_conv_pw_2_1": Expected(cycles=37_754, total_memory=53_205_344, max_tile_mem=136_408),
    "mobilenet_conv_pw_3_1": Expected(cycles=61_566, total_memory=85_507_185, max_tile_mem=136_408),
    "mobilenet_conv_pw_4_1": Expected(cycles=38_125, total_memory=47_802_290, max_tile_mem=136_408),
    "mobilenet_conv_pw_5_1": Expected(cycles=62_192, total_memory=54_979_195, max_tile_mem=136_600),
    "mobilenet_conv_pw_6_1": Expected(cycles=39_046, total_memory=44_546_361, max_tile_mem=136_728),
    "mobilenet_conv_pw_7_1": Expected(cycles=61_492, total_memory=51_793_696, max_tile_mem=137_688),
    "mobilenet_depthwise_11": Expected(cycles=55_447, total_memory=23_991_735, max_tile_mem=132_696),
    "mobilenet_depthwise_12": Expected(cycles=55_237, total_memory=22_046_742, max_tile_mem=133_784),
    "mobilenet_depthwise_1": Expected(cycles=135_296, total_memory=63_024_270, max_tile_mem=141_848),
    "mobilenet_depthwise_2": Expected(cycles=155_616, total_memory=62_472_432, max_tile_mem=136_472),
    "mobilenet_depthwise_3": Expected(cycles=86_186, total_memory=43_020_590, max_tile_mem=136_472),
    "mobilenet_depthwise_4": Expected(cycles=94_620, total_memory=40_095_647, max_tile_mem=134_232),
    "mobilenet_depthwise_5": Expected(cycles=58_094, total_memory=33_516_841, max_tile_mem=134_296),
    "mobilenet_depthwise_6": Expected(cycles=33_835, total_memory=18_310_753, max_tile_mem=132_824),
    "mobilenet_depthwise": Expected(cycles=175_426, total_memory=63_945_784, max_tile_mem=136_472),
    "resnet50_tr_bs1_bm128L0A0_reduce": Expected(cycles=2_709, total_memory=4_391_559, max_tile_mem=16_576),
    "resnet50_tr_bs1_bm128L0_reduce": Expected(cycles=3_092, total_memory=5_577_650, max_tile_mem=17_072),
    "resnet50_tr_bs1_bm64L0A0_reduce": Expected(cycles=2_893, total_memory=4_343_628, max_tile_mem=16_744),
    "resnet50_tr_bs1_cnv_reduce": Expected(cycles=3_408, total_memory=6_069_210, max_tile_mem=17_728),
    "resnet50_tr_bs4_bm128L0A0": Expected(cycles=46_047, total_memory=59_605_741, max_tile_mem=136_408),
    "resnet50_tr_bs4_bm128L0A1": Expected(cycles=100_548, total_memory=76_035_845, max_tile_mem=138_488),
    "resnet50_tr_bs4_bm128L0A2": Expected(cycles=57_988, total_memory=56_867_044, max_tile_mem=136_600),
    "resnet50_tr_bs4_bm128L0_projection": Expected(cycles=104_725, total_memory=105_764_738, max_tile_mem=141_720),
    "resnet50_tr_bs4_bm128L1A0": Expected(cycles=59_195, total_memory=56_869_459, max_tile_mem=136_600),
    "resnet50_tr_bs4_bm256L0A0": Expected(cycles=43_411, total_memory=51_087_349, max_tile_mem=136_728),
    "resnet50_tr_bs4_bm256L0A1": Expected(cycles=104_923, total_memory=74_595_368, max_tile_mem=139_352),
    "resnet50_tr_bs4_bm256L0A2": Expected(cycles=58_902, total_memory=53_167_394, max_tile_mem=137_688),
    "resnet50_tr_bs4_bm256L0_projection": Expected(cycles=103_542, total_memory=92_625_486, max_tile_mem=142_040),
    "resnet50_tr_bs4_bm256L1A0": Expected(cycles=59_997, total_memory=53_224_954, max_tile_mem=137_688),
    "resnet50_tr_bs4_bm512L0A0": Expected(cycles=43_343, total_memory=50_144_460, max_tile_mem=137_368),
    "resnet50_tr_bs4_bm512L0A1": Expected(cycles=108_278, total_memory=102_897_391, max_tile_mem=143_640),
    "resnet50_tr_bs4_bm512L0A2": Expected(cycles=63_086, total_memory=79_870_076, max_tile_mem=140_504),
    "resnet50_tr_bs4_bm512L0_projection": Expected(cycles=105_104, total_memory=97_530_900, max_tile_mem=149_912),
    "resnet50_tr_bs4_bm512L1A0": Expected(cycles=61_872, total_memory=57_252_986, max_tile_mem=140_504),
    "resnet50_tr_bs4_bm64L0A1": Expected(cycles=94_292, total_memory=79_993_461, max_tile_mem=141_848),
    "resnet50_tr_bs4_bm64L0": Expected(cycles=27_881, total_memory=35_598_745, max_tile_mem=133_752),
    "resnet50_tr_bs4_bm64L0_projection": Expected(cycles=59_270, total_memory=66_254_920, max_tile_mem=141_720),
    "resnet50_tr_bs4_bm64L1A0": Expected(cycles=60_554, total_memory=66_162_977, max_tile_mem=141_720),
    "resnet50_tr_bs4_cnv": Expected(cycles=172_625, total_memory=113_434_803, max_tile_mem=152_600),
    "vgg16_tr_bs4_v1L0": Expected(cycles=211_515, total_memory=184_925_482, max_tile_mem=218_136),
    "vgg16_tr_bs4_v1L1": Expected(cycles=1_273_552, total_memory=359_668_305, max_tile_mem=328_856),
    "vgg16_tr_bs4_v2L0": Expected(cycles=569_493, total_memory=233_346_139, max_tile_mem=223_690),
    "vgg16_tr_bs4_v2L1": Expected(cycles=1_172_911, total_memory=265_208_841, max_tile_mem=270_746),
    "vgg16_tr_bs4_v3L0": Expected(cycles=530_964, total_memory=213_623_020, max_tile_mem=190_250),
    "vgg16_tr_bs4_v3L1": Expected(cycles=1_043_982, total_memory=233_579_814, max_tile_mem=232_884),
    "vgg16_tr_bs4_v4L0": Expected(cycles=578_004, total_memory=205_513_669, max_tile_mem=232_312),
    "vgg16_tr_bs4_v4L1": Expected(cycles=1_109_056, total_memory=203_561_258, max_tile_mem=206_642),
    "vgg16_tr_bs4_v5L0": Expected(cycles=334_840, total_memory=191_197_595, max_tile_mem=202_936),
}
# fmt: on


class TestFailureException(Exception):
    """Raised when a test fails"""

    def __init__(self):
        super(TestFailureException, self).__init__()


def get_always_live(liveness, args):
    """Returns memory usage of always-live variables in bytes."""
    return sum(liveness["alwaysLive"]["bytesByTile"])


def get_max_temp(liveness, args):
    """Returns sum of maximum memory usage per tile of temporary variables."""
    return sum(liveness["notAlwaysLive"]["maxBytesByTile"])


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark regression tool, compares memory and cycles "
        "against the expected values."
    )
    parser.add_argument(
        "--name", help="Test name used to look-up expected results"
    )
    parser.add_argument(
        "test", nargs=argparse.REMAINDER, help="Which test to run"
    )
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile() as out:
        cmd = args.test + ["--profile-json", out.name]
        print("Command: ", *cmd)
        subprocess.run(cmd, check=True)
        result = json.load(out)
        liveness = result["graphProfile"]["memory"]["liveness"]

        cycles = result["executionProfile"]["simulation"]["cycles"]
        memory = get_always_live(liveness, args) + get_max_temp(liveness, args)
        max_tile_mem = max(
            result["graphProfile"]["memory"]["byTile"]["totalIncludingGaps"]
        )

    expected = EXPECTED_RESULTS.get(args.name, NONE)

    def check_value(name, actual_value, expected_value):
        changed = not math.isclose(
            expected_value, actual_value, rel_tol=RELATIVE_TOLERANCE
        )
        if changed:
            pc_diff = actual_value / expected_value * 100 - 100
            print(
                f"ERROR: {name} usage ({actual_value:,}) differs by "
                f"{pc_diff:.1f}% from the expected value ({expected_value:,})"
            )
        return not changed

    passed = True
    passed &= check_value("Total memory", memory, expected.total_memory)
    passed &= check_value(
        "Max tile memory", max_tile_mem, expected.max_tile_mem
    )
    passed &= check_value("Cycles", cycles, expected.cycles)

    if not passed:
        print(
            "To update the benchmark with the new result use the "
            "following line:"
        )
        print(
            f'    "{args.name}": Expected(cycles={cycles:_}, total_memory='
            f"{memory:_}, max_tile_mem={max_tile_mem:_}),"
        )
        raise TestFailureException()


if __name__ == "__main__":
    main()
