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
    "alexnet_tr_bs4_layer11": Expected(cycles=80_687, total_memory=67_077_805, max_tile_mem=137_304),
    "alexnet_tr_bs4_layer1": Expected(cycles=155_877, total_memory=129_799_932, max_tile_mem=174_104),
    "alexnet_tr_bs4_layer4": Expected(cycles=142_598, total_memory=86_619_640, max_tile_mem=138_872),
    "alexnet_tr_bs4_layer7": Expected(cycles=78_727, total_memory=64_657_599, max_tile_mem=138_840),
    "alexnet_tr_bs4_layer9": Expected(cycles=101_806, total_memory=78_659_560, max_tile_mem=141_912),
    "bert_ffn1_128x1024x4096": Expected(cycles=110_702, total_memory=105_640_292, max_tile_mem=144_920),
    "bert_ffn1_128x768x3072": Expected(cycles=71_779, total_memory=64_008_548, max_tile_mem=138_968),
    "bert_ffn1_384x1024x4096": Expected(cycles=259_391, total_memory=193_574_954, max_tile_mem=171_256),
    "bert_ffn1_384x768x3072": Expected(cycles=162_958, total_memory=154_575_530, max_tile_mem=146_648),
    "bert_ffn1_512x1024x4096": Expected(cycles=334_699, total_memory=211_983_224, max_tile_mem=179_516),
    "bert_ffn1_512x768x3072": Expected(cycles=205_159, total_memory=152_397_944, max_tile_mem=151_832),
    "bert_ffn2_128x3072x768": Expected(cycles=76_284, total_memory=63_607_988, max_tile_mem=138_968),
    "bert_ffn2_128x4096x1024": Expected(cycles=116_615, total_memory=105_011_688, max_tile_mem=144_920),
    "bert_ffn2_384x3072x768": Expected(cycles=164_968, total_memory=158_144_272, max_tile_mem=146_648),
    "bert_ffn2_384x4096x1024": Expected(cycles=266_857, total_memory=193_555_834, max_tile_mem=170_776),
    "bert_ffn2_512x3072x768": Expected(cycles=205_751, total_memory=152_300_404, max_tile_mem=151_832),
    "bert_ffn2_512x4096x1024": Expected(cycles=340_553, total_memory=212_265_980, max_tile_mem=203_800),
    "bert_grouped_12x128x64x128": Expected(cycles=14_337, total_memory=10_995_313, max_tile_mem=132_472),
    "bert_grouped_12x384x64x384": Expected(cycles=36_783, total_memory=36_791_438, max_tile_mem=137_240),
    "bert_grouped_12x512x64x512": Expected(cycles=54_895, total_memory=56_037_738, max_tile_mem=142_040),
    "bert_grouped_16x128x64x128": Expected(cycles=15_938, total_memory=13_138_447, max_tile_mem=133_144),
    "bert_grouped_16x384x64x384": Expected(cycles=45_084, total_memory=44_819_372, max_tile_mem=139_288),
    "bert_grouped_16x512x64x512": Expected(cycles=72_116, total_memory=73_313_104, max_tile_mem=147_512),
    "bert_kqv_128x1024x3072": Expected(cycles=87_587, total_memory=81_836_516, max_tile_mem=141_464),
    "bert_kqv_128x768x2304": Expected(cycles=58_766, total_memory=51_169_922, max_tile_mem=138_904),
    "bert_kqv_384x1024x3072": Expected(cycles=202_261, total_memory=196_283_184, max_tile_mem=179_352),
    "bert_kqv_384x768x2304": Expected(cycles=131_950, total_memory=96_956_788, max_tile_mem=145_880),
    "bert_kqv_512x1024x3072": Expected(cycles=254_391, total_memory=166_894_580, max_tile_mem=157_336),
    "bert_kqv_512x768x2304": Expected(cycles=165_628, total_memory=119_627_540, max_tile_mem=150_808),
    "bert_proj_128x1024x1024": Expected(cycles=43_123, total_memory=38_377_626, max_tile_mem=135_192),
    "bert_proj_128x768x768": Expected(cycles=29_251, total_memory=29_530_766, max_tile_mem=135_192),
    "bert_proj_384x1024x1024": Expected(cycles=91_296, total_memory=77_975_714, max_tile_mem=142_040),
    "bert_proj_384x768x768": Expected(cycles=59_213, total_memory=48_652_484, max_tile_mem=137_240),
    "bert_proj_512x1024x1024": Expected(cycles=113_299, total_memory=98_002_374, max_tile_mem=145_688),
    "bert_proj_512x768x768": Expected(cycles=72_547, total_memory=61_093_112, max_tile_mem=139_288),
    "bert_reduce_0": Expected(cycles=1_670, total_memory=3_998_792, max_tile_mem=4_100),
    "bert_reduce_1": Expected(cycles=1_882, total_memory=4_107_848, max_tile_mem=4_140),
    "bert_reduce_2": Expected(cycles=2_800, total_memory=3_849_224, max_tile_mem=5_784),
    "bert_reduce_3": Expected(cycles=2_800, total_memory=3_849_224, max_tile_mem=5_784),
    "bert_reduce_4": Expected(cycles=4_615, total_memory=5_048_512, max_tile_mem=16_632),
    "conv_5x200_1_in_100_out_bs1440": Expected(cycles=312_448, total_memory=191_371_079, max_tile_mem=211_576),
    "embedding_small": Expected(cycles=175_841, total_memory=71_097_744, max_tile_mem=87_544),
    "embedding_vlarge": Expected(cycles=148_071, total_memory=59_240_142, max_tile_mem=77_080),
    "fc_layer_1440x100x200": Expected(cycles=24_435, total_memory=18_810_272, max_tile_mem=132_696),
    "fc_layer_1440x200x400": Expected(cycles=41_303, total_memory=44_357_646, max_tile_mem=135_896),
    "fc_layer_16x1324x100": Expected(cycles=12_098, total_memory=6_388_054, max_tile_mem=131_608),
    "fc_layer_1_1000_1000_float": Expected(cycles=34_079, total_memory=12_535_060, max_tile_mem=32_952),
    "fc_layer_1_1000_1000_half": Expected(cycles=12_909, total_memory=8_390_260, max_tile_mem=32_872),
    "fc_layer_1_1000_5_float": Expected(cycles=9_996, total_memory=4_408_406, max_tile_mem=131_128),
    "fc_layer_1_1000_5_half": Expected(cycles=12_304, total_memory=4_383_737, max_tile_mem=131_128),
    "fc_layer_4_1000_1000_float": Expected(cycles=20_152, total_memory=14_989_366, max_tile_mem=134_936),
    "fc_layer_4_1000_1000_half": Expected(cycles=25_109, total_memory=16_332_681, max_tile_mem=134_488),
    "fc_layer_4_1000_5_float": Expected(cycles=8_586, total_memory=4_547_273, max_tile_mem=131_128),
    "fc_layer_4_1000_5_half": Expected(cycles=12_075, total_memory=4_489_229, max_tile_mem=131_128),
    "fc_layer_80_1324_100": Expected(cycles=16_867, total_memory=10_865_321, max_tile_mem=132_536),
    "gemm_1000x256x10000": Expected(cycles=114_964, total_memory=198_319_954, max_tile_mem=197_464),
    "gemm_1000x256x20000": Expected(cycles=357_332, total_memory=210_911_168, max_tile_mem=192_136),
    "gemm_1000x256x30000": Expected(cycles=497_237, total_memory=302_111_576, max_tile_mem=314_840),
    "gemm_1000x512x10000": Expected(cycles=311_177, total_memory=181_645_942, max_tile_mem=182_040),
    "gemm_1000x512x20000": Expected(cycles=669_398, total_memory=272_790_236, max_tile_mem=299_032),
    "gemm_1000x512x30000": Expected(cycles=948_780, total_memory=302_634_182, max_tile_mem=317_208),
    "gemm_1000x64x10000": Expected(cycles=34_702, total_memory=113_354_416, max_tile_mem=164_888),
    "gemm_1000x64x20000": Expected(cycles=62_828, total_memory=183_579_822, max_tile_mem=198_072),
    "gemm_1000x64x30000": Expected(cycles=91_282, total_memory=255_033_904, max_tile_mem=281_864),
    "gemm_200x256x10000": Expected(cycles=36_563, total_memory=66_988_592, max_tile_mem=144_376),
    "gemm_200x256x20000": Expected(cycles=59_434, total_memory=109_212_224, max_tile_mem=144_376),
    "gemm_200x256x30000": Expected(cycles=78_708, total_memory=148_217_892, max_tile_mem=150_936),
    "gemm_200x512x10000": Expected(cycles=62_218, total_memory=111_055_220, max_tile_mem=157_816),
    "gemm_200x512x20000": Expected(cycles=106_711, total_memory=179_071_552, max_tile_mem=190_424),
    "gemm_200x512x30000": Expected(cycles=165_518, total_memory=195_856_080, max_tile_mem=193_082),
    "gemm_200x64x10000": Expected(cycles=11_641, total_memory=38_251_180, max_tile_mem=138_008),
    "gemm_200x64x20000": Expected(cycles=18_588, total_memory=51_797_980, max_tile_mem=144_376),
    "gemm_200x64x30000": Expected(cycles=24_894, total_memory=71_515_872, max_tile_mem=150_936),
    "gemm_600x256x10000": Expected(cycles=80_890, total_memory=148_972_864, max_tile_mem=183_704),
    "gemm_600x256x20000": Expected(cycles=145_504, total_memory=207_800_684, max_tile_mem=206_104),
    "gemm_600x256x30000": Expected(cycles=248_312, total_memory=209_878_396, max_tile_mem=191_640),
    "gemm_600x512x10000": Expected(cycles=158_867, total_memory=195_476_772, max_tile_mem=206_104),
    "gemm_600x512x20000": Expected(cycles=329_918, total_memory=221_201_776, max_tile_mem=210_344),
    "gemm_600x512x30000": Expected(cycles=520_275, total_memory=230_388_384, max_tile_mem=213_616),
    "gemm_600x64x10000": Expected(cycles=23_140, total_memory=94_388_456, max_tile_mem=151_128),
    "gemm_600x64x20000": Expected(cycles=43_110, total_memory=180_309_440, max_tile_mem=203_864),
    "gemm_600x64x30000": Expected(cycles=60_708, total_memory=165_917_626, max_tile_mem=194_584),
    "inception_tr_bs4_i10_a1x1": Expected(cycles=51_603, total_memory=48_620_282, max_tile_mem=139_864),
    "inception_tr_bs4_i10_b1x1": Expected(cycles=56_755, total_memory=51_808_813, max_tile_mem=136_600),
    "inception_tr_bs4_i10_c1x1": Expected(cycles=64_221, total_memory=58_337_880, max_tile_mem=143_384),
    "inception_tr_bs4_i10_d1x1": Expected(cycles=37_858, total_memory=44_745_631, max_tile_mem=136_600),
    "inception_tr_bs4_i1_a1x1": Expected(cycles=28_243, total_memory=36_169_574, max_tile_mem=135_256),
    "inception_tr_bs4_i1_b1x1": Expected(cycles=24_747, total_memory=31_087_302, max_tile_mem=134_232),
    "inception_tr_bs4_i1_b5x5": Expected(cycles=84_818, total_memory=59_055_961, max_tile_mem=138_936),
    "inception_tr_bs4_i1_c3x3a": Expected(cycles=64_836, total_memory=54_802_732, max_tile_mem=138_008),
    "inception_tr_bs4_i1_c3x3b": Expected(cycles=89_908, total_memory=73_482_912, max_tile_mem=141_176),
    "inception_tr_bs4_i1_db1x1": Expected(cycles=19_978, total_memory=26_472_828, max_tile_mem=134_232),
    "inception_tr_bs4_i2_a1x1": Expected(cycles=32_137, total_memory=42_791_118, max_tile_mem=135_256),
    "inception_tr_bs4_i2_b1x1": Expected(cycles=28_175, total_memory=37_380_685, max_tile_mem=135_256),
    "inception_tr_bs4_i3_a1x1": Expected(cycles=34_301, total_memory=38_013_363, max_tile_mem=137_368),
    "inception_tr_bs4_i3_b1x1": Expected(cycles=29_624, total_memory=40_179_328, max_tile_mem=135_832),
    "inception_tr_bs4_i4_a3x3": Expected(cycles=230_323, total_memory=162_188_671, max_tile_mem=165_656),
    "inception_tr_bs4_i4_b3x3b": Expected(cycles=49_758, total_memory=49_895_442, max_tile_mem=141_464),
    "inception_tr_bs4_i5_a1x1": Expected(cycles=50_046, total_memory=47_597_046, max_tile_mem=138_488),
    "inception_tr_bs4_i5_b1x1": Expected(cycles=38_067, total_memory=38_888_614, max_tile_mem=136_024),
    "inception_tr_bs4_i5_b1x7": Expected(cycles=43_924, total_memory=43_557_143, max_tile_mem=136_856),
    "inception_tr_bs4_i5_b7x1": Expected(cycles=58_319, total_memory=57_032_425, max_tile_mem=139_736),
    "inception_tr_bs4_i5_c1x7b": Expected(cycles=56_182, total_memory=56_275_725, max_tile_mem=139_736),
    "inception_tr_bs4_i5_c7x1a": Expected(cycles=45_722, total_memory=43_472_267, max_tile_mem=136_856),
    "inception_tr_bs4_i6_b1x1": Expected(cycles=43_856, total_memory=43_925_592, max_tile_mem=137_240),
    "inception_tr_bs4_i6_b1x7": Expected(cycles=57_508, total_memory=55_748_994, max_tile_mem=137_624),
    "inception_tr_bs4_i6_b7x1": Expected(cycles=58_766, total_memory=42_178_383, max_tile_mem=138_712),
    "inception_tr_bs4_i6_c1x7b": Expected(cycles=63_343, total_memory=64_699_931, max_tile_mem=138_968),
    "inception_tr_bs4_i6_c7x1a": Expected(cycles=61_597, total_memory=49_544_931, max_tile_mem=138_264),
    "inception_tr_bs4_i6_c7x1b": Expected(cycles=54_748, total_memory=37_738_271, max_tile_mem=137_624),
    "inception_tr_bs4_i7_b1x7": Expected(cycles=74_477, total_memory=63_963_742, max_tile_mem=144_056),
    "inception_tr_bs4_i7_b7x1": Expected(cycles=69_214, total_memory=40_674_696, max_tile_mem=140_344),
    "inception_tr_bs4_i8_a3x3": Expected(cycles=55_862, total_memory=50_521_994, max_tile_mem=140_344),
    "inception_tr_bs4_i8_b3x3": Expected(cycles=44_720, total_memory=38_332_513, max_tile_mem=140_344),
    "inception_tr_bs4_i9_a1x1": Expected(cycles=37_887, total_memory=44_364_037, max_tile_mem=136_600),
    "inception_tr_bs4_i9_ba1x3": Expected(cycles=41_952, total_memory=35_840_630, max_tile_mem=137_240),
    "inception_tr_bs4_i9_bb3x1": Expected(cycles=37_480, total_memory=29_788_385, max_tile_mem=135_192),
    "inception_tr_bs4_i9_c1x1": Expected(cycles=45_919, total_memory=43_055_067, max_tile_mem=136_600),
    "inception_tr_bs4_i9_c3x3": Expected(cycles=90_970, total_memory=86_233_392, max_tile_mem=136_856),
    "inception_tr_bs4_i9_d1x1": Expected(cycles=28_331, total_memory=32_098_137, max_tile_mem=134_424),
    "inception_tr_bs4_layer10": Expected(cycles=402_483, total_memory=195_658_253, max_tile_mem=206_392),
    "inception_tr_bs4_layer1": Expected(cycles=64_304, total_memory=74_805_569, max_tile_mem=140_696),
    "inception_tr_bs4_layer3": Expected(cycles=152_037, total_memory=141_798_050, max_tile_mem=140_696),
    "inception_tr_bs4_layer5": Expected(cycles=260_276, total_memory=229_059_484, max_tile_mem=206_474),
    "inception_tr_bs4_layer8": Expected(cycles=100_639, total_memory=117_758_363, max_tile_mem=153_976),
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
    "mobilenet_conv1_1": Expected(cycles=46_450, total_memory=54_696_913, max_tile_mem=136_472),
    "mobilenet_conv_pw_1_1": Expected(cycles=38_785, total_memory=65_304_229, max_tile_mem=141_720),
    "mobilenet_conv_pw_12_1": Expected(cycles=38_442, total_memory=46_992_203, max_tile_mem=135_320),
    "mobilenet_conv_pw_13_1": Expected(cycles=60_031, total_memory=54_167_625, max_tile_mem=135_320),
    "mobilenet_conv_pw_2_1": Expected(cycles=36_030, total_memory=53_210_208, max_tile_mem=136_408),
    "mobilenet_conv_pw_3_1": Expected(cycles=58_159, total_memory=63_424_971, max_tile_mem=136_408),
    "mobilenet_conv_pw_4_1": Expected(cycles=36_617, total_memory=47_807_154, max_tile_mem=136_408),
    "mobilenet_conv_pw_5_1": Expected(cycles=58_958, total_memory=54_984_059, max_tile_mem=136_600),
    "mobilenet_conv_pw_6_1": Expected(cycles=36_856, total_memory=34_129_575, max_tile_mem=136_728),
    "mobilenet_conv_pw_7_1": Expected(cycles=58_402, total_memory=51_798_304, max_tile_mem=137_688),
    "mobilenet_depthwise_11": Expected(cycles=54_933, total_memory=24_042_423, max_tile_mem=132_696),
    "mobilenet_depthwise_12": Expected(cycles=54_899, total_memory=22_097_430, max_tile_mem=133_784),
    "mobilenet_depthwise_1": Expected(cycles=134_778, total_memory=63_077_774, max_tile_mem=141_848),
    "mobilenet_depthwise_2": Expected(cycles=155_098, total_memory=62_525_936, max_tile_mem=136_472),
    "mobilenet_depthwise_3": Expected(cycles=75_415, total_memory=41_825_263, max_tile_mem=136_472),
    "mobilenet_depthwise_4": Expected(cycles=93_454, total_memory=40_140_703, max_tile_mem=134_232),
    "mobilenet_depthwise_5": Expected(cycles=57_342, total_memory=33_567_529, max_tile_mem=134_296),
    "mobilenet_depthwise_6": Expected(cycles=33_837, total_memory=18_315_617, max_tile_mem=132_824),
    "mobilenet_depthwise": Expected(cycles=175_046, total_memory=64_027_864, max_tile_mem=136_472),
    "resnet50_tr_bs1_bm128L0A0_reduce": Expected(cycles=2_709, total_memory=4_391_559, max_tile_mem=16_576),
    "resnet50_tr_bs1_bm128L0_reduce": Expected(cycles=3_092, total_memory=5_577_650, max_tile_mem=17_072),
    "resnet50_tr_bs1_bm64L0A0_reduce": Expected(cycles=2_893, total_memory=4_343_628, max_tile_mem=16_744),
    "resnet50_tr_bs1_cnv_reduce": Expected(cycles=3_408, total_memory=6_069_210, max_tile_mem=17_728),
    "resnet50_tr_bs4_bm128L0A0": Expected(cycles=44_539, total_memory=59_610_605, max_tile_mem=136_408),
    "resnet50_tr_bs4_bm128L0A1": Expected(cycles=98_114, total_memory=76_089_349, max_tile_mem=138_488),
    "resnet50_tr_bs4_bm128L0A2": Expected(cycles=54_754, total_memory=56_871_908, max_tile_mem=136_600),
    "resnet50_tr_bs4_bm128L0_projection": Expected(cycles=99_907, total_memory=105_769_602, max_tile_mem=141_720),
    "resnet50_tr_bs4_bm128L1A0": Expected(cycles=55_961, total_memory=56_874_323, max_tile_mem=136_600),
    "resnet50_tr_bs4_bm256L0A0": Expected(cycles=41_976, total_memory=51_091_957, max_tile_mem=136_728),
    "resnet50_tr_bs4_bm256L0A1": Expected(cycles=101_905, total_memory=93_127_747, max_tile_mem=140_952),
    "resnet50_tr_bs4_bm256L0A2": Expected(cycles=55_685, total_memory=53_144_484, max_tile_mem=136_728),
    "resnet50_tr_bs4_bm256L0_projection": Expected(cycles=99_068, total_memory=89_282_476, max_tile_mem=144_280),
    "resnet50_tr_bs4_bm256L1A0": Expected(cycles=56_773, total_memory=52_903_108, max_tile_mem=136_728),
    "resnet50_tr_bs4_bm512L0A0": Expected(cycles=41_527, total_memory=50_185_338, max_tile_mem=135_320),
    "resnet50_tr_bs4_bm512L0A1": Expected(cycles=105_698, total_memory=102_955_503, max_tile_mem=143_640),
    "resnet50_tr_bs4_bm512L0A2": Expected(cycles=57_976, total_memory=55_379_449, max_tile_mem=137_368),
    "resnet50_tr_bs4_bm512L0_projection": Expected(cycles=100_493, total_memory=95_677_462, max_tile_mem=140_504),
    "resnet50_tr_bs4_bm512L1A0": Expected(cycles=58_135, total_memory=57_257_850, max_tile_mem=140_504),
    "resnet50_tr_bs4_bm64L0A1": Expected(cycles=92_430, total_memory=79_929_262, max_tile_mem=137_240),
    "resnet50_tr_bs4_bm64L0": Expected(cycles=27_092, total_memory=35_603_609, max_tile_mem=133_752),
    "resnet50_tr_bs4_bm64L0_projection": Expected(cycles=56_612, total_memory=66_259_784, max_tile_mem=141_720),
    "resnet50_tr_bs4_bm64L1A0": Expected(cycles=57_896, total_memory=66_167_841, max_tile_mem=141_720),
    "resnet50_tr_bs4_cnv": Expected(cycles=168_369, total_memory=113_488_307, max_tile_mem=152_600),
    "vgg16_tr_bs4_v1L0": Expected(cycles=210_144, total_memory=184_978_986, max_tile_mem=218_136),
    "vgg16_tr_bs4_v1L1": Expected(cycles=1_258_656, total_memory=359_721_809, max_tile_mem=328_856),
    "vgg16_tr_bs4_v2L0": Expected(cycles=565_119, total_memory=233_399_643, max_tile_mem=223_690),
    "vgg16_tr_bs4_v2L1": Expected(cycles=1_159_335, total_memory=265_262_345, max_tile_mem=270_786),
    "vgg16_tr_bs4_v3L0": Expected(cycles=524_214, total_memory=213_676_524, max_tile_mem=190_250),
    "vgg16_tr_bs4_v3L1": Expected(cycles=1_030_406, total_memory=233_633_318, max_tile_mem=232_884),
    "vgg16_tr_bs4_v4L0": Expected(cycles=542_440, total_memory=191_307_876, max_tile_mem=202_712),
    "vgg16_tr_bs4_v4L1": Expected(cycles=1_067_711, total_memory=198_060_628, max_tile_mem=200_466),
    "vgg16_tr_bs4_v5L0": Expected(cycles=321_662, total_memory=191_255_899, max_tile_mem=202_936),
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
