#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

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
    "alexnet_tr_bs4_layer11": Expected(cycles=80_517, total_memory=67_097_005, max_tile_mem=137_304),
    "alexnet_tr_bs4_layer1": Expected(cycles=161_767, total_memory=132_178_945, max_tile_mem=174_104),
    "alexnet_tr_bs4_layer4": Expected(cycles=142_598, total_memory=86_619_640, max_tile_mem=138_872),
    "alexnet_tr_bs4_layer7": Expected(cycles=78_727, total_memory=64_657_599, max_tile_mem=138_840),
    "alexnet_tr_bs4_layer9": Expected(cycles=101_806, total_memory=78_659_560, max_tile_mem=141_912),
    "bert_ffn1_128x1024x4096": Expected(cycles=112_110, total_memory=103_248_484, max_tile_mem=144_920),
    "bert_ffn1_128x768x3072": Expected(cycles=71_286, total_memory=64_008_548, max_tile_mem=138_968),
    "bert_ffn1_384x1024x4096": Expected(cycles=258_772, total_memory=183_703_866, max_tile_mem=155_000),
    "bert_ffn1_384x768x3072": Expected(cycles=163_523, total_memory=121_496_024, max_tile_mem=146_648),
    "bert_ffn1_512x1024x4096": Expected(cycles=331_253, total_memory=211_983_224, max_tile_mem=179_516),
    "bert_ffn1_512x768x3072": Expected(cycles=203_274, total_memory=152_397_944, max_tile_mem=151_832),
    "bert_ffn2_128x3072x768": Expected(cycles=75_791, total_memory=63_607_988, max_tile_mem=138_968),
    "bert_ffn2_128x4096x1024": Expected(cycles=114_188, total_memory=96_053_224, max_tile_mem=144_920),
    "bert_ffn2_384x3072x768": Expected(cycles=163_685, total_memory=158_144_272, max_tile_mem=146_648),
    "bert_ffn2_384x4096x1024": Expected(cycles=267_329, total_memory=192_272_430, max_tile_mem=170_776),
    "bert_ffn2_512x3072x768": Expected(cycles=203_866, total_memory=152_300_404, max_tile_mem=151_832),
    "bert_ffn2_512x4096x1024": Expected(cycles=337_107, total_memory=212_265_980, max_tile_mem=203_800),
    "bert_grouped_12x128x64x128": Expected(cycles=14_347, total_memory=10_995_313, max_tile_mem=132_472),
    "bert_grouped_12x384x64x384": Expected(cycles=36_528, total_memory=36_791_438, max_tile_mem=137_240),
    "bert_grouped_12x512x64x512": Expected(cycles=54_504, total_memory=56_037_738, max_tile_mem=142_040),
    "bert_grouped_16x128x64x128": Expected(cycles=15_947, total_memory=13_138_447, max_tile_mem=133_144),
    "bert_grouped_16x384x64x384": Expected(cycles=44_744, total_memory=44_819_372, max_tile_mem=139_288),
    "bert_grouped_16x512x64x512": Expected(cycles=71_365, total_memory=73_313_104, max_tile_mem=147_512),
    "bert_kqv_128x1024x3072": Expected(cycles=87_094, total_memory=81_836_516, max_tile_mem=141_464),
    "bert_kqv_128x768x2304": Expected(cycles=58_358, total_memory=51_169_922, max_tile_mem=138_904),
    "bert_kqv_384x1024x3072": Expected(cycles=200_838, total_memory=196_283_184, max_tile_mem=179_352),
    "bert_kqv_384x768x2304": Expected(cycles=130_760, total_memory=96_956_788, max_tile_mem=145_880),
    "bert_kqv_512x1024x3072": Expected(cycles=252_006, total_memory=166_894_580, max_tile_mem=157_336),
    "bert_kqv_512x768x2304": Expected(cycles=163_902, total_memory=119_627_540, max_tile_mem=150_808),
    "bert_proj_128x1024x1024": Expected(cycles=42_715, total_memory=38_377_626, max_tile_mem=135_192),
    "bert_proj_128x768x768": Expected(cycles=29_123, total_memory=29_530_766, max_tile_mem=135_192),
    "bert_proj_384x1024x1024": Expected(cycles=90_786, total_memory=77_975_714, max_tile_mem=142_040),
    "bert_proj_384x768x768": Expected(cycles=58_805, total_memory=48_652_484, max_tile_mem=137_240),
    "bert_proj_512x1024x1024": Expected(cycles=112_501, total_memory=98_002_374, max_tile_mem=145_688),
    "bert_proj_512x768x768": Expected(cycles=72_139, total_memory=61_093_112, max_tile_mem=139_288),
    "bert_reduce_0": Expected(cycles=1_670, total_memory=3_998_792, max_tile_mem=4_100),
    "bert_reduce_1": Expected(cycles=1_882, total_memory=4_107_848, max_tile_mem=4_140),
    "bert_reduce_2": Expected(cycles=2_800, total_memory=3_849_224, max_tile_mem=5_784),
    "bert_reduce_3": Expected(cycles=2_800, total_memory=3_849_224, max_tile_mem=5_784),
    "bert_reduce_4": Expected(cycles=4_615, total_memory=5_048_512, max_tile_mem=16_632),
    "conv_5x200_1_in_100_out_bs1440": Expected(cycles=310_957, total_memory=191_390_087, max_tile_mem=211_576),
    "embedding_small": Expected(cycles=175_841, total_memory=71_097_744, max_tile_mem=87_544),
    "embedding_vlarge": Expected(cycles=148_071, total_memory=59_240_142, max_tile_mem=77_080),
    "fc_layer_1440x100x200": Expected(cycles=24_453, total_memory=18_810_272, max_tile_mem=132_696),
    "fc_layer_1440x200x400": Expected(cycles=41_149, total_memory=44_357_646, max_tile_mem=135_896),
    "fc_layer_16x1324x100": Expected(cycles=12_101, total_memory=6_388_054, max_tile_mem=131_608),
    "fc_layer_1_1000_1000_float": Expected(cycles=34_079, total_memory=12_535_060, max_tile_mem=32_952),
    "fc_layer_1_1000_1000_half": Expected(cycles=12_909, total_memory=8_390_260, max_tile_mem=32_872),
    "fc_layer_1_1000_5_float": Expected(cycles=9_997, total_memory=4_408_406, max_tile_mem=131_128),
    "fc_layer_1_1000_5_half": Expected(cycles=12_306, total_memory=4_383_737, max_tile_mem=131_128),
    "fc_layer_4_1000_1000_float": Expected(cycles=20_161, total_memory=14_989_366, max_tile_mem=134_936),
    "fc_layer_4_1000_1000_half": Expected(cycles=25_094, total_memory=16_332_681, max_tile_mem=134_488),
    "fc_layer_4_1000_5_float": Expected(cycles=8_587, total_memory=4_547_273, max_tile_mem=131_128),
    "fc_layer_4_1000_5_half": Expected(cycles=12_077, total_memory=4_489_229, max_tile_mem=131_128),
    "fc_layer_80_1324_100": Expected(cycles=16_877, total_memory=10_865_321, max_tile_mem=132_536),
    "gemm_1000x256x10000": Expected(cycles=113_252, total_memory=198_339_250, max_tile_mem=197_464),
    "gemm_1000x256x20000": Expected(cycles=353_940, total_memory=210_930_624, max_tile_mem=192_136),
    "gemm_1000x256x30000": Expected(cycles=493_845, total_memory=302_131_032, max_tile_mem=314_840),
    "gemm_1000x512x10000": Expected(cycles=359_787, total_memory=171_415_748, max_tile_mem=158_768),
    "gemm_1000x512x20000": Expected(cycles=609_704, total_memory=252_255_996, max_tile_mem=233_400),
    "gemm_1000x512x30000": Expected(cycles=941_996, total_memory=302_653_638, max_tile_mem=317_208),
    "gemm_1000x64x10000": Expected(cycles=34_490, total_memory=113_373_872, max_tile_mem=164_888),
    "gemm_1000x64x20000": Expected(cycles=62_400, total_memory=183_599_118, max_tile_mem=198_072),
    "gemm_1000x64x30000": Expected(cycles=90_710, total_memory=255_053_168, max_tile_mem=281_864),
    "gemm_200x256x10000": Expected(cycles=35_995, total_memory=67_007_952, max_tile_mem=144_376),
    "gemm_200x256x20000": Expected(cycles=58_298, total_memory=109_231_504, max_tile_mem=144_376),
    "gemm_200x256x30000": Expected(cycles=77_572, total_memory=148_237_252, max_tile_mem=150_936),
    "gemm_200x512x10000": Expected(cycles=61_650, total_memory=111_074_420, max_tile_mem=157_816),
    "gemm_200x512x20000": Expected(cycles=105_575, total_memory=179_090_912, max_tile_mem=190_424),
    "gemm_200x512x30000": Expected(cycles=163_153, total_memory=195_875_472, max_tile_mem=193_082),
    "gemm_200x64x10000": Expected(cycles=11_573, total_memory=38_270_524, max_tile_mem=138_008),
    "gemm_200x64x20000": Expected(cycles=18_304, total_memory=51_817_260, max_tile_mem=144_376),
    "gemm_200x64x30000": Expected(cycles=24_610, total_memory=71_535_232, max_tile_mem=150_936),
    "gemm_600x256x10000": Expected(cycles=79_754, total_memory=148_992_304, max_tile_mem=183_704),
    "gemm_600x256x20000": Expected(cycles=142_928, total_memory=207_820_140, max_tile_mem=206_104),
    "gemm_600x256x30000": Expected(cycles=244_904, total_memory=209_897_756, max_tile_mem=191_640),
    "gemm_600x512x10000": Expected(cycles=156_291, total_memory=195_496_228, max_tile_mem=206_104),
    "gemm_600x512x20000": Expected(cycles=326_510, total_memory=221_221_136, max_tile_mem=210_344),
    "gemm_600x512x30000": Expected(cycles=514_595, total_memory=230_407_776, max_tile_mem=213_616),
    "gemm_600x64x10000": Expected(cycles=23_072, total_memory=94_407_912, max_tile_mem=151_128),
    "gemm_600x64x20000": Expected(cycles=42_890, total_memory=115_857_398, max_tile_mem=173_592),
    "gemm_600x64x30000": Expected(cycles=60_208, total_memory=165_936_986, max_tile_mem=194_584),
    "inception_tr_bs4_i10_a1x1": Expected(cycles=51_087, total_memory=46_921_782, max_tile_mem=136_600),
    "inception_tr_bs4_i10_b1x1": Expected(cycles=56_296, total_memory=51_828_013, max_tile_mem=136_600),
    "inception_tr_bs4_i10_c1x1": Expected(cycles=63_631, total_memory=58_357_144, max_tile_mem=143_384),
    "inception_tr_bs4_i10_d1x1": Expected(cycles=37_733, total_memory=44_764_831, max_tile_mem=136_600),
    "inception_tr_bs4_i1_a1x1": Expected(cycles=28_131, total_memory=36_189_030, max_tile_mem=135_256),
    "inception_tr_bs4_i1_b1x1": Expected(cycles=24_705, total_memory=31_106_502, max_tile_mem=134_232),
    "inception_tr_bs4_i1_b5x5": Expected(cycles=84_818, total_memory=59_055_961, max_tile_mem=138_936),
    "inception_tr_bs4_i1_c3x3a": Expected(cycles=64_836, total_memory=54_802_732, max_tile_mem=138_008),
    "inception_tr_bs4_i1_c3x3b": Expected(cycles=89_908, total_memory=73_482_912, max_tile_mem=141_176),
    "inception_tr_bs4_i1_db1x1": Expected(cycles=19_954, total_memory=26_492_284, max_tile_mem=134_232),
    "inception_tr_bs4_i2_a1x1": Expected(cycles=32_010, total_memory=42_810_574, max_tile_mem=135_256),
    "inception_tr_bs4_i2_b1x1": Expected(cycles=28_136, total_memory=37_400_141, max_tile_mem=135_256),
    "inception_tr_bs4_i3_a1x1": Expected(cycles=34_097, total_memory=38_032_659, max_tile_mem=137_368),
    "inception_tr_bs4_i3_b1x1": Expected(cycles=29_587, total_memory=40_198_624, max_tile_mem=135_832),
    "inception_tr_bs4_i4_a3x3": Expected(cycles=229_413, total_memory=162_207_871, max_tile_mem=165_656),
    "inception_tr_bs4_i4_b3x3b": Expected(cycles=49_663, total_memory=49_914_450, max_tile_mem=141_464),
    "inception_tr_bs4_i5_a1x1": Expected(cycles=49_548, total_memory=47_616_246, max_tile_mem=138_488),
    "inception_tr_bs4_i5_b1x1": Expected(cycles=37_744, total_memory=38_908_070, max_tile_mem=136_024),
    "inception_tr_bs4_i5_b1x7": Expected(cycles=43_839, total_memory=43_576_343, max_tile_mem=136_856),
    "inception_tr_bs4_i5_b7x1": Expected(cycles=53_697, total_memory=38_001_979, max_tile_mem=137_624),
    "inception_tr_bs4_i5_c1x7b": Expected(cycles=57_987, total_memory=49_244_864, max_tile_mem=139_736),
    "inception_tr_bs4_i5_c7x1a": Expected(cycles=43_870, total_memory=29_023_592, max_tile_mem=135_512),
    "inception_tr_bs4_i6_b1x1": Expected(cycles=43_499, total_memory=43_944_792, max_tile_mem=137_240),
    "inception_tr_bs4_i6_b1x7": Expected(cycles=57_389, total_memory=55_768_354, max_tile_mem=137_624),
    "inception_tr_bs4_i6_b7x1": Expected(cycles=61_215, total_memory=42_219_033, max_tile_mem=138_712),
    "inception_tr_bs4_i6_c1x7b": Expected(cycles=63_224, total_memory=64_719_291, max_tile_mem=138_968),
    "inception_tr_bs4_i6_c7x1a": Expected(cycles=56_009, total_memory=34_622_554, max_tile_mem=135_576),
    "inception_tr_bs4_i6_c7x1b": Expected(cycles=56_641, total_memory=37_890_860, max_tile_mem=137_624),
    "inception_tr_bs4_i7_b1x7": Expected(cycles=74_302, total_memory=63_982_942, max_tile_mem=144_056),
    "inception_tr_bs4_i7_b7x1": Expected(cycles=71_195, total_memory=45_193_382, max_tile_mem=140_344),
    "inception_tr_bs4_i8_a3x3": Expected(cycles=55_709, total_memory=50_541_194, max_tile_mem=140_344),
    "inception_tr_bs4_i8_b3x3": Expected(cycles=44_652, total_memory=38_351_713, max_tile_mem=140_344),
    "inception_tr_bs4_i9_a1x1": Expected(cycles=37_759, total_memory=34_886_970, max_tile_mem=136_600),
    "inception_tr_bs4_i9_ba1x3": Expected(cycles=41_680, total_memory=35_859_062, max_tile_mem=137_240),
    "inception_tr_bs4_i9_bb3x1": Expected(cycles=37_480, total_memory=29_788_385, max_tile_mem=135_192),
    "inception_tr_bs4_i9_c1x1": Expected(cycles=45_579, total_memory=43_074_267, max_tile_mem=136_600),
    "inception_tr_bs4_i9_c3x3": Expected(cycles=94_499, total_memory=87_099_125, max_tile_mem=136_856),
    "inception_tr_bs4_i9_d1x1": Expected(cycles=28_203, total_memory=32_117_337, max_tile_mem=134_424),
    "inception_tr_bs4_layer10": Expected(cycles=399_216, total_memory=217_732_684, max_tile_mem=216_938),
    "inception_tr_bs4_layer1": Expected(cycles=63_352, total_memory=72_666_649, max_tile_mem=140_696),
    "inception_tr_bs4_layer3": Expected(cycles=152_037, total_memory=141_798_050, max_tile_mem=140_696),
    "inception_tr_bs4_layer5": Expected(cycles=260_276, total_memory=229_059_484, max_tile_mem=206_474),
    "inception_tr_bs4_layer8": Expected(cycles=99_613, total_memory=117_777_723, max_tile_mem=153_976),
    "inception_tr_bs1_i10_dmax_pool": Expected(cycles=38_124, total_memory=8_176_421, max_tile_mem=17_304),
    "inception_tr_bs1_pool2": Expected(cycles=26_142, total_memory=14_458_452, max_tile_mem=20_568),
    "inception_tr_bs1_i1_dmaxpool": Expected(cycles=18_493, total_memory=8_663_444, max_tile_mem=18_056),
    "inception_tr_bs1_i2_dmaxpool": Expected(cycles=14_962, total_memory=11_191_284, max_tile_mem=18_632),
    "inception_tr_bs1_i3_dmaxpool": Expected(cycles=33_476, total_memory=11_912_606, max_tile_mem=18_904),
    "inception_tr_bs1_i4_cmaxpool": Expected(cycles=30_281, total_memory=7_989_140, max_tile_mem=17_976),
    "inception_tr_bs1_i5_dmaxpool": Expected(cycles=25_710, total_memory=9_574_597, max_tile_mem=18_056),
    "inception_tr_bs1_i6_dmaxpool": Expected(cycles=15_704, total_memory=6_544_809, max_tile_mem=17_448),
    "inception_tr_bs1_i9_dmax_pool": Expected(cycles=50_486, total_memory=8_949_874, max_tile_mem=17_160),
    "inception_tr_bs1_pool1": Expected(cycles=17_706, total_memory=18_465_372, max_tile_mem=22_248),
    "mobilenet_conv1_1": Expected(cycles=45_119, total_memory=53_196_742, max_tile_mem=136_472),
	"mobilenet_conv_pw_1_1": Expected(cycles=38_694, total_memory=65_323_621, max_tile_mem=141_720),
    "mobilenet_conv_pw_12_1": Expected(cycles=38_251, total_memory=47_011_659, max_tile_mem=135_320),
    "mobilenet_conv_pw_13_1": Expected(cycles=59_436, total_memory=54_187_081, max_tile_mem=135_320),
    "mobilenet_conv_pw_2_1": Expected(cycles=35_981, total_memory=53_229_664, max_tile_mem=136_408),
    "mobilenet_conv_pw_3_1": Expected(cycles=57_700, total_memory=63_444_427, max_tile_mem=136_408),
    "mobilenet_conv_pw_4_1": Expected(cycles=35_976, total_memory=45_596_206, max_tile_mem=136_600),
    "mobilenet_conv_pw_5_1": Expected(cycles=58_550, total_memory=55_003_515, max_tile_mem=136_600),
    "mobilenet_conv_pw_6_1": Expected(cycles=36_635, total_memory=34_148_007, max_tile_mem=136_728),
    "mobilenet_conv_pw_7_1": Expected(cycles=58_011, total_memory=51_816_736, max_tile_mem=137_688),
    "mobilenet_depthwise_11": Expected(cycles=48_369, total_memory=23_167_336, max_tile_mem=132_696),
    "mobilenet_depthwise_12": Expected(cycles=46_778, total_memory=19_214_144, max_tile_mem=132_664),
    "mobilenet_depthwise_1": Expected(cycles=133_228, total_memory=62_757_315, max_tile_mem=141_848),
    "mobilenet_depthwise_2": Expected(cycles=137_022, total_memory=67_864_275, max_tile_mem=136_472),
    "mobilenet_depthwise_3": Expected(cycles=75_773, total_memory=41_939_153, max_tile_mem=136_472),
    "mobilenet_depthwise_4": Expected(cycles=90_795, total_memory=39_986_938, max_tile_mem=134_232),
    "mobilenet_depthwise_5": Expected(cycles=54_527, total_memory=29_159_554, max_tile_mem=134_296),
    "mobilenet_depthwise_6": Expected(cycles=33_839, total_memory=18_335_073, max_tile_mem=132_824),
    "mobilenet_depthwise": Expected(cycles=170_349, total_memory=63_609_280, max_tile_mem=136_472),
    "resnet50_tr_bs1_bm128L0A0_reduce": Expected(cycles=2_709, total_memory=4_391_559, max_tile_mem=16_576),
    "resnet50_tr_bs1_bm128L0_reduce": Expected(cycles=3_092, total_memory=5_577_650, max_tile_mem=17_072),
    "resnet50_tr_bs1_bm64L0A0_reduce": Expected(cycles=2_893, total_memory=4_343_628, max_tile_mem=16_744),
    "resnet50_tr_bs1_cnv_reduce": Expected(cycles=3_408, total_memory=6_069_210, max_tile_mem=17_728),
    "resnet50_tr_bs4_bm128L0A0": Expected(cycles=44_919, total_memory=52_763_705, max_tile_mem=136_600),
    "resnet50_tr_bs4_bm128L0A1": Expected(cycles=98_421, total_memory=75_705_889, max_tile_mem=138_488),
    "resnet50_tr_bs4_bm128L0A2": Expected(cycles=54_346, total_memory=56_891_364, max_tile_mem=136_600),
    "resnet50_tr_bs4_bm128L0_projection": Expected(cycles=99_312, total_memory=105_789_058, max_tile_mem=141_720),
    "resnet50_tr_bs4_bm128L1A0": Expected(cycles=55_553, total_memory=56_893_779, max_tile_mem=136_600),
    "resnet50_tr_bs4_bm256L0A0": Expected(cycles=42_024, total_memory=40_796_362, max_tile_mem=136_728),
    "resnet50_tr_bs4_bm256L0A1": Expected(cycles=101_735, total_memory=93_146_947, max_tile_mem=140_952),
    "resnet50_tr_bs4_bm256L0A2": Expected(cycles=55_243, total_memory=53_163_684, max_tile_mem=136_728),
    "resnet50_tr_bs4_bm256L0_projection": Expected(cycles=99_980, total_memory=73_512_415, max_tile_mem=144_280),
    "resnet50_tr_bs4_bm256L1A0": Expected(cycles=56_331, total_memory=52_922_308, max_tile_mem=136_728),
    "resnet50_tr_bs4_bm512L0A0": Expected(cycles=41_336, total_memory=50_204_794, max_tile_mem=135_320),
    "resnet50_tr_bs4_bm512L0A1": Expected(cycles=105_426, total_memory=102_973_935, max_tile_mem=143_640),
    "resnet50_tr_bs4_bm512L0A2": Expected(cycles=57_816, total_memory=55_070_873, max_tile_mem=135_320),
    "resnet50_tr_bs4_bm512L0_projection": Expected(cycles=99_667, total_memory=95_696_726, max_tile_mem=140_504),
    "resnet50_tr_bs4_bm512L1A0": Expected(cycles=58_104, total_memory=56_363_947, max_tile_mem=140_504),
    "resnet50_tr_bs4_bm64L0A1": Expected(cycles=92_430, total_memory=79_929_262, max_tile_mem=137_240),
    "resnet50_tr_bs4_bm64L0": Expected(cycles=27_626, total_memory=34_697_588, max_tile_mem=133_784),
    "resnet50_tr_bs4_bm64L0_projection": Expected(cycles=56_272, total_memory=66_279_240, max_tile_mem=141_720),
    "resnet50_tr_bs4_bm64L1A0": Expected(cycles=57_556, total_memory=66_187_297, max_tile_mem=141_720),
    "resnet50_tr_bs4_cnv": Expected(cycles=164_042, total_memory=113_891_184, max_tile_mem=152_600),
    "vgg16_tr_bs4_v1L0": Expected(cycles=201_005, total_memory=184_659_694, max_tile_mem=218_136),
    "vgg16_tr_bs4_v2L0": Expected(cycles=557_448, total_memory=252_381_578, max_tile_mem=228_026),
    "vgg16_tr_bs4_v2L1": Expected(cycles=1_154_183, total_memory=256_640_558, max_tile_mem=259_108),
    "vgg16_tr_bs4_v3L0": Expected(cycles=520_570, total_memory=219_249_387, max_tile_mem=191_866),
    "vgg16_tr_bs4_v3L1": Expected(cycles=1_014_964, total_memory=225_564_449, max_tile_mem=200_932),
    "vgg16_tr_bs4_v4L0": Expected(cycles=542_440, total_memory=191_307_876, max_tile_mem=202_712),
    "vgg16_tr_bs4_v4L1": Expected(cycles=1_067_711, total_memory=198_060_628, max_tile_mem=200_466),
    "vgg16_tr_bs4_v5L0": Expected(cycles=317_605, total_memory=245_475_989, max_tile_mem=220_338),
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
