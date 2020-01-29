#!/usr/bin/env python3

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
# fmt: off
EXPECTED_RESULTS = {
    "alexnet_tr_bs4_layer1": Expected(cycles=159_823, total_memory=129_644_360, max_tile_mem=174_104),
    "alexnet_tr_bs4_layer11": Expected(cycles=84_425, total_memory=58_266_621, max_tile_mem=138_584),
    "alexnet_tr_bs4_layer4": Expected(cycles=150_063, total_memory=86_616_328, max_tile_mem=138_872),
    "alexnet_tr_bs4_layer7": Expected(cycles=83_241, total_memory=64_700_815, max_tile_mem=138_840),
    "alexnet_tr_bs4_layer9": Expected(cycles=107_222, total_memory=80_373_091, max_tile_mem=141_912),
    "bert_ffn1_128x1024x4096": Expected(cycles=38_359, total_memory=106_174_216, max_tile_mem=144_920),
    "bert_ffn1_128x768x3072": Expected(cycles=24_850, total_memory=64_505_912, max_tile_mem=138_968),
    "bert_ffn1_384x1024x4096": Expected(cycles=88_244, total_memory=193_653_604, max_tile_mem=177_016),
    "bert_ffn1_384x768x3072": Expected(cycles=55_952, total_memory=154_592_094, max_tile_mem=146_648),
    "bert_ffn1_512x1024x4096": Expected(cycles=111_232, total_memory=212_139_560, max_tile_mem=176_568),
    "bert_ffn1_512x768x3072": Expected(cycles=70_707, total_memory=152_411_436, max_tile_mem=151_832),
    "bert_ffn2_128x3072x768": Expected(cycles=26_920, total_memory=64_677_320, max_tile_mem=138_968),
    "bert_ffn2_128x4096x1024": Expected(cycles=41_391, total_memory=106_458_076, max_tile_mem=144_920),
    "bert_ffn2_384x3072x768": Expected(cycles=57_739, total_memory=154_636_270, max_tile_mem=146_648),
    "bert_ffn2_384x4096x1024": Expected(cycles=93_217, total_memory=193_822_612, max_tile_mem=175_904),
    "bert_ffn2_512x3072x768": Expected(cycles=72_015, total_memory=152_312_264, max_tile_mem=151_832),
    "bert_ffn2_512x4096x1024": Expected(cycles=131_557, total_memory=212_289_200, max_tile_mem=182_452),
    "bert_grouped_12x128x64x128": Expected(cycles=3_785, total_memory=11_044_301, max_tile_mem=132_472),
    "bert_grouped_12x384x64x384": Expected(cycles=10_527, total_memory=36_798_402, max_tile_mem=137_240),
    "bert_grouped_12x512x64x512": Expected(cycles=15_241, total_memory=56_044_630, max_tile_mem=142_040),
    "bert_grouped_16x128x64x128": Expected(cycles=4_265, total_memory=13_144_645, max_tile_mem=133_144),
    "bert_grouped_16x384x64x384": Expected(cycles=12_537, total_memory=61_394_496, max_tile_mem=139_288),
    "bert_grouped_16x512x64x512": Expected(cycles=20_867, total_memory=73_325_636, max_tile_mem=147_512),
    "bert_kqv_1024x128x3072": Expected(cycles=30_613, total_memory=82_334_776, max_tile_mem=141_464),
    "bert_kqv_1024x384x3072": Expected(cycles=70_586, total_memory=196_298_724, max_tile_mem=177_336),
    "bert_kqv_1024x512x3072": Expected(cycles=89_012, total_memory=166_910_120, max_tile_mem=157_336),
    "bert_kqv_768x128x2304": Expected(cycles=20_352, total_memory=51_561_398, max_tile_mem=138_904),
    "bert_kqv_768x384x2304": Expected(cycles=45_655, total_memory=121_508_840, max_tile_mem=143_384),
    "bert_kqv_768x512x2304": Expected(cycles=57_659, total_memory=156_577_128, max_tile_mem=147_480),
    "conv_5x200_1_in_100_out_bs1440": Expected(cycles=338_710, total_memory=196_977_743, max_tile_mem=261_752),
    "embedding_small": Expected(cycles=175_843, total_memory=71_097_812, max_tile_mem=87_544),
    "embedding_vlarge": Expected(cycles=148_073, total_memory=59_138_138, max_tile_mem=77_080),
    "fc_layer_1440x100x200": Expected(cycles=8_579, total_memory=18_810_324, max_tile_mem=132_696),
    "fc_layer_1440x200x400": Expected(cycles=13_953, total_memory=44_357_698, max_tile_mem=135_896),
    "fc_layer_16x1324x100": Expected(cycles=3_956, total_memory=6_388_106, max_tile_mem=131_608),
    "fc_layer_80_1324_100": Expected(cycles=6_033, total_memory=10_865_613, max_tile_mem=132_536),
    "gemm_1000x256x10000": Expected(cycles=128_198, total_memory=198_901_814, max_tile_mem=197_976),
    "gemm_1000x256x20000": Expected(cycles=391_802, total_memory=221_175_496, max_tile_mem=205_080),
    "gemm_1000x256x30000": Expected(cycles=540_155, total_memory=317_495_904, max_tile_mem=314_856),
    "gemm_1000x512x10000": Expected(cycles=345_839, total_memory=191_954_214, max_tile_mem=205_080),
    "gemm_1000x512x20000": Expected(cycles=720_412, total_memory=272_463_972, max_tile_mem=299_032),
    "gemm_1000x512x30000": Expected(cycles=1_007_510, total_memory=302_306_830, max_tile_mem=317_208),
    "gemm_1000x64x10000": Expected(cycles=40_742, total_memory=113_543_544, max_tile_mem=164_888),
    "gemm_1000x64x20000": Expected(cycles=74_187, total_memory=183_767_682, max_tile_mem=198_072),
    "gemm_1000x64x30000": Expected(cycles=107_827, total_memory=255_221_512, max_tile_mem=231_864),
    "gemm_200x256x10000": Expected(cycles=41_195, total_memory=66_877_272, max_tile_mem=144_376),
    "gemm_200x256x20000": Expected(cycles=66_593, total_memory=109_101_928, max_tile_mem=144_376),
    "gemm_200x256x30000": Expected(cycles=86_707, total_memory=148_106_596, max_tile_mem=151_192),
    "gemm_200x512x10000": Expected(cycles=68_530, total_memory=111_178_660, max_tile_mem=157_816),
    "gemm_200x512x20000": Expected(cycles=115_607, total_memory=179_193_032, max_tile_mem=190_424),
    "gemm_200x512x30000": Expected(cycles=177_748, total_memory=204_662_036, max_tile_mem=207_032),
    "gemm_200x64x10000": Expected(cycles=13_697, total_memory=38_329_018, max_tile_mem=138_008),
    "gemm_200x64x20000": Expected(cycles=21_835, total_memory=51_513_084, max_tile_mem=144_376),
    "gemm_200x64x30000": Expected(cycles=28_981, total_memory=71_229_976, max_tile_mem=150_936),
    "gemm_600x256x10000": Expected(cycles=90_228, total_memory=150_039_130, max_tile_mem=151_192),
    "gemm_600x256x20000": Expected(cycles=164_050, total_memory=218_418_212, max_tile_mem=238_872),
    "gemm_600x256x30000": Expected(cycles=271_600, total_memory=209_552_020, max_tile_mem=191_628),
    "gemm_600x512x10000": Expected(cycles=177_473, total_memory=206_403_180, max_tile_mem=238_872),
    "gemm_600x512x20000": Expected(cycles=355_726, total_memory=220_878_576, max_tile_mem=210_344),
    "gemm_600x512x30000": Expected(cycles=561_487, total_memory=230_062_256, max_tile_mem=213_612),
    "gemm_600x64x10000": Expected(cycles=26_894, total_memory=94_522_384, max_tile_mem=151_128),
    "gemm_600x64x20000": Expected(cycles=49_624, total_memory=180_443_368, max_tile_mem=171_352),
    "gemm_600x64x30000": Expected(cycles=72_452, total_memory=182_121_146, max_tile_mem=190_456),
    "inception_tr_bs4_i10_a1x1": Expected(cycles=56_146, total_memory=64_372_412, max_tile_mem=136_600),
    "inception_tr_bs4_i10_b1x1": Expected(cycles=62_838, total_memory=51_720_743, max_tile_mem=136_600),
    "inception_tr_bs4_i10_c1x1": Expected(cycles=71_683, total_memory=58_238_092, max_tile_mem=143_384),
    "inception_tr_bs4_i10_d1x1": Expected(cycles=40_977, total_memory=45_557_209, max_tile_mem=136_600),
    "inception_tr_bs4_i1_a1x1": Expected(cycles=30_505, total_memory=36_956_948, max_tile_mem=134_232),
    "inception_tr_bs4_i1_b1x1": Expected(cycles=26_940, total_memory=31_222_479, max_tile_mem=134_232),
    "inception_tr_bs4_i1_b5x5": Expected(cycles=88_014, total_memory=59_840_330, max_tile_mem=140_696),
    "inception_tr_bs4_i1_c3x3a": Expected(cycles=67_276, total_memory=54_994_820, max_tile_mem=138_008),
    "inception_tr_bs4_i1_c3x3b": Expected(cycles=93_286, total_memory=73_390_872, max_tile_mem=141_176),
    "inception_tr_bs4_i1_db1x1": Expected(cycles=22_521, total_memory=26_453_604, max_tile_mem=134_232),
    "inception_tr_bs4_i2_a1x1": Expected(cycles=36_024, total_memory=42_818_178, max_tile_mem=135_256),
    "inception_tr_bs4_i2_b1x1": Expected(cycles=31_313, total_memory=37_370_465, max_tile_mem=135_256),
    "inception_tr_bs4_i3_a1x1": Expected(cycles=38_694, total_memory=45_453_606, max_tile_mem=135_832),
    "inception_tr_bs4_i3_b1x1": Expected(cycles=32_915, total_memory=40_164_052, max_tile_mem=135_832),
    "inception_tr_bs4_i4_a3x3": Expected(cycles=248_837, total_memory=162_447_122, max_tile_mem=174_264),
    "inception_tr_bs4_i4_b3x3b": Expected(cycles=51_354, total_memory=50_624_252, max_tile_mem=141_464),
    "inception_tr_bs4_i5_a1x1": Expected(cycles=56_438, total_memory=64_727_681, max_tile_mem=136_600),
    "inception_tr_bs4_i5_b1x1": Expected(cycles=43_149, total_memory=49_067_749, max_tile_mem=136_024),
    "inception_tr_bs4_i5_b1x7": Expected(cycles=47_474, total_memory=43_198_407, max_tile_mem=136_856),
    "inception_tr_bs4_i5_b7x1": Expected(cycles=54_195, total_memory=38_852_191, max_tile_mem=137_624),
    "inception_tr_bs4_i5_c1x7b": Expected(cycles=61_376, total_memory=56_098_109, max_tile_mem=139_736),
    "inception_tr_bs4_i5_c7x1a": Expected(cycles=45_188, total_memory=32_283_296, max_tile_mem=136_920),
    "inception_tr_bs4_i6_b1x1": Expected(cycles=49_384, total_memory=56_879_691, max_tile_mem=137_240),
    "inception_tr_bs4_i6_b1x7": Expected(cycles=62_696, total_memory=55_382_514, max_tile_mem=137_624),
    "inception_tr_bs4_i6_b7x1": Expected(cycles=61_168, total_memory=42_009_331, max_tile_mem=138_712),
    "inception_tr_bs4_i6_c1x7b": Expected(cycles=67_395, total_memory=64_477_727, max_tile_mem=138_968),
    "inception_tr_bs4_i6_c7x1a": Expected(cycles=57_771, total_memory=38_281_564, max_tile_mem=138_264),
    "inception_tr_bs4_i6_c7x1b": Expected(cycles=55_926, total_memory=37_569_371, max_tile_mem=137_624),
    "inception_tr_bs4_i7_b1x7": Expected(cycles=82_435, total_memory=70_817_232, max_tile_mem=140_696),
    "inception_tr_bs4_i7_b7x1": Expected(cycles=70_756, total_memory=46_130_209, max_tile_mem=140_888),
    "inception_tr_bs4_i8_a3x3": Expected(cycles=64_478, total_memory=60_963_821, max_tile_mem=140_344),
    "inception_tr_bs4_i8_b3x3": Expected(cycles=42_807, total_memory=38_119_570, max_tile_mem=140_344),
    "inception_tr_bs4_i9_a1x1": Expected(cycles=41_243, total_memory=45_782_561, max_tile_mem=136_600),
    "inception_tr_bs4_i9_ba1x3": Expected(cycles=44_947, total_memory=47_244_579, max_tile_mem=137_240),
    "inception_tr_bs4_i9_bb3x1": Expected(cycles=38_712, total_memory=29_687_113, max_tile_mem=135_192),
    "inception_tr_bs4_i9_c1x1": Expected(cycles=51_139, total_memory=43_485_087, max_tile_mem=136_600),
    "inception_tr_bs4_i9_c3x3": Expected(cycles=95_998, total_memory=85_838_476, max_tile_mem=136_856),
    "inception_tr_bs4_i9_d1x1": Expected(cycles=31_101, total_memory=34_607_465, max_tile_mem=134_520),
    "inception_tr_bs4_layer1": Expected(cycles=65_491, total_memory=68_861_695, max_tile_mem=140_696),
    "inception_tr_bs4_layer10": Expected(cycles=412_134, total_memory=195_762_845, max_tile_mem=210_264),
    "inception_tr_bs4_layer3": Expected(cycles=154_996, total_memory=141_660_486, max_tile_mem=140_696),
    "inception_tr_bs4_layer5": Expected(cycles=263_143, total_memory=228_944_000, max_tile_mem=206_430),
    "inception_tr_bs4_layer8": Expected(cycles=112_962, total_memory=138_877_597, max_tile_mem=153_976),
    "mobilenet_conv1_1": Expected(cycles=47_338, total_memory=48_148_661, max_tile_mem=136_920),
    "mobilenet_conv_pw_12_1": Expected(cycles=41_948, total_memory=47_843_638, max_tile_mem=133_208),
    "mobilenet_conv_pw_13_1": Expected(cycles=66_629, total_memory=53_869_497, max_tile_mem=135_320),
    "mobilenet_conv_pw_1_1": Expected(cycles=43_044, total_memory=64_970_721, max_tile_mem=141_720),
    "mobilenet_conv_pw_2_1": Expected(cycles=40_898, total_memory=53_441_688, max_tile_mem=136_408),
    "mobilenet_conv_pw_3_1": Expected(cycles=65_742, total_memory=85_218_537, max_tile_mem=136_408),
    "mobilenet_conv_pw_4_1": Expected(cycles=40_927, total_memory=47_882_698, max_tile_mem=136_408),
    "mobilenet_conv_pw_5_1": Expected(cycles=65_897, total_memory=54_693_427, max_tile_mem=136_600),
    "mobilenet_conv_pw_6_1": Expected(cycles=41_927, total_memory=44_759_397, max_tile_mem=136_728),
    "mobilenet_conv_pw_7_1": Expected(cycles=65_140, total_memory=51_530_008, max_tile_mem=137_688),
    "mobilenet_depthwise": Expected(cycles=175_428, total_memory=63_808_684, max_tile_mem=136_472),
    "mobilenet_depthwise_1": Expected(cycles=135_472, total_memory=62_858_754, max_tile_mem=141_848),
    "mobilenet_depthwise_11": Expected(cycles=55_573, total_memory=23_828_155, max_tile_mem=132_696),
    "mobilenet_depthwise_12": Expected(cycles=55_313, total_memory=21_883_306, max_tile_mem=133_784),
    "mobilenet_depthwise_2": Expected(cycles=155_791, total_memory=62_307_172, max_tile_mem=137_472),
    "mobilenet_depthwise_3": Expected(cycles=85_852, total_memory=42_854_002, max_tile_mem=136_472),
    "mobilenet_depthwise_4": Expected(cycles=94_796, total_memory=39_936_575, max_tile_mem=134_232),
    "mobilenet_depthwise_5": Expected(cycles=58_220, total_memory=33_353_081, max_tile_mem=134_296),
    "mobilenet_depthwise_6": Expected(cycles=31_424, total_memory=18_755_119, max_tile_mem=132_824),
    "resnet50_tr_bs4_bm128L0_projection": Expected(cycles=108_877, total_memory=105_757_178, max_tile_mem=141_720),
    "resnet50_tr_bs4_bm128L0A0": Expected(cycles=48_241, total_memory=59_553_133, max_tile_mem=136_408),
    "resnet50_tr_bs4_bm128L0A1": Expected(cycles=101_192, total_memory=75_943_637, max_tile_mem=138_488),
    "resnet50_tr_bs4_bm128L0A2": Expected(cycles=61_774, total_memory=57_308_476, max_tile_mem=136_600),
    "resnet50_tr_bs4_bm128L1A0": Expected(cycles=63_063, total_memory=57_314_059, max_tile_mem=136_600),
    "resnet50_tr_bs4_bm256L0_projection": Expected(cycles=107_817, total_memory=92_606_790, max_tile_mem=142_040),
    "resnet50_tr_bs4_bm256L0A0": Expected(cycles=45_709, total_memory=51_675_213, max_tile_mem=136_728),
    "resnet50_tr_bs4_bm256L0A1": Expected(cycles=106_079, total_memory=74_536_444, max_tile_mem=139_352),
    "resnet50_tr_bs4_bm256L0A2": Expected(cycles=62_606, total_memory=53_201_300, max_tile_mem=136_728),
    "resnet50_tr_bs4_bm256L1A0": Expected(cycles=63_693, total_memory=53_287_764, max_tile_mem=136_728),
    "resnet50_tr_bs4_bm512L0_projection": Expected(cycles=110_464, total_memory=97_757_825, max_tile_mem=143_640),
    "resnet50_tr_bs4_bm512L0A0": Expected(cycles=45_229, total_memory=50_908_019, max_tile_mem=133_208),
    "resnet50_tr_bs4_bm512L0A1": Expected(cycles=111_486, total_memory=102_574_199, max_tile_mem=143_640),
    "resnet50_tr_bs4_bm512L0A2": Expected(cycles=64_969, total_memory=79_493_862, max_tile_mem=137_368),
    "resnet50_tr_bs4_bm512L1A0": Expected(cycles=64_736, total_memory=57_620_831, max_tile_mem=137_368),
    "resnet50_tr_bs4_bm64L0": Expected(cycles=30_482, total_memory=35_308_817, max_tile_mem=133_752),
    "resnet50_tr_bs4_bm64L0_projection": Expected(cycles=63_058, total_memory=66_288_800, max_tile_mem=141_720),
    "resnet50_tr_bs4_bm64L0A1": Expected(cycles=96_338, total_memory=79_941_937, max_tile_mem=141_848),
    "resnet50_tr_bs4_bm64L1A0": Expected(cycles=64_429, total_memory=66_196_249, max_tile_mem=141_720),
    "resnet50_tr_bs4_cnv": Expected(cycles=173_401, total_memory=113_331_115, max_tile_mem=152_600),
    "vgg16_tr_bs4_v1L0": Expected(cycles=211_971, total_memory=184_822_158, max_tile_mem=218_136),
    "vgg16_tr_bs4_v1L1": Expected(cycles=1_264_274, total_memory=359_532_331, max_tile_mem=328_856),
    "vgg16_tr_bs4_v2L0": Expected(cycles=575_562, total_memory=233_699_851, max_tile_mem=223_688),
    "vgg16_tr_bs4_v2L1": Expected(cycles=1_167_633, total_memory=265_111_649, max_tile_mem=270_842),
    "vgg16_tr_bs4_v3L0": Expected(cycles=535_194, total_memory=213_980_544, max_tile_mem=191_100),
    "vgg16_tr_bs4_v3L1": Expected(cycles=1_050_156, total_memory=233_520_506, max_tile_mem=232_874),
    "vgg16_tr_bs4_v4L0": Expected(cycles=567_062, total_memory=205_887_682, max_tile_mem=202_836),
    "vgg16_tr_bs4_v4L1": Expected(cycles=1_115_371, total_memory=203_502_808, max_tile_mem=206_626),
    "vgg16_tr_bs4_v5L0": Expected(cycles=322_418, total_memory=148_166_847, max_tile_mem=148_120),
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
            pc_diff = actual_value / expected_value * 100
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
