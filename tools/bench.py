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
    "alexnet_tr_bs4_layer11": Expected(cycles=84_107, total_memory=58_288_797, max_tile_mem=138_584),
    "alexnet_tr_bs4_layer1": Expected(cycles=159_821, total_memory=129_717_244, max_tile_mem=174_104),
    "alexnet_tr_bs4_layer4": Expected(cycles=148_414, total_memory=86_539_672, max_tile_mem=138_872),
    "alexnet_tr_bs4_layer7": Expected(cycles=82_837, total_memory=64_579_263, max_tile_mem=138_840),
    "alexnet_tr_bs4_layer9": Expected(cycles=107_220, total_memory=80_418_435, max_tile_mem=141_912),
    "bert_ffn1_128x1024x4096": Expected(cycles=119_045, total_memory=105_640_292, max_tile_mem=144_920),
    "bert_ffn1_128x768x3072": Expected(cycles=78_121, total_memory=64_008_548, max_tile_mem=138_968),
    "bert_ffn1_384x1024x4096": Expected(cycles=278_055, total_memory=193_423_152, max_tile_mem=177_016),
    "bert_ffn1_384x768x3072": Expected(cycles=175_722, total_memory=154_575_530, max_tile_mem=146_648),
    "bert_ffn1_512x1024x4096": Expected(cycles=357_643, total_memory=212_121_358, max_tile_mem=175_596),
    "bert_ffn1_512x768x3072": Expected(cycles=221_885, total_memory=152_397_944, max_tile_mem=151_832),
    "bert_ffn2_128x3072x768": Expected(cycles=82_626, total_memory=63_607_988, max_tile_mem=138_968),
    "bert_ffn2_128x4096x1024": Expected(cycles=123_734, total_memory=105_011_688, max_tile_mem=144_920),
    "bert_ffn2_384x3072x768": Expected(cycles=178_484, total_memory=154_622_106, max_tile_mem=146_648),
    "bert_ffn2_384x4096x1024": Expected(cycles=285_356, total_memory=193_404_032, max_tile_mem=174_970),
    "bert_ffn2_512x3072x768": Expected(cycles=222_477, total_memory=152_300_404, max_tile_mem=151_832),
    "bert_ffn2_512x4096x1024": Expected(cycles=366_344, total_memory=212_265_980, max_tile_mem=203_800),
    "bert_grouped_12x128x64x128": Expected(cycles=15_046, total_memory=11_038_055, max_tile_mem=132_472),
    "bert_grouped_12x384x64x384": Expected(cycles=40_599, total_memory=36_791_438, max_tile_mem=137_240),
    "bert_grouped_12x512x64x512": Expected(cycles=60_463, total_memory=56_037_738, max_tile_mem=142_040),
    "bert_grouped_16x128x64x128": Expected(cycles=16_979, total_memory=13_138_447, max_tile_mem=133_144),
    "bert_grouped_16x384x64x384": Expected(cycles=50_173, total_memory=61_387_532, max_tile_mem=139_288),
    "bert_grouped_16x512x64x512": Expected(cycles=79_455, total_memory=73_319_248, max_tile_mem=147_512),
    "bert_kqv_128x1024x3072": Expected(cycles=94_745, total_memory=81_836_516, max_tile_mem=141_464),
    "bert_kqv_128x768x2304": Expected(cycles=64_223, total_memory=51_169_922, max_tile_mem=138_904),
    "bert_kqv_384x1024x3072": Expected(cycles=216_733, total_memory=196_283_184, max_tile_mem=179_352),
    "bert_kqv_384x768x2304": Expected(cycles=143_581, total_memory=121_492_276, max_tile_mem=143_384),
    "bert_kqv_512x1024x3072": Expected(cycles=274_770, total_memory=166_894_580, max_tile_mem=157_336),
    "bert_kqv_512x768x2304": Expected(cycles=179_600, total_memory=156_563_636, max_tile_mem=147_480),
    "bert_proj_128x1024x1024": Expected(cycles=46_888, total_memory=53_551_802, max_tile_mem=135_192),
    "bert_proj_128x768x768": Expected(cycles=31_913, total_memory=32_154_235, max_tile_mem=135_192),
    "bert_proj_384x1024x1024": Expected(cycles=98_631, total_memory=77_975_714, max_tile_mem=142_040),
    "bert_proj_384x768x768": Expected(cycles=64_574, total_memory=48_652_484, max_tile_mem=137_240),
    "bert_proj_512x1024x1024": Expected(cycles=121_839, total_memory=98_548_422, max_tile_mem=142_040),
    "bert_proj_512x768x768": Expected(cycles=78_772, total_memory=61_093_112, max_tile_mem=139_288),
    "conv_5x200_1_in_100_out_bs1440": Expected(cycles=338_708, total_memory=197_070_163, max_tile_mem=261_752),
    "embedding_small": Expected(cycles=175_841, total_memory=71_097_744, max_tile_mem=87_544),
    "embedding_vlarge": Expected(cycles=148_071, total_memory=59_240_206, max_tile_mem=77_080),
    "fc_layer_1440x100x200": Expected(cycles=25_947, total_memory=18_810_272, max_tile_mem=132_696),
    "fc_layer_1440x200x400": Expected(cycles=44_819, total_memory=44_357_646, max_tile_mem=135_896),
    "fc_layer_16x1324x100": Expected(cycles=12_335, total_memory=6_388_054, max_tile_mem=131_608),
    "fc_layer_80_1324_100": Expected(cycles=17_717, total_memory=10_865_321, max_tile_mem=132_536),
    "gemm_1000x256x10000": Expected(cycles=126_544, total_memory=198_078_754, max_tile_mem=197_464),
    "gemm_1000x256x20000": Expected(cycles=382_472, total_memory=210_667_968, max_tile_mem=192_136),
    "gemm_1000x256x30000": Expected(cycles=526_601, total_memory=301_868_376, max_tile_mem=314_840),
    "gemm_1000x512x10000": Expected(cycles=336_509, total_memory=181_402_742, max_tile_mem=182_040),
    "gemm_1000x512x20000": Expected(cycles=720_410, total_memory=272_547_036, max_tile_mem=299_032),
    "gemm_1000x512x30000": Expected(cycles=1_007_508, total_memory=302_390_982, max_tile_mem=317_208),
    "gemm_1000x64x10000": Expected(cycles=40_003, total_memory=113_111_216, max_tile_mem=164_888),
    "gemm_1000x64x20000": Expected(cycles=73_172, total_memory=183_338_622, max_tile_mem=198_072),
    "gemm_1000x64x30000": Expected(cycles=106_548, total_memory=254_793_104, max_tile_mem=281_864),
    "gemm_200x256x10000": Expected(cycles=40_901, total_memory=66_746_592, max_tile_mem=144_376),
    "gemm_200x256x20000": Expected(cycles=66_356, total_memory=108_971_224, max_tile_mem=144_376),
    "gemm_200x256x30000": Expected(cycles=86_470, total_memory=147_975_892, max_tile_mem=150_936),
    "gemm_200x512x10000": Expected(cycles=68_236, total_memory=110_815_220, max_tile_mem=157_816),
    "gemm_200x512x20000": Expected(cycles=115_313, total_memory=178_829_552, max_tile_mem=190_424),
    "gemm_200x512x30000": Expected(cycles=177_454, total_memory=204_298_548, max_tile_mem=206_776),
    "gemm_200x64x10000": Expected(cycles=13_004, total_memory=38_009_380, max_tile_mem=138_008),
    "gemm_200x64x20000": Expected(cycles=21_634, total_memory=51_556_980, max_tile_mem=144_376),
    "gemm_200x64x30000": Expected(cycles=28_780, total_memory=71_273_872, max_tile_mem=150_936),
    "gemm_600x256x10000": Expected(cycles=88_652, total_memory=148_729_864, max_tile_mem=183_704),
    "gemm_600x256x20000": Expected(cycles=161_371, total_memory=207_557_484, max_tile_mem=206_104),
    "gemm_600x256x30000": Expected(cycles=271_598, total_memory=209_636_396, max_tile_mem=191_640),
    "gemm_600x512x10000": Expected(cycles=174_734, total_memory=195_233_572, max_tile_mem=206_104),
    "gemm_600x512x20000": Expected(cycles=355_724, total_memory=220_959_776, max_tile_mem=210_344),
    "gemm_600x512x30000": Expected(cycles=561_485, total_memory=230_145_984, max_tile_mem=213_616),
    "gemm_600x64x10000": Expected(cycles=26_183, total_memory=94_145_256, max_tile_mem=151_128),
    "gemm_600x64x20000": Expected(cycles=48_649, total_memory=180_066_240, max_tile_mem=203_864),
    "gemm_600x64x30000": Expected(cycles=71_212, total_memory=181_744_344, max_tile_mem=190_456),
    "inception_tr_bs4_i10_a1x1": Expected(cycles=55_422, total_memory=64_173_020, max_tile_mem=136_600),
    "inception_tr_bs4_i10_b1x1": Expected(cycles=62_119, total_memory=51_568_813, max_tile_mem=136_600),
    "inception_tr_bs4_i10_c1x1": Expected(cycles=70_942, total_memory=58_097_080, max_tile_mem=143_384),
    "inception_tr_bs4_i10_d1x1": Expected(cycles=40_356, total_memory=45_385_237, max_tile_mem=136_600),
    "inception_tr_bs4_i1_a1x1": Expected(cycles=28_952, total_memory=36_644_284, max_tile_mem=134_232),
    "inception_tr_bs4_i1_b1x1": Expected(cycles=26_048, total_memory=30_994_435, max_tile_mem=134_232),
    "inception_tr_bs4_i1_b5x5": Expected(cycles=86_717, total_memory=59_706_524, max_tile_mem=140_696),
    "inception_tr_bs4_i1_c3x3a": Expected(cycles=65_895, total_memory=54_724_396, max_tile_mem=138_008),
    "inception_tr_bs4_i1_c3x3b": Expected(cycles=91_828, total_memory=73_401_312, max_tile_mem=141_176),
    "inception_tr_bs4_i1_db1x1": Expected(cycles=21_609, total_memory=26_229_628, max_tile_mem=134_232),
    "inception_tr_bs4_i2_a1x1": Expected(cycles=34_988, total_memory=42_547_918, max_tile_mem=135_256),
    "inception_tr_bs4_i2_b1x1": Expected(cycles=30_373, total_memory=37_137_485, max_tile_mem=135_256),
    "inception_tr_bs4_i3_a1x1": Expected(cycles=36_903, total_memory=45_113_702, max_tile_mem=135_832),
    "inception_tr_bs4_i3_b1x1": Expected(cycles=31_972, total_memory=39_938_128, max_tile_mem=135_832),
    "inception_tr_bs4_i4_a3x3": Expected(cycles=248_411, total_memory=162_455_014, max_tile_mem=174_264),
    "inception_tr_bs4_i4_b3x3b": Expected(cycles=51_352, total_memory=50_699_560, max_tile_mem=141_464),
    "inception_tr_bs4_i5_a1x1": Expected(cycles=54_874, total_memory=64_433_473, max_tile_mem=136_600),
    "inception_tr_bs4_i5_b1x1": Expected(cycles=41_620, total_memory=48_776_173, max_tile_mem=136_024),
    "inception_tr_bs4_i5_b1x7": Expected(cycles=46_304, total_memory=43_238_807, max_tile_mem=136_856),
    "inception_tr_bs4_i5_b7x1": Expected(cycles=54_193, total_memory=38_939_343, max_tile_mem=137_624),
    "inception_tr_bs4_i5_c1x7b": Expected(cycles=60_062, total_memory=55_957_389, max_tile_mem=139_736),
    "inception_tr_bs4_i5_c7x1a": Expected(cycles=44_618, total_memory=32_007_576, max_tile_mem=136_920),
    "inception_tr_bs4_i6_b1x1": Expected(cycles=47_830, total_memory=56_588_123, max_tile_mem=137_240),
    "inception_tr_bs4_i6_b1x7": Expected(cycles=61_358, total_memory=55_425_394, max_tile_mem=137_624),
    "inception_tr_bs4_i6_b7x1": Expected(cycles=61_166, total_memory=42_096_783, max_tile_mem=138_712),
    "inception_tr_bs4_i6_c1x7b": Expected(cycles=66_773, total_memory=64_376_331, max_tile_mem=138_968),
    "inception_tr_bs4_i6_c7x1a": Expected(cycles=57_291, total_memory=37_818_952, max_tile_mem=138_264),
    "inception_tr_bs4_i6_c7x1b": Expected(cycles=55_924, total_memory=37_656_671, max_tile_mem=137_624),
    "inception_tr_bs4_i7_b1x7": Expected(cycles=81_073, total_memory=70_854_848, max_tile_mem=140_696),
    "inception_tr_bs4_i7_b7x1": Expected(cycles=70_754, total_memory=46_217_073, max_tile_mem=140_888),
    "inception_tr_bs4_i8_a3x3": Expected(cycles=64_411, total_memory=60_986_189, max_tile_mem=140_344),
    "inception_tr_bs4_i8_b3x3": Expected(cycles=42_222, total_memory=38_055_838, max_tile_mem=140_344),
    "inception_tr_bs4_i9_a1x1": Expected(cycles=40_517, total_memory=45_139_095, max_tile_mem=136_600),
    "inception_tr_bs4_i9_ba1x3": Expected(cycles=44_945, total_memory=47_335_851, max_tile_mem=137_240),
    "inception_tr_bs4_i9_bb3x1": Expected(cycles=38_710, total_memory=29_710_049, max_tile_mem=135_192),
    "inception_tr_bs4_i9_c1x1": Expected(cycles=50_428, total_memory=42_815_067, max_tile_mem=136_600),
    "inception_tr_bs4_i9_c3x3": Expected(cycles=95_996, total_memory=85_913_424, max_tile_mem=136_856),
    "inception_tr_bs4_i9_d1x1": Expected(cycles=30_497, total_memory=34_005_285, max_tile_mem=134_520),
    "inception_tr_bs4_layer10": Expected(cycles=408_681, total_memory=195_576_653, max_tile_mem=206_392),
    "inception_tr_bs4_layer1": Expected(cycles=65_311, total_memory=68_950_823, max_tile_mem=140_696),
    "inception_tr_bs4_layer3": Expected(cycles=154_503, total_memory=141_716_450, max_tile_mem=140_696),
    "inception_tr_bs4_layer5": Expected(cycles=262_590, total_memory=228_977_868, max_tile_mem=206_474),
    "inception_tr_bs4_layer8": Expected(cycles=112_409, total_memory=138_957_985, max_tile_mem=153_976),
    "mobilenet_conv1_1": Expected(cycles=47_158, total_memory=48_260_525, max_tile_mem=136_920),
    "mobilenet_conv_pw_1_1": Expected(cycles=42_464, total_memory=65_061_829, max_tile_mem=141_720),
    "mobilenet_conv_pw_12_1": Expected(cycles=41_304, total_memory=47_332_668, max_tile_mem=133_208),
    "mobilenet_conv_pw_13_1": Expected(cycles=65_995, total_memory=53_924_425, max_tile_mem=135_320),
    "mobilenet_conv_pw_2_1": Expected(cycles=38_918, total_memory=52_967_008, max_tile_mem=136_408),
    "mobilenet_conv_pw_3_1": Expected(cycles=63_252, total_memory=85_268_849, max_tile_mem=136_408),
    "mobilenet_conv_pw_4_1": Expected(cycles=39_412, total_memory=47_563_954, max_tile_mem=136_408),
    "mobilenet_conv_pw_5_1": Expected(cycles=64_079, total_memory=54_740_859, max_tile_mem=136_600),
    "mobilenet_conv_pw_6_1": Expected(cycles=40_577, total_memory=44_584_781, max_tile_mem=136_728),
    "mobilenet_conv_pw_7_1": Expected(cycles=63_634, total_memory=51_567_904, max_tile_mem=137_688),
    "mobilenet_depthwise_11": Expected(cycles=55_571, total_memory=23_964_087, max_tile_mem=132_696),
    "mobilenet_depthwise_12": Expected(cycles=55_311, total_memory=22_019_094, max_tile_mem=133_784),
    "mobilenet_depthwise_1": Expected(cycles=135_470, total_memory=62_995_086, max_tile_mem=141_848),
    "mobilenet_depthwise_2": Expected(cycles=155_790, total_memory=62_443_248, max_tile_mem=136_472),
    "mobilenet_depthwise_3": Expected(cycles=86_310, total_memory=42_991_406, max_tile_mem=136_472),
    "mobilenet_depthwise_4": Expected(cycles=94_794, total_memory=40_071_071, max_tile_mem=134_232),
    "mobilenet_depthwise_5": Expected(cycles=58_218, total_memory=33_489_193, max_tile_mem=134_296),
    "mobilenet_depthwise_6": Expected(cycles=31_422, total_memory=18_896_823, max_tile_mem=132_824),
    "mobilenet_depthwise": Expected(cycles=175_426, total_memory=63_945_784, max_tile_mem=136_472),
    "resnet50_tr_bs4_bm128L0A0": Expected(cycles=47_334, total_memory=59_367_405, max_tile_mem=136_408),
    "resnet50_tr_bs4_bm128L0A1": Expected(cycles=101_190, total_memory=76_006_661, max_tile_mem=138_488),
    "resnet50_tr_bs4_bm128L0A2": Expected(cycles=59_875, total_memory=56_628_708, max_tile_mem=136_600),
    "resnet50_tr_bs4_bm128L0_projection": Expected(cycles=107_455, total_memory=105_526_402, max_tile_mem=141_720),
    "resnet50_tr_bs4_bm128L1A0": Expected(cycles=61_082, total_memory=56_631_123, max_tile_mem=136_600),
    "resnet50_tr_bs4_bm256L0A0": Expected(cycles=44_957, total_memory=51_493_977, max_tile_mem=136_728),
    "resnet50_tr_bs4_bm256L0A1": Expected(cycles=105_421, total_memory=74_567_720, max_tile_mem=139_352),
    "resnet50_tr_bs4_bm256L0A2": Expected(cycles=61_016, total_memory=52_904_484, max_tile_mem=136_728),
    "resnet50_tr_bs4_bm256L0_projection": Expected(cycles=106_764, total_memory=92_399_694, max_tile_mem=142_040),
    "resnet50_tr_bs4_bm256L1A0": Expected(cycles=62_104, total_memory=52_663_108, max_tile_mem=136_728),
    "resnet50_tr_bs4_bm512L0A0": Expected(cycles=44_494, total_memory=50_581_019, max_tile_mem=133_208),
    "resnet50_tr_bs4_bm512L0A1": Expected(cycles=111_484, total_memory=102_642_415, max_tile_mem=143_640),
    "resnet50_tr_bs4_bm512L0A2": Expected(cycles=63_780, total_memory=78_994_778, max_tile_mem=137_368),
    "resnet50_tr_bs4_bm512L0_projection": Expected(cycles=108_954, total_memory=97_514_761, max_tile_mem=143_640),
    "resnet50_tr_bs4_bm512L1A0": Expected(cycles=64_520, total_memory=56_915_129, max_tile_mem=137_368),
    "resnet50_tr_bs4_bm64L0A1": Expected(cycles=94_640, total_memory=79_964_277, max_tile_mem=141_848),
    "resnet50_tr_bs4_bm64L0": Expected(cycles=28_784, total_memory=35_360_409, max_tile_mem=133_752),
    "resnet50_tr_bs4_bm64L0_projection": Expected(cycles=61_889, total_memory=66_016_584, max_tile_mem=141_720),
    "resnet50_tr_bs4_bm64L1A0": Expected(cycles=63_173, total_memory=65_924_641, max_tile_mem=141_720),
    "resnet50_tr_bs4_cnv": Expected(cycles=173_399, total_memory=113_405_619, max_tile_mem=152_600),
    "vgg16_tr_bs4_v1L0": Expected(cycles=211_969, total_memory=184_896_298, max_tile_mem=218_136),
    "vgg16_tr_bs4_v1L1": Expected(cycles=1_263_568, total_memory=359_639_123, max_tile_mem=328_856),
    "vgg16_tr_bs4_v2L0": Expected(cycles=570_534, total_memory=233_316_955, max_tile_mem=223_690),
    "vgg16_tr_bs4_v2L1": Expected(cycles=1_166_927, total_memory=265_179_657, max_tile_mem=270_722),
    "vgg16_tr_bs4_v3L0": Expected(cycles=532_032, total_memory=213_593_836, max_tile_mem=190_250),
    "vgg16_tr_bs4_v3L1": Expected(cycles=1_046_058, total_memory=233_550_630, max_tile_mem=232_884),
    "vgg16_tr_bs4_v4L0": Expected(cycles=566_048, total_memory=205_909_894, max_tile_mem=203_786),
    "vgg16_tr_bs4_v4L1": Expected(cycles=1_110_697, total_memory=203_532_218, max_tile_mem=206_642),
    "vgg16_tr_bs4_v5L0": Expected(cycles=322_416, total_memory=148_223_671, max_tile_mem=148_120),
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
