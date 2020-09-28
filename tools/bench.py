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
import csv
import re

TestKey = collections.namedtuple(
    "TestKey", ["target", "config", "name"]
)
Expected = collections.namedtuple(
    "Expected", ["cycles", "cycles_change", 
                 "total_memory", "total_memory_change", 
                 "max_tile_memory", "max_tile_memory_change"]
)

CHANGED_RESULT_PREFIX = "CHANGED_BENCHMARK_RESULT"
CHANGED_RESULT_PATTERN = re.compile(
    '^' + CHANGED_RESULT_PREFIX + ': '
    'target=(\w+),config=(\w+),name=(\w+),'
    'cycles=(\d+)\((.+)%\),'
    'total_memory=(\d+)\((.+)%\),'
    'max_tile_memory=(\d+)\((.+)%\)$')

NONE = Expected(
    cycles = sys.maxsize,
    cycles_change = sys.maxsize,
    total_memory = sys.maxsize,
    total_memory_change = sys.maxsize,
    max_tile_memory = sys.maxsize,
    max_tile_memory_change = sys.maxsize
)

def read_ignoring_comments(f):
    for row in f:
        if not row.startswith("#"):
            yield row

def read_results_file(path):
    with open(path) as results_file:
        results_reader = csv.reader(read_ignoring_comments(results_file), delimiter=',')
        expected_dict = {
          TestKey._make(row[0:3]): Expected._make(float(x) for x in row[3:])
          for row in results_reader if row
        }
        return expected_dict

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


def parse_test_command_args(args):
    """Function to populate args with additional values extracted from the test command (args.test).
    For example, add_multitarget_test in tests/CMakeLists.txt appends --device-type to the test
    command, which is needed by this script. Such values cannot be extracted by the main
    ArgumentParser as they get captured by args.test."""
    test_parser = argparse.ArgumentParser(
        description="Test command argument parser."
    )
    test_parser.add_argument(
        "--device-type",
        help="Test device type used to look-up expected results."
    )
    test_args, _ = test_parser.parse_known_args(args.test)
    args.device_type = test_args.device_type


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark regression tool, compares memory and cycles "
        "against the expected values."
    )
    parser.add_argument(
        "--name", help="Test name used to look-up expected results"
    )
    parser.add_argument(
        "--config", help="Test config used to look-up expected results"
    )
    parser.add_argument(
        "--expected_csv",
        help="Path to a file containing csv with expected results for benchmarks"
    )
    parser.add_argument(
        "test", nargs=argparse.REMAINDER, help="Which test to run"
    )
    args = parser.parse_args()
    parse_test_command_args(args)

    with tempfile.NamedTemporaryFile() as out:
        cmd = args.test + ["--profile-json", out.name]
        print("Command: ", *("'" + s + "'" for s in cmd))
        subprocess.run(cmd, check=True)
        result = json.load(out)
        liveness = result["graphProfile"]["memory"]["liveness"]

        cycles = result["executionProfile"]["simulation"]["cycles"]
        memory = get_always_live(liveness, args) + get_max_temp(liveness, args)
        max_tile_mem = max(
            result["graphProfile"]["memory"]["byTile"]["totalIncludingGaps"]
        )

    expected_dict = read_results_file(args.expected_csv)
    key = TestKey(
        target = args.device_type,
        config = args.config,
        name = args.name
    )
    if key in expected_dict:
        expected = expected_dict[key]

        def check_value(name, actual_value, expected_value):
            if expected_value != actual_value:
                pc_diff = actual_value / expected_value * 100 - 100
                print(
                    f"ERROR: {name} usage ({actual_value:,}) differs by "
                    f"{pc_diff:.1f}% from the expected value ({expected_value:,})"
                )
                return [False, pc_diff]
            return [True, float(0)]

        [mem_passed, mem_diff] = check_value(
            "Total memory", memory, expected.total_memory)
        [tile_mem_passed, tile_mem_diff] = check_value(
            "Max tile memory", max_tile_mem, expected.max_tile_memory)
        [cycles_passed, cycles_diff] = check_value("Cycles", cycles, expected.cycles)
    else:
        [mem_passed, mem_diff] = [False, 0]
        [tile_mem_passed, tile_mem_diff] = [False, 0]
        [cycles_passed, cycles_diff] = [False, 0]

    if not (mem_passed and tile_mem_passed and cycles_passed):
        out_line = (
            CHANGED_RESULT_PREFIX +
            f': target={args.device_type},'
            f'config={args.config},name={args.name},'
            f'cycles={cycles}({cycles_diff:.1f}%),'
            f'total_memory={memory}({mem_diff:.1f}%),'
            f'max_tile_memory={max_tile_mem}({tile_mem_diff:.1f}%)'
        )
        # update_bench.py relies on this format of output
        assert CHANGED_RESULT_PATTERN.match(out_line)
        print(out_line)
        raise TestFailureException()


if __name__ == "__main__":
    main()
