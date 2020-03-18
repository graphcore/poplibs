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
    "Expected", ["cycles", "total_memory", "max_tile_memory"]
)

CHANGED_RESULT_PREFIX = "CHANGED_BENCHMARK_RESULT"
CHANGED_RESULT_PATTERN = re.compile(
    '^' + CHANGED_RESULT_PREFIX + ': '
    'target=(\w+),config=(\w+),name=(\w+),'
    'cycles=(\d+),total_memory=(\d+),max_tile_memory=(\d+)$')

NONE = Expected(
    cycles = sys.maxsize,
    total_memory = sys.maxsize,
    max_tile_memory = sys.maxsize
)

def read_ignoring_comments(f):
    for row in f:
        if not row.startswith("#"):
            yield row

def read_results_file(path):
    with open(path) as results_file:
        results_reader = csv.reader(read_ignoring_comments(results_file), delimiter=',')
        expected_dict = {
          TestKey._make(row[0:3]): Expected._make(int(x) for x in row[3:])
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

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark regression tool, compares memory and cycles "
        "against the expected values."
    )
    parser.add_argument(
        "--name", help="Test name used to look-up expected results"
    )
    parser.add_argument(
        "--target", help="Test target used to look-up expected results"
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
    expected = expected_dict.get(TestKey(
        target = args.target,
        config = args.config,
        name = args.name
    ), NONE)

    def check_value(name, actual_value, expected_value):

        # =================================================================
        # =================================================================
        # Workaround to allow D19573 (poplar diff) to be landed. Will be
        # removed by D22662.
        # Temporary adjustment to allow passing the test even if memory usage
        # is slightly different.
        #
        # This accounts for:
        #   1. When D19573 (compute stack sizes) is landed, most poplibs tests
        #      will have a decrease in memory usage (tile and total) as the
        #      stack space will now be smaller.
        #   2. But tests that set 'target.workerStackSizeInBytes' will see an
        #      *increase* of memory, up to 6x48 bytes (tile mem) / 6x48x1216
        #      (total mem) because that option will now be setting the stack
        #      size alone, not the stack+scratch space
        #
        # D22662 will remove all the 'target.workerStackSizeInBytes' option
        # setting, and set tests/benchmark_results.csv to the updated values
        # (lower memory all around).
        # It will also remove this code.
        if name == "Max tile memory" or name == "Total memory":
            if actual_value < expected_value:
                # Less memory used, as expected, benchmark is not using
                # 'target.workerStackSizeInBytes'
                print("Improved memory for {}  ({}=>{}): ignoring",
                      name, expected_value, actual_value)
                actual_value = expected_value
            elif actual_value > expected_value:
                # More memory used, this must be one of the tests that sets
                # 'target.workerStackSizeInBytes', let's check if the increase
                # is within the expected limit
                if name == "Max tile memory":
                    if actual_value <= (expected_value + 48*6):
                        print("Worse max tile memory for {}  ({}=>{}): ignoring",
                              name, expected_value, actual_value)
                        actual_value = expected_value
                elif name == "Total memory":
                    if actual_value <= (expected_value + 48*6*1216):
                        print("Worse total memory for {}  ({}=>{}): ignoring",
                              name, expected_value, actual_value)
                        actual_value = expected_value
        # =================================================================
        # =================================================================


        if expected_value != actual_value:
            pc_diff = actual_value / expected_value * 100 - 100
            print(
                f"ERROR: {name} usage ({actual_value:,}) differs by "
                f"{pc_diff:.1f}% from the expected value ({expected_value:,})"
            )
            return False
        return True

    passed = True
    passed &= check_value("Total memory", memory, expected.total_memory)
    passed &= check_value(
        "Max tile memory", max_tile_mem, expected.max_tile_memory
    )
    passed &= check_value("Cycles", cycles, expected.cycles)
    if not passed:
        out_line = (
            CHANGED_RESULT_PREFIX +
            f': target={args.target},'
            f'config={args.config},name={args.name},'
            f'cycles={cycles},total_memory={memory},'
            f'max_tile_memory={max_tile_mem}'
        )
        # update_bench.py relies on this format of output
        assert CHANGED_RESULT_PATTERN.match(out_line)
        print(out_line)
        raise TestFailureException()


if __name__ == "__main__":
    main()
