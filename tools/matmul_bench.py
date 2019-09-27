#!/usr/bin/env python3

"""
A tool to run a matrix multiplication and ensure its cycle count and memory
usage is within a given limit. The tool performs A . B + C where A, B and C
are matrices of dimensions m * k, k * n and m * n respectively. For all runs
data type is half (2 bytes) and partials type is float (4 bytes).
"""

import argparse
import json
import os
import subprocess
import tempfile
import collections
import math

# All input and outputs are half-precision, therefore 2 bytes big.
DATA_SIZE = 2
# The maximum allowed relative difference in memory/cycles before an exception
RELATIVE_TOLERANCE = 0.01

Input = collections.namedtuple('Input', ['m', 'n', 'k'])
Expected = collections.namedtuple('Expected', ['cycles', 'memory'])

# WARNING: All benchmark tests must be added to CMakeLists.txt (not just here!)
EXPECTED_RESULTS = {
    Input(m=200, k=64, n=10000): Expected(cycles=13691, memory=38330942),
    Input(m=200, k=64, n=20000): Expected(cycles=21829, memory=51556544),
    Input(m=200, k=64, n=30000): Expected(cycles=28975, memory=71273596),
    Input(m=200, k=256, n=10000): Expected(cycles=41189, memory=66901532),
    Input(m=200, k=256, n=20000): Expected(cycles=66587, memory=109145388),
    Input(m=200, k=256, n=30000): Expected(cycles=86701, memory=148150216),
    Input(m=200, k=512, n=10000): Expected(cycles=68524, memory=111195560),
    Input(m=200, k=512, n=20000): Expected(cycles=115601, memory=179217292),
    Input(m=200, k=512, n=30000): Expected(cycles=177742, memory=204690000),
    Input(m=600, k=64, n=10000): Expected(cycles=26888, memory=94532596),
    Input(m=600, k=64, n=20000): Expected(cycles=49618, memory=180453580),
    Input(m=600, k=64, n=30000): Expected(cycles=72446, memory=182164862),
    Input(m=600, k=256, n=10000): Expected(cycles=90222, memory=150082846),
    Input(m=600, k=256, n=20000): Expected(cycles=164044, memory=218482024),
    Input(m=600, k=256, n=30000): Expected(cycles=269674, memory=209474640),
    Input(m=600, k=512, n=10000): Expected(cycles=177467, memory=206439424),
    Input(m=600, k=512, n=20000): Expected(cycles=351316, memory=220796356),
    Input(m=600, k=512, n=30000): Expected(cycles=554659, memory=229979892),
    Input(m=1000, k=64, n=10000): Expected(cycles=40728, memory=113552036),
    Input(m=1000, k=64, n=20000): Expected(cycles=74181, memory=183823234),
    Input(m=1000, k=64, n=30000): Expected(cycles=107821, memory=255279388),
    Input(m=1000, k=256, n=10000): Expected(cycles=128192, memory=198957366),
    Input(m=1000, k=256, n=20000): Expected(cycles=385859, memory=221097708),
    Input(m=1000, k=256, n=30000): Expected(cycles=532496, memory=317418116),
    Input(m=1000, k=512, n=10000): Expected(cycles=342806, memory=191871778),
    Input(m=1000, k=512, n=20000): Expected(cycles=715261, memory=272381384),
    Input(m=1000, k=512, n=30000): Expected(cycles=997487, memory=302224850)
}

class TestFailureException(Exception):
    """Raised when a test fails"""
    def __init__(self, message, returncode):
        super(TestFailureException, self).__init__(message)
        self.returncode = returncode

def get_always_live(liveness, args):
    """Returns memory usage of always-live variables in bytes."""
    per_tile = liveness['alwaysLive']['bytesByTile']
    # the output tensor is not marked as always live so add that in ourselves.
    return sum(per_tile) + (DATA_SIZE * args.m * args.n)

def get_max_temp(liveness, args):
    """Returns sum of maximum memory usage per tile of temporary variables."""
    per_tile = liveness['notAlwaysLive']['maxBytesByTile']
    # the output tensor is marked as temporary memory so remove that from
    # the total.
    return sum(per_tile) - (DATA_SIZE * args.m * args.n)

def main():
    parser = argparse.ArgumentParser(
        description='Matrix multiplication benchmark suite')
    parser.add_argument('--binary', default='general_matrix_multiply',
                        help='General_matrix_multiply binary to run')
    parser.add_argument('--m', type=int, required=True,
                        help='Size of matrix dimension m')
    parser.add_argument('--k', type=int, required=True,
                        help='Size of matrix dimension k')
    parser.add_argument('--n', type=int, required=True,
                        help='Size of matrix dimension n')
    parser.add_argument('--device-type', default='IpuModel',
                        help='Underlying target to use')
    args = parser.parse_args()
    my_env = os.environ
    my_env['POPLAR_ENGINE_OPTIONS'] = '{"debug.allowOutOfMemory":"true"}'

    with tempfile.NamedTemporaryFile() as out:
        cmd = [args.binary, '--ignore-data', '--m', str(args.m), '--k',
               str(args.k), '--n', str(args.n), '--profile-json', out.name,
               '--device-type', args.device_type]
        print("Command: ", *cmd)
        subprocess.run(cmd, env=my_env, check=True)
        result = json.load(out)
        cycles = result['executionProfile']['simulation']['cycles']
        liveness = result['graphProfile']['memory']['liveness']
        memory = get_always_live(liveness, args) + get_max_temp(liveness, args)

        # Asserts if no expected value found
        expected = EXPECTED_RESULTS[Input(m=args.m, n=args.n, k=args.k)]
        memory_changed = not math.isclose(
            expected.memory, memory, rel_tol=RELATIVE_TOLERANCE)
        cycles_changed = not math.isclose(
            expected.cycles, cycles, rel_tol=RELATIVE_TOLERANCE)
        tolerance_message = ' differs by over ' + \
            str(RELATIVE_TOLERANCE * 100) + '% from the expected value '

        message = ''
        if memory_changed:
            message += '\nERROR: Memory usage (' + str(memory) + ')' + \
                        tolerance_message + '(' + str(expected.memory) + ')'
        if cycles_changed:
            message += '\nERROR: Cycle count (' + str(cycles) + ')' + \
                        tolerance_message + '(' + str(expected.cycles) + ')'

        if memory_changed or cycles_changed:
            raise TestFailureException(message, 1)

if __name__ == '__main__':
    main()
