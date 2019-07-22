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
    Input(m=200, k=64, n=10000): Expected(cycles=14903, memory=31800344),
    Input(m=200, k=64, n=20000): Expected(cycles=21829, memory=51556544),
    Input(m=200, k=64, n=30000): Expected(cycles=33514, memory=65839448),
    Input(m=200, k=256, n=10000): Expected(cycles=41189, memory=66901532),
    Input(m=200, k=256, n=20000): Expected(cycles=78337, memory=100266604),
    Input(m=200, k=256, n=30000): Expected(cycles=106383, memory=132066512),
    Input(m=200, k=512, n=10000): Expected(cycles=78267, memory=96517004),
    Input(m=200, k=512, n=20000): Expected(cycles=136826, memory=146430240),
    Input(m=200, k=512, n=30000): Expected(cycles=185248, memory=191435724),
    Input(m=600, k=64, n=10000): Expected(cycles=31772, memory=66311912),
    Input(m=600, k=64, n=20000): Expected(cycles=57664, memory=112519164),
    Input(m=600, k=64, n=30000): Expected(cycles=146247, memory=146504460),
    Input(m=600, k=256, n=10000): Expected(cycles=98386, memory=138450898),
    Input(m=600, k=256, n=20000): Expected(cycles=179650, memory=208376148),
    Input(m=600, k=256, n=30000): Expected(cycles=329614, memory=226340792),
    Input(m=600, k=512, n=10000): Expected(cycles=181942, memory=195952624),
    Input(m=600, k=512, n=20000): Expected(cycles=323561, memory=303552956),
    Input(m=600, k=512, n=30000): Expected(cycles=592771, memory=314466080),
    Input(m=1000, k=64, n=10000): Expected(cycles=44746, memory=96856160),
    Input(m=1000, k=64, n=20000): Expected(cycles=165494, memory=160261872),
    Input(m=1000, k=64, n=30000): Expected(cycles=201814, memory=225415184),
    Input(m=1000, k=256, n=10000): Expected(cycles=136400, memory=188522904),
    Input(m=1000, k=256, n=20000): Expected(cycles=354657, memory=250326988),
    Input(m=1000, k=256, n=30000): Expected(cycles=488293, memory=334484940),
    Input(m=1000, k=512, n=10000): Expected(cycles=266408, memory=285218976),
    Input(m=1000, k=512, n=20000): Expected(cycles=623722, memory=333341320),
    Input(m=1000, k=512, n=30000): Expected(cycles=852874, memory=455706604)
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
