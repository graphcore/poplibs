#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
import sympy


def generate_polynomial(items, x):
    sum = 0
    n = len(items)
    sums = []
    for j in range(n):
        mul = 1
        muls = []
        for k in range(n):
            if k == j:
                continue
            muls.append("((x - %d) / %d)" %(items[k][0], (items[j][0] - items[k][0])))
            mul *= (x - items[k][0]) / (items[j][0] - items[k][0])
        sums.append("%d * %s" %(items[j][1], " * ".join(muls)))
        sum += items[j][1] * mul
    poly =  str(" + ".join(sums))
    return sympy.simplify(poly), sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Lagrange interpolation polynom from given dataset.')
    parser.add_argument('-x', type=int, nargs='+', help='Input argument, e.g. codelet size.')
    parser.add_argument('-y', type=int, nargs='+', help='Measured cycles count.')
    parser.add_argument('-e', type=int, default=None, help='Check interpolated value for given argument.')
    parser.add_argument('--all', action='store_true', help='Run triangular solve/inverse/Cholesky and generate polynomials for those.')

    args = parser.parse_args()
    if args.all:
        print("Updating all benchmarks")
        import subprocess
        import itertools
        import tempfile
        import pva

        def collect(codelet, common_args, params, ranks):
            params = list(params.items())
            keys, values = zip(*params)
            for v in itertools.product(*values):
                samples = []
                for rank_opts in ranks:
                    x = rank_opts[0]
                    rank_opts = rank_opts[1:]
                    args = ["./build/poplibs/tools/matrix_solver"] + list(common_args)
                    def format(opt):
                        k, v = opt
                        return f"{k}={v}"
                    args += map(format, zip(keys, v))
                    args += rank_opts
                    with tempfile.TemporaryDirectory() as d:
                        args.append("--profile-dir=" + d.name)
                        print("Running", " ".join(args))
                        subprocess.call(args)
                        data = pva.openReport(d.name + "/profile.pop")
                    steps = data.executionProfile.steps
                    steps = list(filter(lambda step: step.name.endswith("/" + codelet), steps))
                    if len(steps) != 1:
                        raise Exception("Can't find codelet in steps")
                    cycles = steps[0].cycles
                    samples.append((x, cycles))

                x, y = list(zip(*samples))
                x = " ".join(map(str, x))
                y = " ".join(map(str, y))
                print(f"Collected samples: generate_cycle_estimate.py -x {x} -y {y} -e 128")
                poly, _ = generate_polynomial(samples, 0)
                print(f"Polynomial: {poly}")


        collect("triangularSolve", ['--b-rank=1', '--device-type=Hw'], {
            '--data-type': ['half', 'float'],
            '--lower': ['true', 'false']
        }, [(16, '--a-rank=16'), (32, '--a-rank=32'), (64, '--a-rank=64')])

        collect("cholesky", ['--cholesky', '--device-type=Hw'], {
            '--data-type': ['half', 'float'],
            '--lower': ['true', 'false']
        }, [(8, '--a-rank=8'), (16, '--a-rank=16'), (32, '--a-rank=32'), (64, '--a-rank=64')])

        collect("triangularInverse", ['--cholesky', '--device-type=Hw'], {
            '--data-type': ['half', 'float'],
            '--lower': ['true', 'false']
        }, [(8, '--a-rank=16', '--block-size=8'), (16, '--a-rank=32', '--block-size=16'), (32, '--a-rank=64', '--block-size=32'), (64, '--a-rank=128', '--block-size=64')])
    else:
        x, y = args.x, args.y

        if len(x) != len(y):
            raise Exception("Input arrays must have the same length")

        poly, value = generate_polynomial(list(zip(x, y)), 0 if args.e is None else args.e)
        if args.e is not None:
            print(f"Interpolated value: {value}")
        print(f"Polynomial: {poly}")
