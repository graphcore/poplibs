#!/usr/bin/env python3
#
# A tool to run a suite of matrix multiplications. Each run performs A . B + C
# where A, B and C are matrices of dimensions m * k, k * n and m * n
# respectively. For all runs data type is half (2 bytes) and partials type is
# float (4 bytes).
#
# For each benchmark we record the entire profile report along with the values
# of n, m and k in the json file that is produced.

import argparse
import collections
import datetime
import itertools
import json
import os
import subprocess
import tempfile

Shape = collections.namedtuple('Shape', ['m', 'k', 'n'])

# the matrix dimensions used in the benchmark 
ms = (200, 600, 1000)
ks = (64, 128, 256, 512, 1024, 2048)
ns = (10000, 20000, 30000, 40000, 50000)

def main():
	parser = argparse.ArgumentParser(description='Matrix multiplication benchmark suite')
	parser.add_argument('--binary', default='general_matrix_multiply',
	                    help='general_matrix_multiply binary to run')
	parser.add_argument('--out', help='output json file to store results', required=True)
	args = parser.parse_args()

	my_env = os.environ
	my_env['POPLAR_ENGINE_OPTIONS'] = '{"debug.allowOutOfMemory":"true"}'

	# initialise our output file, truncating it if it already exists.
	with open(args.out, 'w') as out_file:
		json.dump([], out_file)

	test = 1
	num_tests = len(ms) * len(ks) * len(ns)
	for shape in map(Shape._make, itertools.product(ms, ks, ns)):
		with tempfile.NamedTemporaryFile() as result_file:
			cmd = [args.binary, '--ignore-data', '--m', str(shape.m),
				   '--k', str(shape.k), '--n', str(shape.n),
				   '--profile-json', result_file.name]

			subprocess.run(cmd, env=my_env, check=True)

			this_result = json.load(result_file)
			this_result['input'] = shape._asdict()

			# load, parse and update the output file after each test case such
			# that in the case of an unexpected early exit we will at least have
			# all results that have finished.
			with open(args.out, 'r+') as out_file:
				report = json.load(out_file)
				report.append(this_result)

				out_file.seek(0)
				json.dump(report, out_file)
				out_file.truncate()

			print(f'{datetime.datetime.now()} ({test}/{num_tests}): {shape}')
			test += 1

if __name__ == '__main__':
	main()
