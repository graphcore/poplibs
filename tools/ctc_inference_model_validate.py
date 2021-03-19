#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import argparse
import numpy as np
import random
import subprocess
import sys


def print_beam(output_sequence, probability):
    print("{o}".format(o=" ".join([str(x) for x in output_sequence])))
    print("P = {p:.4f}".format(p=probability))
    print("Log(P) = {p:.4f}".format(p=np.log(probability)))


def beam_search_tf(softmax_input, beam_width=4, top_paths=1):
    print("tensorflow:")
    sess = tf.Session()

    # Shape : [max_time, batch_size, num_classes]
    # Blank is the last class
    inputs = tf.constant(softmax_input)

    # Shape : [batch_size]
    sequence_length = tf.constant(
        np.array([softmax_input.shape[0]]), dtype=np.int32)

    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
        inputs, sequence_length, merge_repeated=False, beam_width=beam_width,
        top_paths=top_paths)

    x = sess.run(decoded)
    y = sess.run(log_probabilities)[0]

    for sequence, prob in zip(x, y):
        print_beam(sequence.values, np.exp(prob))

    sess.close()
    return sequence.values, prob


def softmax_to_poplibs_arg(softmax_input):
    return ",".join([format(x, '.8f') for x in softmax_input.swapaxes(0, 2).flatten()])


def vector_to_poplibs_arg(vec):
    return ",".join([str(x) for x in vec])


def beam_search_poplibs(program, softmax_input, expectedSequence, expectedLogProb, beam_width=4, verbose=False):
    print("poplibs:")
    args = ["--verbose", str(verbose),
            "--inference", "on",
            "--beam-width", str(beam_width),
            "--input", "{" + softmax_to_poplibs_arg(softmax_input) + "}",
            "--input-shape", "{" + str(softmax_input.shape[2]) + "," + str(
                softmax_input.shape[0]) + "}",  # (classes, t)
            "--blank-class", str(softmax_input.shape[2] - 1),
            "--expectedSequence", "{" + vector_to_poplibs_arg(expectedSequence) + "}",
            "--expectedLogProb", str(expectedLogProb),
            "--log", "on",
            ]
    # print(program, *args)
    process = subprocess.Popen([program, *args], stdout=subprocess.PIPE)
    process.wait()
    output = [x.decode('utf-8').rstrip('\n')
              for x in iter(process.stdout.readlines())]
    for o in output:
        print(o)
    return process.returncode


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def generate_random_test_input():
    min_classes_inc_blank = 1 + 1  # + 1 for blank
    max_classes_inc_blank = 30
    num_classes_inc_blank = random.randrange(min_classes_inc_blank, max_classes_inc_blank)

    min_t = 3
    max_t = 50
    t = random.randrange(min_t, max_t)

    return softmax(np.random.rand(t, 1, num_classes_inc_blank).astype(np.float32), axis=2)


def main():
    parser = argparse.ArgumentParser(
        description='Generate and run random ctc inference'
    )

    parser.add_argument('--print-input', default=False,
                        help='Whether to print input softmax activations')
    parser.add_argument('--binary', default='ctc_model_validate',
                        help='ctc inference binary to run')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random number seed')
    parser.add_argument('--verbose', default=False,
                        help='poplibs verbose logging')
    args = parser.parse_args()

    print("seed =", args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    softmax_input = generate_random_test_input()
    if (args.print_input):
        print("log(softmax input):")
        print(np.log(softmax_input))
        print("-----")

    beam_width = random.randint(1, 5)

    expectedSequence, expectedLogProb = beam_search_tf(
        np.log(softmax_input), beam_width=beam_width, top_paths=1)

    success = beam_search_poplibs(args.binary, softmax_input, expectedSequence,
                        expectedLogProb, beam_width=beam_width, verbose=args.verbose)
    return success


if __name__ == "__main__":
    sys.exit(main())
