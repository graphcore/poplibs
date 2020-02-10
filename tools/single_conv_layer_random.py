#!/usr/bin/env python
# Copyright (c) Graphcore Ltd, All rights reserved.

"""
Script to generate random convolutions and run them with single_conv_layer
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from bisect import bisect_left
from operator import mul
from functools import reduce

vector_width = {
    'half': 4,
    'float': 2
}
amp_in_chans = {
    'half': 16,
    'float': 4
}
amp_out_chans = {
    'half': 8,
    'float': 8
}

max_conv_groups = 8
max_batch_size = 16
min_field_dims = 2
max_field_dims = 3
max_field_size = 256
max_in_dilation = 4
max_padding = 5

max_kernel_size = 16
max_kernel_dilation = 4
max_kernel_padding = 3

max_chans_per_group = 128
max_stride = 4
max_flops = 500000
max_flops_per_tile = 50000
max_start_tile_multiplier = 100

def geometric_sequence(a, r):
    """
    Generator for a geometric series

    Args:
        a: First term in the series
        r: Common ratio between successive terms
    """
    x = a
    while True:
        yield x
        x *= r


def weighted_choice(seq, weights):
    """
    Return a random element from a sequence based on a weighting.

    The probability of selecting an element is proportional the weight of the
    element.

    Args:
        seq: The sequence to pick from
        weights: The weight of each element in the sequence
    """
    cumulative_sum = []
    sum = 0
    # Use zip to ensure we don't read more weights than elements in the
    # sequence.
    for _, w in zip(seq, weights):
        sum += w
        cumulative_sum.append(sum)
    x = random.uniform(0, sum)
    return seq[bisect_left(cumulative_sum, x)]


def geometric_choice(seq, r):
    """
    Return a random element from a sequence based on a geometric weighting.

    The probability of selecting an element is determined by a truncated
    geometric series

    Args:
        seq: The sequence to pick from
        r: The common ratio of the geometric series
    """
    return weighted_choice(seq, geometric_sequence(1, r))


def dilate_and_pad(shape, dilation, padding_lower, padding_upper):
    """Return the shape of a volume with dilation and padding applied"""
    result = []
    for s, d, l, u in zip(shape, dilation, padding_lower, padding_upper):
        result.append((1 + (s - 1) * d + l + u) if s > 0 else 0)
    return result


class Params:
    def __init__(self):
        self.conv_options = {}

    def get_args(self):
        """Convert the parameters to a list of arguments to pass"""
        def shape_to_str(shape):
            return '{' + ','.join(str(e) for e in shape) + '}'
        cmd = []
        cmd.append('--data-type=' + self.data_type)
        cmd.append('--conv-groups=' + str(self.conv_groups))
        cmd.append('--batch-size=' + str(self.batch_size))
        cmd.append('--field=' + shape_to_str(self.field))
        cmd.append('--input-channels=' + str(self.in_chans_per_group))
        cmd.append('--in-dilation=' + shape_to_str(self.in_dilation))
        cmd.append('--padding-upper=' + shape_to_str(self.padding_upper))
        cmd.append('--padding-lower=' + shape_to_str(self.padding_lower))
        cmd.append('--output-channels=' + str(self.out_chans_per_group))
        cmd.append('--kernel-size=' + shape_to_str(self.kernel_size))
        cmd.append('--kernel-dilation=' + shape_to_str(self.kernel_dilation))
        cmd.append('--kernel-padding-upper=' +
                   shape_to_str(self.kernel_padding_upper))
        cmd.append('--kernel-padding-lower=' +
                   shape_to_str(self.kernel_padding_lower))
        cmd.append('--stride=' + shape_to_str(self.stride))
        cmd.append('--convolution-options=' + json.dumps(self.conv_options))
        return cmd

    def get_dilated_and_padded_input_size(self):
        return dilate_and_pad(self.field,
                              self.in_dilation,
                              self.padding_lower,
                              self.padding_upper)

    def get_dilated_and_padded_kernel_size(self):
        return dilate_and_pad(self.kernel_size,
                              self.kernel_dilation,
                              self.kernel_padding_lower,
                              self.kernel_padding_upper)

    def get_output_size(self):
        """Return the shape of the output field"""
        transformed_field = self.get_dilated_and_padded_input_size()
        transformed_kernel = self.get_dilated_and_padded_kernel_size()
        output_size = []
        for f, k, s in zip(transformed_field, transformed_kernel, self.stride):
            out = abs(f - k) + 1
            out = 1 + (out - 1) // s
            output_size.append(out)
        return output_size

    def get_flops(self):
        """Return the number of FLOPs required to compute the convolution"""
        output_size = self.get_output_size()
        output_elements = reduce(mul, output_size, 1)
        output_elements *= self.conv_groups * self.batch_size
        kernel_elements = reduce(mul, self.kernel_size, 1)
        in_chans_per_group = self.in_chans_per_group
        out_chans_per_group = self.out_chans_per_group
        return (output_elements * kernel_elements * in_chans_per_group *
                out_chans_per_group)

def make_params(num_ipus):
    """Return a random set of convolution parameters"""
    params = Params()

    # on creating random, symmetrical convolutions:
    # the formula for calculating the output size (o) of a convolution given
    # it's input size (i), kernel size (k), stride(s) and padding (p) is:
    #   o = (i + k + p) / s + 1
    # a symmetrical convolution will have the same input size as output size:
    #   x = (x + k + p) / s + 1
    # to randomly generate these we generate several parameters and then apply
    # this formula for the last one, using an bounded range that keeps any
    # invariants that remaining variable may have. we will do this for k, so:
    #   k = x - xs + p + s, where k > 0
    # for k to be positive we must keep the inequality:
    #   x - xs + p + s > 0
    #   p > xs - x - s
    # therefore given any values for x and s (which are both > 0), we can create
    # a padding that allows us to have a positive kernel size.
    symmetrical = random.randrange(0, 100) > 70

    types = [('float', 'float'), ('half', 'half')]
    if not symmetrical:
        types.append(('half', 'float'))
    params.data_type, params.conv_options['partialsType'] = random.choice(types)

    # We want to avoid generating layers that take a long time to compile/run
    # and therefore we would like to avoid many large parameters. Weight the
    # choice of parameters that affect the number of FLOPs by the geometric
    # distribution. This makes small values more likely while still allowing
    # large possible range of values.
    params.batch_size = geometric_choice(range(1, max_batch_size + 1), 0.4)
    params.conv_groups = geometric_choice(range(1, max_conv_groups + 1), 0.4)
    in_chans_multiple = weighted_choice(
        [1, vector_width[params.data_type], amp_in_chans[params.data_type]],
        [0.25, 0.25, 0.5]
    ) * params.conv_groups
    out_chans_multiple = weighted_choice(
        [1, vector_width[params.data_type], amp_in_chans[params.data_type]],
        [0.25, 0.25, 0.5]
    ) * params.conv_groups
    params.in_chans_per_group = geometric_choice(
        range(1, max_chans_per_group // in_chans_multiple + 1),
        pow(0.9, in_chans_multiple)
    ) * in_chans_multiple
    if symmetrical:
        params.out_chans_per_group = params.in_chans_per_group
    else:
        params.out_chans_per_group = geometric_choice(
            range(1, max_chans_per_group // out_chans_multiple + 1),
            pow(0.9, out_chans_multiple)
        ) * out_chans_multiple

    field_dims = random.randrange(min_field_dims, max_field_dims + 1)
    params.field = []
    params.stride = []
    for i in range(field_dims):
        params.field.append(
            geometric_choice(range(1, max_field_size + 1), 0.9)
        )
        params.stride.append(geometric_choice(range(1, max_stride + 1), 0.5))

    params.padding_lower = []
    params.padding_upper = []
    params.in_dilation = []
    for i in range(field_dims):
        # for symmetrical convolutions we will need to calculate the minimum
        # padding required:
        #   p > xs - x - s
        p = 1
        if symmetrical:
            # for symmetrical convolutions input dilation = stride.
            # input size = field size + (dilation - 1) * (field size - 1)
            s = params.stride[i]
            x = params.field[i] + (s - 1) * (params.field[i] - 1)
            p = x * s - x - s + 1

        params.padding_lower.append(
            geometric_choice(range(p, max(p, max_padding) + 1), 0.5)
        )

        if symmetrical:
            params.padding_upper.append(params.padding_lower[i])
            params.in_dilation.append(params.stride[i])
        else:
            params.padding_upper.append(
                geometric_choice(range(1, max_padding + 1), 0.5)
            )
            params.in_dilation.append(
                geometric_choice(range(1, max_in_dilation + 1), 0.5)
            )

    params.kernel_size = []
    params.kernel_padding_lower = []
    params.kernel_padding_upper = []
    params.kernel_dilation = []
    for i in range(field_dims):
        if symmetrical:
            # calculate how big the kernel must be in this dimension:
            #   k = x - xs + p + s
            s = params.stride[i]
            x = params.field[i] + (s - 1) * (params.field[i] - 1)
            p = params.padding_upper[i] + params.padding_lower[i]
            k = x - x * s + p + s
            assert k > 0

            params.kernel_size.append(k)

            # TODO: T12991 Add kernel padding and dilation to symmetrical
            # convolutions.
            params.kernel_padding_lower.append(0)
            params.kernel_padding_upper.append(0)
            params.kernel_dilation.append(1)
        else:
            params.kernel_size.append(
                geometric_choice(range(1, max_kernel_size + 1), 0.70)
            )
            params.kernel_padding_lower.append(
                geometric_choice(range(0, max_kernel_padding + 1), 0.3)
            )
            params.kernel_padding_upper.append(
                geometric_choice(range(0, max_kernel_padding + 1), 0.3)
            )
            params.kernel_dilation.append(
                geometric_choice(range(1, max_kernel_dilation + 1), 0.5)
            )

    params.conv_options['startTileMultiplier'] = random.randint(0, max_start_tile_multiplier/2) * 2
    params.conv_options['use128BitConvUnitLoad'] = random.choice([True, False])
    return params

def make_constrained_params(tiles_per_ipu, num_ipus):
    """
    Return a random set of convolution parameters subject to constraints

    The convolution parameters returned are constrained to be valid, supported
    by single_conv_layer and not exceed the maximum number of FLOPs.
    """
    while True:
        p = make_params(num_ipus)
        if any(f < k for f, k in zip(p.get_dilated_and_padded_input_size(),
                                     p.get_dilated_and_padded_kernel_size())):
            continue
        # Odd strides have a cost/memory penalty, so de-rate the flops to
        # compensate. This is only problematic when the number of tiles is low
        flops = p.get_flops();
        nOddDims=len([a for a in p.stride if a > 1 and a%2])
        if nOddDims:
          print("odd: " + str(p.stride))
        if (flops > max_flops):
            continue
        if (flops * (1 + 0.5*nOddDims) > max_flops_per_tile * tiles_per_ipu):
            continue;
        return p

def select_tiles_per_ipu():
    return weighted_choice([1, 16, 24], [0.3, 0.4, 0.3])

def select_io_tiles_per_ipu(can_be_zero):
    zero_weight = 0.3 if can_be_zero else 0
    return weighted_choice([0, 1, 2, 4], [zero_weight, 0.3, 0.4, 0.3])

def make_device_args(tiles_per_ipu, num_io_tiles, enable_shared_structures):
    """Return a random set of device arguments to pass to single_conv_layer"""
    args = []
    args.append('--tiles-per-ipu=' + str(tiles_per_ipu + num_io_tiles))
    args.append('--num-io-tiles=' + str(num_io_tiles))
    if enable_shared_structures:
        args.append('--enable-shared-structures')
    return args


class TestFailureException(Exception):
    """Raised when a test fails"""
    def __init__(self, message, returncode):
        super(TestFailureException, self).__init__(message)
        self.returncode = returncode


def run(params, binary='single_conv_layer', extra_args=None, dummy_run=False):
    """Run single_conv_layer with the specified convolution parameters"""
    cmd = [binary]
    cmd += params.get_args()
    if (extra_args):
        cmd += extra_args
    cmd_str = ' '.join("'" + e + "'" for e in cmd)
    print('CMD=' + cmd_str)
    print('FLOPS=' + str(params.get_flops()))

    my_env = os.environ
    if platform.system() == 'Darwin':
        # TODO: T12992 This should be set when LIBRARY_PATH is also set, not
        # here.
        my_env['DYLD_LIBRARY_PATH'] = my_env['LIBRARY_PATH']

    if not dummy_run:
        process = subprocess.Popen(cmd, env=my_env, stdout=sys.stdout,
                                   stderr=sys.stderr)
        process.communicate()
        if process.returncode != 0:
            message = 'Failed to run ' + cmd_str + ''
            raise TestFailureException(message, process.returncode)

def main():
    parser = argparse.ArgumentParser(
        description='Generate and run random convolutions'
    )
    parser.add_argument('--binary', default='single_conv_layer',
                        help='single_conv_layer binary to run')
    parser.add_argument('--n', type=int, required=True,
                        help='Number of convolutions to run')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random number seed')
    parser.add_argument('--device-type', default='IpuModel',
                        help='Underlying target to use')
    parser.add_argument('--profile', action='store_true',
                        help='Print profiling report once complete')
    parser.add_argument('--dummy', action='store_true',
                        help='Print parameters without running convolution')
    parser.add_argument('--ipus', default=1, type=int,
                        help='Number of ipus to use')
    parser.add_argument('--tiles-per-ipu', type=int,
                        help='Number of tiles per ipu to use')
    args = parser.parse_args()

    random.seed(args.seed)

    for i in range(args.n):
        enable_shared_structures = random.choice([True, False])

        num_io_tiles = select_io_tiles_per_ipu(not enable_shared_structures)
        if args.tiles_per_ipu is not None:
            # user has explicitly asked for this number of tiles, make sure
            # that is upheld (ie. the io tiles are taken out of that set rather
            # than appended to it).
            assert args.tiles_per_ipu > num_io_tiles
            tiles_per_ipu = args.tiles_per_ipu - num_io_tiles
        else:
            tiles_per_ipu = select_tiles_per_ipu()

        device_args = make_device_args(tiles_per_ipu,
                                       num_io_tiles,
                                       enable_shared_structures)
        device_args.append('--ipus=' + str(args.ipus))
        params = make_constrained_params(tiles_per_ipu, args.ipus)
        print('Run #{}:'.format(i))
        try:
            extra_args=device_args + ['--device-type=' +
                str(args.device_type)];
            if args.profile:
                extra_args.append('--profile');
            run(params, binary=args.binary,
                extra_args=extra_args,
                dummy_run=args.dummy)
        except TestFailureException as inst:
            print('TestFailure: ' + str(inst.args))
            sys.exit(inst.returncode);

if __name__ == "__main__":
    sys.exit(main())
