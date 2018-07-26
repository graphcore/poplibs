#!/usr/bin/env python

"""
Script to generate random convolutions and run them with single_conv_layer
"""

from __future__ import print_function
import argparse
import random
import subprocess
import sys
from bisect import bisect_left
from operator import mul

if sys.version_info[0] >= 3:
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
max_flops_per_tile = 70000


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
    def get_args(self):
        """Convert the parameters to a list of arguments to pass"""
        def shape_to_str(shape):
            return '{' + ','.join(str(e) for e in shape) + '}'
        cmd = []
        cmd.append('--data-type=' + self.data_type)
        cmd.append('--partials-type=' + self.partials_type)
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

def make_params():
    """Return a random set of convolution parameters"""
    params = Params()
    params.data_type, params.partials_type = random.choice([('float', 'float'),
                                                            ('half', 'float'),
                                                            ('half', 'half')])
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
    params.out_chans_per_group = geometric_choice(
        range(1, max_chans_per_group // out_chans_multiple + 1),
        pow(0.9, out_chans_multiple)
    ) * out_chans_multiple
    field_dims = random.randrange(min_field_dims, max_field_dims + 1)
    params.field = []
    params.padding_lower = []
    params.padding_upper = []
    params.in_dilation = []
    for i in range(field_dims):
        params.field.append(
            geometric_choice(range(1, max_field_size + 1), 0.9)
        )
        params.padding_lower.append(
            geometric_choice(range(1, max_padding + 1), 0.5)
        )
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

    params.stride = []
    for i in range(field_dims):
        params.stride.append(geometric_choice(range(1, max_stride + 1), 0.5))
    return params


def make_constrained_params(tiles_per_ipu):
    """
    Return a random set of convolution parameters subject to constraints

    The convolution parameters returned are constrained to be valid, supported
    by single_conv_layer and not exceed the maximum number of FLOPs.
    """
    while True:
        p = make_params()
        if any(f < k for f, k in zip(p.get_dilated_and_padded_input_size(),
                                     p.get_dilated_and_padded_kernel_size())):
            continue
        # Odd strides have a cost/memory penalty, so derate the flops to
        # compensate. This is only problematic when the number of tiles is low
        flops = p.get_flops();
        nOddDims=len(filter(lambda a: a > 1 and a%2, p.stride))
        if (len(filter(lambda a: a > 1 and a%2, p.stride))):
          print("odd: " + str(p.stride))
        if (flops > max_flops):
            continue
        if (flops * (1 + 0.5*nOddDims) > max_flops_per_tile * tiles_per_ipu):
            continue;
        return p

def select_tiles_per_ipu():
    return weighted_choice([1, 16, 24], [0.3, 0.4, 0.3])

def make_device_args(tiles_per_ipu):
    """Return a random set of device arguments to pass to single_conv_layer"""
    args = []
    args.append('--tiles-per-ipu=' + str(tiles_per_ipu))
    return args


class TestFailureException(Exception):
    """Raised when a test fails"""
    pass


def run(params, binary='single_conv_layer', extra_args=None, dummy_run=False):
    """Run single_conv_layer with the specified convolution parameters"""
    cmd = [binary]
    cmd += params.get_args()
    if (extra_args):
        cmd += extra_args
    cmd_str = ' '.join('"' + e + '"' for e in cmd)
    print('CMD=' + cmd_str)
    print('FLOPS=' + str(params.get_flops()))
    if not dummy_run:
        if subprocess.call(cmd) != 0:
            raise TestFailureException('Failed to run ' + cmd_str + '')


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
    parser.add_argument('--dummy', action='store_true',
                        help='Print parameters without running convolution')
    args = parser.parse_args()

    if sys.version_info[0] < 3:
        random.seed(args.seed)
    else:
        random.seed(args.seed, 1) # Use old PRNG algorithm for compatibility.

    for i in range(args.n):
        tiles_per_ipu = select_tiles_per_ipu()
        device_args = make_device_args(tiles_per_ipu)
        params = make_constrained_params(tiles_per_ipu)
        print('Run #{}:'.format(i))
        try: 
            run(params, binary=args.binary,
                extra_args=device_args + ['--device-type=' +
                   str(args.device_type)],
                dummy_run=args.dummy)
        except TestFailureException as inst:
            print('TestFailure: ' + str(inst.args))
            sys.exit(1);

if __name__ == "__main__":
    sys.exit(main())
