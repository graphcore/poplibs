#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

"""
A tool to plot graph and execution statistics for a Poplar program as a
single parameter is methodically incremented.
"""

import argparse
import json
import os
import pickle
import re
import tempfile
from collections import defaultdict
from math import ceil
from subprocess import DEVNULL, call
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg') # Do not load GTK (prevents warning message)
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
VAR_PATTERN_RANGE = re.compile(r'\{(\d+):(\d+):(\d+)\}') # E.g. {10:100:5}
VAR_PATTERN_LIST = re.compile(r'\{((\d+,?)+)\}') # E.g. {1,2,3,4}
MB_SCALE = 1024*1024


def sum_sum(lists):
    """Returns total sum for list of lists."""
    return sum(sum(x) for x in lists)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''A tool to plot graph and execution statistics for a Poplar \
program as a single parameter is methodically incremented.

NOTE: Your test executable must support the '--profile-json' option flag.''',
        epilog=r'''EXAMPLES
- Show how Resnet metrics scale with batch-size, saving Pickle and Png to default locations:
    {0} resnet --variant RESNET_32 --batch-size \{{10:51:10}}

- Same again but using variable list instead of range - save to custom Pickle file:
    {0} --pickle my.pickle resnet --variant RESNET_32 --batch-size \{{10,20,30,40,50}}

- Load plot data from my.pickle and save plot to my.png:
    {0} --pickle my.pickle --output my.png

- Merge several Pickle files and save png to default location:
    {0} --pickle-merge part1.pickle part2.pickle part3.pickle --pickle merge.pickle
        '''.format(SCRIPT_NAME + ".py")
    )
    parser.add_argument(
        "--output", default=SCRIPT_NAME + ".png",
        help="[=%(default)s] Name of file to output plot figure (in PNG format)."
    )
    parser.add_argument(
        "--pickle", default=SCRIPT_NAME + ".pickle",
        help="[=%(default)s] Name of Pickle file used for plot data save/restore. "
        "Pickle file is used for output if 'command' is provided - input otherwise."
    )
    parser.add_argument(
        "--pickle-off", action='store_true',
        help="[=%(default)s] Disable save/restore of plot data in Pickle file."
    )
    parser.add_argument(
        "--pickle-merge", nargs='*', help="Got results spread across several Pickle files? "
        "List them all here and they'll be plotted on the same graph and Pickled to --pickle. "
        "Cannot be used if `command` is provided."
    )
    parser.add_argument(
        "--title", help="Plot figure title. If not set, test command will be used."
    )
    parser.add_argument(
        "--max-parallel", type=int, default=8, help="Maximum number of parallel processes "
        "to be running at any time."
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER,
        help="Must be last argument. The program command to call and "
        "analyse. The command must include a parameter to be varied which can "
        "be represented in two ways. Either the pattern '{A:B:C}', for which the "
        "command will be executed once for each value D = A, A+C, A+C+C, ... for "
        "all D < B. Or '{E,F,G...,H}', in which the command will be executed once "
        "for each value in the list. All values must be integers. Remember to escape "
        r"the opening brace - e.g. \{ - to avoid shell expansion."
    )
    return parser.parse_args()


def get_var_name_params(cmd):
    all_var_matches_range = list(filter(VAR_PATTERN_RANGE.search, cmd))
    all_var_matches_list = list(filter(VAR_PATTERN_LIST.search, cmd))
    if len(all_var_matches_range) is 1:
        var_match = VAR_PATTERN_RANGE.search(all_var_matches_range[0])
        params = np.arange(int(var_match.group(1)),
                           int(var_match.group(2)),
                           int(var_match.group(3)))

        name_pattern = re.compile(r'--?([a-zA-Z\-]+)[=|\s]' + VAR_PATTERN_RANGE.pattern)
        var_name = name_pattern.search(" ".join(cmd)).group(1)
        return (var_name, params, VAR_PATTERN_RANGE)
    elif len(all_var_matches_list) is 1:
        var_match = VAR_PATTERN_LIST.search(all_var_matches_list[0]).group(1)
        params = list(map(int, var_match.split(",")))
        name_pattern = re.compile(r'--?([a-zA-Z\-]+)[=|\s]' + VAR_PATTERN_LIST.pattern)
        var_name = name_pattern.search(" ".join(cmd)).group(1)
        return (var_name, params, VAR_PATTERN_LIST)
    else:
        print(r" - ERROR: Exactly one occurrence of {A:B:C} or {D,E,F,..G} must appear in command.")
        exit()

def save_pickle(args, var_name, x_values, data):
    if not args.pickle_off:
        with open(args.pickle, 'wb') as pickle_file:
            pickle.dump(args.command, pickle_file)
            pickle.dump(var_name, pickle_file)
            pickle.dump(x_values, pickle_file)
            pickle.dump(data, pickle_file)
            print(" - Plot data pickled to {}.".format(args.pickle))

def main():
    args = get_args()
    data = defaultdict(list)
    x_values = []
    assert not (args.command and args.pickle_merge),\
        "Error: Cannot provide command when --pickle-merge is used."
    if args.command: # Command provided. Generate data before plotting.
        (var_name, params, pattern) = get_var_name_params(args.command)
        generate_data(args, params, pattern, x_values, data)
    else: # No command provided. Load data from Pickle file.
        cmd_params = []
        cmd_arr = []
        prev_cmd_str = ''
        for pick in args.pickle_merge if args.pickle_merge else [args.pickle]:
            if not os.path.exists(pick):
                print(" - ERROR: {} does not exist. Exiting.".format(pick))
                exit()

            print(" - Loading plot data from {}.".format(pick))
            with open(pick, "rb") as pickle_file:
                cmd_arr = pickle.load(pickle_file)
                var_name = pickle.load(pickle_file)
                x_values += pickle.load(pickle_file)
                new_data = pickle.load(pickle_file)
                for key in new_data:
                    data[key] += new_data[key]

                cmd_str = " ".join(cmd_arr)
                start = cmd_str.find("{") + 1
                end = cmd_str.find("}")

                assert not prev_cmd_str or prev_cmd_str == cmd_str[:start] + cmd_str[end:],\
                    "Error: Pickle files do not use the same command."

                cmd_params.append(cmd_str[start:end])
                prev_cmd_str = cmd_str[:start] + cmd_str[end:]

        param_combo = "{" + '|'.join(cmd_params) + "}"
        args.command = [re.sub(r"(\{).+(\})", param_combo, c) for c in cmd_arr]

    if not x_values:
        print(" - ERROR: No data found.")
    else:
        save_pickle(args, var_name, x_values, data)
        plot_data(var_name, x_values, data, args)


def generate_data(args, params, pattern, x_values, data):
    cmd = args.command

    assert (not os.path.dirname(args.pickle) or os.access(os.path.dirname(args.pickle), os.W_OK)),\
           "- ERROR: Cannot create {}.".format(args.pickle)

    assert (os.path.isfile(cmd[0]) and os.access(cmd[0], os.X_OK)),\
           "- ERROR: File '{}' does not exist or is not executable.".format(cmd[0])

    with tempfile.TemporaryDirectory() as out_dir:
        # Create list of all cmd variants (variable value and JSON output file)
        out_files = ['--profile-json=' + os.path.join(out_dir, str(param) + '.json')
                     for param in params]
        cmds = [[pattern.sub(str(param), substring) for substring in cmd] for param in params]
        cmds = [[*cmd, out] for (cmd, out) in zip(cmds, out_files)]
        print(" - Spawning {} processes ({} at a time)...".format(len(cmds), args.max_parallel))
        with ThreadPoolExecutor(args.max_parallel) as pool:
            exit_codes = pool.map(lambda cmd: call(cmd, stdout=DEVNULL, stderr=DEVNULL), cmds)

        print(" - All processes finished, with the following exit codes:")
        print("\n    | Test value | Exit code |")
        print("\n".join("    |{: >11} | {: <10}|".format(x, y) for x, y in zip(params, exit_codes)))
        print("\n - Investigate any unexpected exit codes by running your test command "
              "again with the corresponding parameter value.")

        for filename in os.listdir(out_dir):
            x_values.append(int(os.path.splitext(filename)[0]))

            with open(os.path.join(out_dir, filename), 'r') as out:
                result = json.load(out)

            # Memory usage
            graph = result['graphProfile']
            data['exchange_bytes'].append(sum_sum(graph['exchanges']['codeBytesByTile']))
            data['memory_bytes_gaps'].append(sum(graph['memory']['byTile']['totalIncludingGaps']))
            liveness = graph['memory']['liveness']
            data['memory_bytes'].append(sum(liveness['alwaysLive']['bytesByTile']) +
                                        sum(liveness['notAlwaysLive']['maxBytesByTile']))

            # Vertex memory usage
            vertex_memory = graph['memory']['byVertexType']
            data['vertex_code'].append(sum_sum(vertex_memory['codeBytes']))
            data['vertex_state'].append(sum_sum(vertex_memory['vertexDataBytes']))
            data['vertex_descriptors'].append(sum_sum(vertex_memory['descriptorBytes']))
            data['vertex_edge_pointers'].append(sum_sum(vertex_memory['edgePtrBytes']))
            data['vertex_copy_pointers'].append(sum_sum(vertex_memory['copyPtrBytes']))

            # Element counts
            data['num_vars'].append(graph['graph']['numVars'])
            data['num_edges'].append(graph['graph']['numEdges'])
            data['num_vertices'].append(graph['graph']['numVertices'])
            data['num_compute_sets'].append(graph['graph']['numComputeSets'])

            # Simulated cycle counts
            # Host exchange and sync cycles are omitted as they are too large for the graph
            if 'simulation' in result['executionProfile']:
                simulation = result['executionProfile']['simulation']
                data['cycles'].append(simulation['cycles'])
                if 'compute' in simulation['tileCycles']:
                    data['compute_cycles'].append(simulation['tileCycles']['compute'])
                    data['exchange_cycles'].append(simulation['tileCycles']['doExchange'])


def plot_data(var_name, x_values, data, args):
    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(7, 7))
    fig.suptitle(args.title if args.title else " ".join(args.command),
                 fontsize="small", wrap=True)

    # X-axis shared so we only need to set xlabel for last row of plots
    axs[-1, 0].set_xlabel(var_name)
    axs[-1, 1].set_xlabel(var_name)

    ############################################################################
    axs[0, 0].set_title("Total Memory Usage", fontsize="medium")
    axs[0, 0].plot(x_values, np.array(data['memory_bytes'])/MB_SCALE, "C1o",
                   x_values, np.array(data['memory_bytes_gaps'])/MB_SCALE, "C2^")
    axs[0, 0].legend(["Excluding gaps", "Including gaps"], fontsize="small")
    axs[0, 0].set_ylabel("Size (MB)")


    ############################################################################
    axs[2, 0].set_title("Memory Usage", fontsize="medium")
    axs[2, 0].plot(x_values, np.array(data['exchange_bytes'])/MB_SCALE, "C1o",
                   x_values, np.array(data['vertex_code'])/MB_SCALE, "C2^")
    axs[2, 0].legend(["Exchange code", "Vertex code"], fontsize="small")
    axs[2, 0].set_ylabel("Size (MB)")


    ############################################################################
    axs[0, 1].set_title("Cycle counts", fontsize="medium")
    axs[0, 1].yaxis.tick_right()
    if data['cycles']:
        axs[0, 1].plot(x_values, data['cycles'], "C1o", label="Total")
    if data['compute_cycles']:
        axs[0, 1].plot(x_values, data['compute_cycles'], "C2^", label="Compute (inc. idle)")
        axs[0, 1].plot(x_values, data['exchange_cycles'], "C3*", label="Exchange")
    else:
        print("Please enable 'debug.instrumentCompute' to see all cycle counts.")
    axs[0, 1].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0), fontsize="small")
    axs[0, 1].set_ylabel("Cycles")


    ############################################################################
    axs[2, 1].set_title("Element counts", fontsize="medium", pad=36)
    axs[2, 1].yaxis.tick_right()
    axs[2, 1].plot(x_values, data['num_vars'], "C1o",
                   x_values, data['num_vertices'], "C2^",
                   x_values, data['num_edges'], "C3*",
                   x_values, data['num_compute_sets'], "C4s")
    axs[2, 1].legend(["Variables", "Vertices", "Edges", "Compute sets"],
                     ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1), fontsize="small")
    axs[2, 1].set_ylabel("Quantity")


    ############################################################################
    axs[1, 0].set_title("Vertex Data Memory Usage", fontsize="medium")
    data = np.array([data['vertex_state'], data['vertex_edge_pointers'],
                     data['vertex_copy_pointers'], data['vertex_descriptors']])
    data = np.divide(data, MB_SCALE) # Scale bytes -> megabytes
    width = ceil((max(x_values) - min(x_values)) / max([(4 * (len(x_values) - 1)), 1]))
    cum_size = np.zeros(len(data[0]))
    for _, row_data in enumerate(data):
        axs[1, 0].bar(x_values, row_data, width, bottom=cum_size)
        cum_size += row_data
    axs[1, 0].legend(["State", "Edge pointers", "Copy pointers", "Descriptors"],
                     loc="center left", ncol=2, bbox_to_anchor=(1, 0.5), fontsize="small")
    axs[1, 0].set_ylabel("Size (MB)")

    axs[1, 1].remove() # Remove empty subplot
    for _, subplot in np.ndenumerate(axs):
        subplot.set_ylim(bottom=0)
        subplot.grid()
    plt.savefig(args.output)
    print(" - Plots saved to {}.".format(args.output))


if __name__ == "__main__":
    main()
