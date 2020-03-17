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
from subprocess import DEVNULL, Popen

import matplotlib
matplotlib.use('Agg') # Do not load GTK (prevents warning message)
import matplotlib.pyplot as plt
import numpy as np

FIGURE_FILENAME = "plot_metric_variation.png"
PICKLE_FILENAME = "plot_metric_variation.pickle"
VAR_PATTERN_RANGE = re.compile(r'\{(\d+):(\d+):(\d+)\}') # E.g. {10:100:5}
VAR_PATTERN_LIST = re.compile(r'\{((\d+,?)+)\}') # E.g. {1,2,3,4}
MB_SCALE = 1024*1024


def sum_sum(lists):
    """Returns total sum for list of lists."""
    return sum(sum(x) for x in lists)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A tool to plot graph and execution statistics for a Poplar "
        "program as a single parameter is methodically incremented."
        "NOTE: Your test executable must support the '--profile-json' option flag."
    )
    parser.add_argument(
        "--output", default=FIGURE_FILENAME, help="Name of file to output plots (in PNG format)."
    )
    parser.add_argument(
        "--pickle", default=PICKLE_FILENAME, help="Name of Pickle file to use for plot data "
        "save/restore. Pickle file is used for output if 'command' is provided - input otherwise."
    )
    parser.add_argument(
        "--pickle-off", action='store_true', help="Disable save/restore of plot data in Pickle file."
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="The program command to call and "
        "analyse. The command must include a parameter to be varied which can"
        " be represented in two ways. Either the pattern '{A:B:C}' (all integers); "
        "The command will be executed several times with this pattern replaced by "
        "a single value from D = A, A+C, A+C+C, ... for all D < B."
        " Or '{D,E,F...,G}'; the command will be executed once for each value"
        " in the list."
    )
    return parser.parse_args()


def get_var_name_params(cmd):
    all_var_matches_range = list(filter(VAR_PATTERN_RANGE.search, cmd))
    all_var_matches_list = list(filter(VAR_PATTERN_LIST.search, cmd))
    if len(all_var_matches_range) is 1:
      var_match = VAR_PATTERN_RANGE.search(all_var_matches_range[0])
      params = np.arange(int(var_match.group(1)), int(var_match.group(2)), int(var_match.group(3)))

      name_pattern = re.compile(r'--?([a-zA-Z\-]+)[=|\s]' + VAR_PATTERN_RANGE.pattern)
      var_name = name_pattern.search(" ".join(cmd)).group(1)
      return (var_name, params, VAR_PATTERN_RANGE)
    elif len(all_var_matches_list) is 1:
      var_match = VAR_PATTERN_LIST.search(all_var_matches_list[0]).group(1)
      params=list(map(int, var_match.split(",")));
      name_pattern = re.compile(r'--?([a-zA-Z\-]+)[=|\s]' + VAR_PATTERN_LIST.pattern)
      var_name = name_pattern.search(" ".join(cmd)).group(1)
      return (var_name, params, VAR_PATTERN_LIST)
    else:
      assert len(all_var_matches) is 1, r"Exactly one occurrence of {A:B:C} or {D,E,F,..G} must appear in command."


def main():
    args = get_args()
    if args.command: # Command provided. Generate data before plotting.
        x_values = []
        data = defaultdict(list)
        (var_name, params, pattern) = get_var_name_params(args.command)
        generate_data(args.command, params, pattern, x_values, data)

        if not args.pickle_off:
            with open(args.pickle, 'wb') as pickle_file:
                pickle.dump(var_name, pickle_file)
                pickle.dump(x_values, pickle_file)
                pickle.dump(data, pickle_file)
                print(" - Plot data pickled to %s." % args.pickle)

    else: # No command provided. Load data from Pickle file.
        if not args.pickle_off:
            if os.path.exists(args.pickle):
                print(" - Loading plot data from %s." % args.pickle)
                with open(args.pickle, "rb") as pickle_file:
                    var_name = pickle.load(pickle_file)
                    x_values = pickle.load(pickle_file)
                    data = pickle.load(pickle_file)
            else:
                print(" - %s does not exist. Exiting." % args.pickle)
                exit()
        else:
            print("No command provided. Exiting.")
            exit()

    plot_data(var_name, x_values, data, args)


def generate_data(cmd, params, pattern, x_values, data):
    cmd.append("--profile-json")

    with tempfile.TemporaryDirectory() as out_dir:
        # Create list of all cmd variants (variable value and JSON output file)
        out_files = [os.path.join(out_dir, str(param) + '.json') for param in params]
        cmds = [[pattern.sub(str(param), substring) for substring in cmd] for param in params]
        cmds = [[*cmd, out] for (cmd, out) in zip(cmds, out_files)]

        print(" - Spawning %d processes..." % len(cmds))
        procs = [Popen(cmd, stdout=DEVNULL, stderr=DEVNULL) for cmd in cmds]
        exit_codes = [proc.wait() for proc in procs]

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
    if not x_values:
        print("No data found.")
        exit()

    _, axs = plt.subplots(3, 2, sharex=True, figsize=(7, 7))
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
    width = ceil((x_values[1] - x_values[0]) / 4)
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
    print(" - Plots saved to %s." % args.output)


if __name__ == "__main__":
    main()
