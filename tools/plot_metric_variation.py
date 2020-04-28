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
from subprocess import DEVNULL, call
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg') # Do not load GTK (prevents warning message)
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
VAR_PATTERN_RANGE = re.compile(r'\\?\{(\d+):(\d+):(\d+)\}') # E.g. {10:100:5}
VAR_PATTERN_LIST = re.compile(r'\\?\{((\d+,?)+)\}') # E.g. {1,2,3,4}
PROFILER_FILE_KEYWORD = "PROFILE_FILE"
GRAPH_PROFILE_FILE_KEYWORD = "GRAPH_PROFILE_FILE"
EXECUTION_PROFILE_FILE_KEYWORD = "EXECUTION_PROFILE_FILE"
PARAMETER_KEYWORD = "<PARAM>"
MB_SCALE = 1024 * 1024

matplotlib.rcParams['ytick.labelsize'] = 'x-small'

def sum_sum(lists):
    """Returns total sum for list of lists."""
    return sum(sum(x) for x in lists)


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''A tool to plot graph and execution statistics for a Poplar program as a \
single parameter is methodically incremented.

-----------------------------
### TEST COMMAND VARIABLE ###
-----------------------------
The test command must include a parameter to be varied which may be represented in two ways:

 - {{A:B:C}} - for which the command will be executed once for each value D = A, A+C, A+C+C, ... \
for all D < B.
 - {{E,F,G...,H}} - in which the command will be executed once for each value in the list. All \
values must be integers.

Remember to escape the opening brace - e.g. \\{{ - to avoid shell expansion.

-----------------------------
### TEST COMMAND KEYWORDS ###
-----------------------------
Your test command may also contain the following keywords:

 - {}
 - {}
 - {}

Each of which will be replaced with a filename for the respective profile file or graph & \
execution profile files. If none of these keywords are used --profile-json <file> will be appended \
to the test command instead.

You may also include {} anywhere in your test command, which will be replaced with the parameter \
value currently being tested.'''.format(PROFILER_FILE_KEYWORD, GRAPH_PROFILE_FILE_KEYWORD,
                                        EXECUTION_PROFILE_FILE_KEYWORD, PARAMETER_KEYWORD),
        epilog='''\
----------------
### EXAMPLES ###
----------------
 - Show how Resnet metrics scale with batch-size, saving Pickle and Png to default locations:
     {0} resnet --variant RESNET_32 --batch-size \\{{10:51:10}}

 - Same again but using variable list instead of range - save to custom Pickle file:
     {0} --pickle my.pickle resnet --variant RESNET_32 --batch-size \\{{10,20,30,40,50}}

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
        help="[=%(default)s] Name of Pickle file used for plot data save/restore. Pickle file is "
        "used for output if 'command' is provided - input otherwise."
    )
    parser.add_argument(
        "--pickle-off", action='store_true',
        help="[=%(default)s] Disable save/restore of plot data in Pickle file."
    )
    parser.add_argument(
        "--pickle-merge", nargs='*', help="Got results spread across several Pickle files? List "
        "them all here and they'll be plotted on the same graph and Pickled to --pickle. Cannot be "
        "used if `command` is provided."
    )
    parser.add_argument(
        "--title", help="Plot figure title. If not set, test command will be used."
    )
    parser.add_argument(
        "--max-parallel", type=int, default=8, help="[=%(default)s] Maximum number of parallel "
        "processes to be running at any time."
    )
    parser.add_argument(
        "--num-iterations", type=int, default=1, help="[=%(default)s] Indicate how many iterations "
        "your test command uses - so that item rates are calculated correctly."
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER,
        help="Must be last argument. The program test command to call and analyse. "
    )
    return parser.parse_args()


def get_var_name_params(cmd):
    all_var_matches_range = list(filter(VAR_PATTERN_RANGE.search, cmd))
    all_var_matches_list = list(filter(VAR_PATTERN_LIST.search, cmd))
    var_name_pattern = r'--?([a-zA-Z\-]+)[=|\s][\"|\']?'
    if len(all_var_matches_range) is 1:
        var_match = VAR_PATTERN_RANGE.search(all_var_matches_range[0])
        params = np.arange(int(var_match.group(1)),
                           int(var_match.group(2)),
                           int(var_match.group(3)))

        name_pattern = re.compile(var_name_pattern + VAR_PATTERN_RANGE.pattern)
        var_name = name_pattern.search(" ".join(cmd)).group(1)
        return (var_name, params, VAR_PATTERN_RANGE)
    elif len(all_var_matches_list) is 1:
        var_match = VAR_PATTERN_LIST.search(all_var_matches_list[0]).group(1)
        params = list(map(int, var_match.split(",")))
        name_pattern = re.compile(var_name_pattern + VAR_PATTERN_LIST.pattern)
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


def get_commands(args, param_pattern, params, out_dir):
    file_pattern_count = 0
    outfile_prefix = os.path.join(out_dir, PARAMETER_KEYWORD)
    cmd = []

    for sub in args.command:
        sub = sub.replace(GRAPH_PROFILE_FILE_KEYWORD, outfile_prefix + '.graph_json')
        sub = sub.replace(EXECUTION_PROFILE_FILE_KEYWORD, outfile_prefix + '.exec_json')
        sub = sub.replace(PROFILER_FILE_KEYWORD, outfile_prefix + '.json')
        file_pattern_count += sub.count(PARAMETER_KEYWORD)
        sub = param_pattern.sub(PARAMETER_KEYWORD, sub)
        cmd.append(sub)

    if file_pattern_count is 0:
        cmd.append('--profile-json')
        cmd.append(outfile_prefix + '.json')

    return [[sub.replace(PARAMETER_KEYWORD, str(param)) for sub in cmd] for param in params]


def generate_data(args, params, pattern, x_values, data):
    assert (not os.path.dirname(args.pickle) or os.access(os.path.dirname(args.pickle), os.W_OK)),\
           "- ERROR: Cannot create {}.".format(args.pickle)

    with tempfile.TemporaryDirectory() as out_dir:
        commands = get_commands(args, pattern, params, out_dir)
        print(" - Spawning {} processes ({} at a time)...".format(len(commands), args.max_parallel))
        with ThreadPoolExecutor(args.max_parallel) as pool:
            exit_codes = pool.map(lambda cmd: call(cmd, stdout=DEVNULL, stderr=DEVNULL), commands)

        print(" - All processes finished, with the following exit codes:")
        print("\n   | Test | Exit |")
        print("   | value| code | Command")
        print("\n".join("   |{: >5} | {: <5}| {}".format(x, y, " ".join(z))
                        for x, y, z in zip(params, exit_codes, commands)))
        print("\n - Investigate any unexpected exit codes by running your test command "
              "again with the corresponding parameter value.")

        for filename in os.listdir(out_dir):
            # To maintain correct order, we will use .graph_json to find matching .exec_json files.
            if filename.endswith(".exec_json"):
                continue

            batch_size = int(os.path.splitext(filename)[0])
            x_values.append(batch_size)

            with open(os.path.join(out_dir, filename), 'r') as out:
                result = json.load(out)

            execution = {}
            graph = {}
            if filename.endswith(".graph_json"):
                graph = result
                exec_path = os.path.join(out_dir, str(batch_size) + '.exec_json')
                if os.path.isfile(exec_path):
                    with open(exec_path, 'r') as out:
                        execution = json.load(out)
            else:
                graph = result['graphProfile']
                execution = result['executionProfile']

            # Target constants
            data['total_memory'] = graph['target']['totalMemory']
            data['bytes_per_tile'] = graph['target']['bytesPerTile']
            data['clock_frequency'] = graph['target']['clockFrequency']

            # Memory usage
            data['exchange_bytes'].append(sum_sum(graph['exchanges']['codeBytesByTile']))
            data['memory_bytes_gaps'].append(sum(graph['memory']['byTile']['totalIncludingGaps']))
            liveness = graph['memory']['liveness']
            data['memory_bytes'].append(sum(liveness['alwaysLive']['bytesByTile']) +
                                        sum(liveness['notAlwaysLive']['maxBytesByTile']))
            data['max_bytes'].append(max(liveness['alwaysLive']['bytesByTile'] +
                                         liveness['notAlwaysLive']['maxBytesByTile']))
            data['max_bytes_gaps'].append(max(graph['memory']['byTile']['totalIncludingGaps']))

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
            if 'simulation' in execution:
                data['cycles'].append(execution['simulation']['cycles'])
                tile_cycles = execution['simulation']['tileCycles']
                if 'compute' in tile_cycles:
                    data['compute_cycles'].append(tile_cycles['compute'])
                    data['exchange_cycles'].append(tile_cycles['doExchange'])


def plot_data(var_name, x_values, data, args):
    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(7, 7))
    fig.suptitle(args.title if args.title else " ".join(args.command),
                 fontsize="small", wrap=True)

    # X-axis shared so we only need to set xlabel for last row of plots
    axs[-1, 0].set_xlabel(var_name)
    axs[-1, 1].set_xlabel(var_name)

    ############################################################################
    axs[0, 0].set_title("Memory Usage (squares = including gaps)", fontsize="medium")
    axs[0, 0].plot(x_values, np.array(data['memory_bytes'])/MB_SCALE, "C5o")
    axs[0, 0].plot(x_values, np.array(data['memory_bytes_gaps'])/MB_SCALE, "C5s", fillstyle='none')
    axs[0, 0].set_ylabel("Total Usage (MB)", color='C5')
    axs[0, 0].tick_params(axis='y', labelcolor='C5')
    _, ymax = axs[0, 0].get_ylim()
    axs[0, 0].axhline(data['total_memory'] / MB_SCALE, dashes=[0, 4, 4, 0], color='C5')

    max_ax = axs[0, 0].twinx()
    max_ax.plot(x_values, np.array(data['max_bytes'])/MB_SCALE, "C0o")
    max_ax.plot(x_values, np.array(data['max_bytes_gaps'])/MB_SCALE, "C0s", fillstyle='none')
    max_ax.set_ylabel("Max. Tile Usage (MB)", color='C0')
    max_ax.tick_params(axis='y', labelcolor='C0')
    _, ymax = max_ax.get_ylim()
    max_ax.axhline(data['bytes_per_tile'] / MB_SCALE, dashes=(4, 4), color='C0')
    max_ax.set_ylim(bottom=0)

    ############################################################################
    axs[2, 0].set_title("Memory Usage", fontsize="medium")
    axs[2, 0].plot(x_values, np.array(data['exchange_bytes'])/MB_SCALE, "C0o",
                   x_values, np.array(data['vertex_code'])/MB_SCALE, "C1^")
    axs[2, 0].legend(["Exchange code", "Vertex code"], fontsize="small")
    axs[2, 0].set_ylabel("Usage (MB)")


    ############################################################################
    axs[2, 1].set_title("Cycle Counts (and item rate)", fontsize="medium")
    axs[2, 1].yaxis.tick_right()
    item_rate_ax = axs[2, 1].twinx()
    if data['cycles']:
        axs[2, 1].plot(x_values, np.array(data['cycles'])*100, "C1^", label="Total (x100)")
        item_rates = (args.num_iterations * data['clock_frequency'] *
                      np.array(x_values) / np.array(data['cycles']))
        item_rate_ax.plot(x_values, item_rates, "C0o", label="Items rate", fillstyle='none')
        item_rate_ax.set_ylabel("Items/second", color='C0')
        item_rate_ax.tick_params(axis='y', labelcolor='C0')
        item_rate_ax.set_ylim(bottom=0)
    if data['compute_cycles']:
        axs[2, 1].plot(x_values, data['exchange_cycles'], "C2P", label="Exchange")
        axs[2, 1].plot(x_values, data['compute_cycles'], "C3*", label="Compute (inc. idle)")
    else:
        print("Please enable 'debug.instrumentCompute' to see all cycle counts.")

    lines, labels = axs[2, 1].get_legend_handles_labels()
    lines2, labels2 = item_rate_ax.get_legend_handles_labels()
    axs[2, 1].legend(lines + lines2, labels + labels2, ncol=2, loc="lower center",
                     bbox_to_anchor=(0.5, 1.12), fontsize="small")
    axs[2, 1].set_ylabel("Cycles")


    ############################################################################
    axs[1, 0].set_title("Element Counts", fontsize="medium")
    axs[1, 0].plot(x_values, data['num_vars'], "C0o",
                   x_values, data['num_vertices'], "C1^",
                   x_values, data['num_edges'], "C2P",
                   x_values, np.array(data['num_compute_sets'])*1000, "C3*")
    axs[1, 0].legend(["Variables", "Vertices", "Edges", "Compute sets (x1000)"],
                     ncol=2, loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    axs[1, 0].set_ylabel("Quantity")


    ############################################################################
    axs[0, 1].set_title("Vertex Data Memory Usage", fontsize="medium")
    axs[0, 1].yaxis.tick_right()
    axs[0, 1].yaxis.set_label_position("right")

    v_data = np.array([data['vertex_state'], data['vertex_edge_pointers'],
                       data['vertex_copy_pointers'], data['vertex_descriptors']])
    v_data = np.divide(v_data, MB_SCALE) # Scale bytes -> megabytes
    width = max(0.1, (max(x_values) - min(x_values)) / max((4 * (len(x_values) - 1)), 1))
    cum_size = np.zeros(len(v_data[0]))
    for _, row_data in enumerate(v_data):
        axs[0, 1].bar(x_values, row_data, width, bottom=cum_size, zorder=3)
        cum_size += row_data
    axs[0, 1].legend(["State", "Edge pointers", "Copy pointers", "Descriptors"],
                     loc="upper center", ncol=2, fontsize="small",
                     bbox_to_anchor=(0.5, 0))
    axs[0, 1].set_ylabel("Usage (MB)")

    axs[1, 1].remove() # Remove empty subplot
    for _, subplot in np.ndenumerate(axs):
        subplot.set_ylim(bottom=0)
        subplot.grid()
    plt.savefig(args.output)
    print(" - Plots saved to {}.".format(args.output))


if __name__ == "__main__":
    main()
