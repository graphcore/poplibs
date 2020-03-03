#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd, All rights reserved.

"""
A tool to plot graph and execution statistics for a Poplar program as a
single parameter is methodically incremented.
"""

import argparse
import json
import os
import re
import tempfile
from math import ceil
from subprocess import DEVNULL, Popen

import matplotlib.pyplot as plt
import numpy as np

FIGURE_FILENAME = "plot_metric_variation.png"
MB_SCALE = 1024*1024

def sum_sum(lists):
    """Returns total sum for list of lists."""
    return sum(sum(x) for x in lists)


def main():
    parser = argparse.ArgumentParser(
        description="A tool to plot graph and execution statistics for a Poplar "
        "program as a single parameter is methodically incremented. The plot figure "
        "is saved to '" + FIGURE_FILENAME + "', unless set with '--output'."
    )
    parser.add_argument(
        "--output", default=FIGURE_FILENAME, help="Name of file to output plots (in PNG format)."
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="The program command to call and "
        "analyse. The command must include one occurrence of pattern '{A:B:C}' (all integers). "
        "The command will be executed several times with this pattern replaced by "
        "a single value from D = A, A+C, A+C+C, ... for all D < B."
    )
    args = parser.parse_args()
    cmd = args.command
    cmd.append("--profile-json")

    var_pattern = re.compile(r'\{(\d+):(\d+):(\d+)\}')
    all_var_matches = list(filter(var_pattern.search, cmd))
    assert len(all_var_matches) is 1, r"Exactly one occurrence of {A:B:C} must appear in command."
    var_match = var_pattern.search(all_var_matches[0])
    params = np.arange(int(var_match.group(1)), int(var_match.group(2)), int(var_match.group(3)))

    name_pattern = re.compile(r'--?([a-zA-Z\-]+)[=|\s]' + var_pattern.pattern)
    var_name = name_pattern.search(" ".join(cmd)).group(1)

    cycles, compute_cycles, exchange_cycles = [], [], []
    memory_bytes, memory_bytes_gaps, exchange_bytes = [], [], []
    num_vars, num_vertices, num_edges, num_compute_sets = [], [], [], []
    vertex_state, vertex_edge_pointers, vertex_copy_pointers = [], [], []
    vertex_descriptors, vertex_code, x_values = [], [], []

    with tempfile.TemporaryDirectory() as out_dir:
        # Create list of all cmd variants (variable value and JSON output file)
        out_files = [os.path.join(out_dir, str(param) + '.json') for param in params]
        cmds = [[var_pattern.sub(str(param), substring) for substring in cmd] for param in params]
        cmds = [[*cmd, out] for (cmd, out) in zip(cmds, out_files)]

        print("Spawning %d processes..." % len(cmds))
        procs = [Popen(cmd, stdout=DEVNULL, stderr=DEVNULL) for cmd in cmds]
        exit_codes = [proc.wait() for proc in procs]

        print("All processes finished, with the following exit codes:")
        print("\nParameter value : Command exit code")
        print("\n".join("{: >15} : {}".format(x, y) for x, y in zip(params, exit_codes)))
        print("\nInvestigate any unexpected exit codes by running your test command "
              "again with the corresponding parameter value.")

        for filename in os.listdir(out_dir):
            x_values.append(int(os.path.splitext(filename)[0]))

            with open(os.path.join(out_dir, filename), 'r') as out:
                result = json.load(out)

            # Memory usage
            graph = result['graphProfile']
            exchange_bytes.append(sum_sum(graph['exchanges']['codeBytesByTile']))
            memory_bytes_gaps.append(sum(graph['memory']['byTile']['totalIncludingGaps']))
            liveness = graph['memory']['liveness']
            memory_bytes.append(sum(liveness['alwaysLive']['bytesByTile']) +
                                sum(liveness['notAlwaysLive']['maxBytesByTile']))

            # Vertex memory usage
            vertex_memory = graph['memory']['byVertexType']
            vertex_code.append(sum_sum(vertex_memory['codeBytes']))
            vertex_state.append(sum_sum(vertex_memory['vertexDataBytes']))
            vertex_descriptors.append(sum_sum(vertex_memory['descriptorBytes']))
            vertex_edge_pointers.append(sum_sum(vertex_memory['edgePtrBytes']))
            vertex_copy_pointers.append(sum_sum(vertex_memory['copyPtrBytes']))

            # Element counts
            num_vars.append(graph['graph']['numVars'])
            num_edges.append(graph['graph']['numEdges'])
            num_vertices.append(graph['graph']['numVertices'])
            num_compute_sets.append(graph['graph']['numComputeSets'])

            # Simulated cycle counts
            # Host exchange and sync cycles are omitted as they are too large for the graph
            if 'simulation' in result['executionProfile']:
                simulation = result['executionProfile']['simulation']
                cycles.append(simulation['cycles'])
                if 'compute' in simulation['tileCycles']:
                    compute_cycles.append(simulation['tileCycles']['compute'])
                    exchange_cycles.append(simulation['tileCycles']['doExchange'])


    if not x_values:
        print("No data found.")
        exit()

    _, axs = plt.subplots(3, 2, sharex=True, figsize=(7, 7))
    # X-axis shared so we only need to set xlabel for last row of plots
    axs[-1, 0].set_xlabel(var_name)
    axs[-1, 1].set_xlabel(var_name)

    ############################################################################
    axs[0, 0].set_title("Total Memory Usage", fontsize="medium")
    axs[0, 0].plot(x_values, np.array(memory_bytes)/MB_SCALE, "C1o",
                   x_values, np.array(memory_bytes_gaps)/MB_SCALE, "C2^")
    axs[0, 0].legend(["Excluding gaps", "Including gaps"], fontsize="small")
    axs[0, 0].set_ylabel("Size (MB)")


    ############################################################################
    axs[2, 0].set_title("Memory Usage", fontsize="medium")
    axs[2, 0].plot(x_values, np.array(exchange_bytes)/MB_SCALE, "C1o",
                   x_values, np.array(vertex_code)/MB_SCALE, "C2^")
    axs[2, 0].legend(["Exchange code", "Vertex code"], fontsize="small")
    axs[2, 0].set_ylabel("Size (MB)")


    ############################################################################
    axs[0, 1].set_title("Cycle counts", fontsize="medium")
    axs[0, 1].yaxis.tick_right()
    if cycles:
        axs[0, 1].plot(x_values, cycles, "C1o", label="Total")
    if compute_cycles:
        axs[0, 1].plot(x_values, compute_cycles, "C2^", label="Compute (inc. idle)")
        axs[0, 1].plot(x_values, exchange_cycles, "C3*", label="Exchange")
    else:
        print("Please enable 'debug.instrumentCompute' to see all cycle counts.")
    axs[0, 1].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0), fontsize="small")
    axs[0, 1].set_ylabel("Cycles")


    ############################################################################
    axs[2, 1].set_title("Element counts", fontsize="medium", pad=36)
    axs[2, 1].yaxis.tick_right()
    axs[2, 1].plot(x_values, num_vars, "C1o",
                   x_values, num_vertices, "C2^",
                   x_values, num_edges, "C3*",
                   x_values, num_compute_sets, "C4s")
    axs[2, 1].legend(["Variables", "Vertices", "Edges", "Compute sets"],
                     ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1), fontsize="small")
    axs[2, 1].set_ylabel("Quantity")


    ############################################################################
    axs[1, 0].set_title("Vertex Data Memory Usage", fontsize="medium")
    data = np.array([vertex_state, vertex_edge_pointers, vertex_copy_pointers, vertex_descriptors])
    data = np.divide(data, MB_SCALE) # Scale bytes -> megabytes
    width = ceil(float(var_match.group(3))/4)
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
    print("Plots saved to %s." % args.output)
if __name__ == "__main__":
    main()
