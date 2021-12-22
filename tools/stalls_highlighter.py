#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
import collections
from dataclasses import dataclass
import logging
import re
import os
import subprocess


# Following regex designed to parse next log:
#   t[0.0]: 0x0004c5e0 (0x1319315a): __vertexCode_40_.text.__runCodelet_poplin__ConvPartial1x4SLIC___half_half4 + 292:   brnzdec      $m1 [ 0x00000002 ], 0x0004c568 m @ 8902
# or
#   t[0.0]: 0x0004c568 (0x1361317f): __vertexCode_40_.text.__runCodelet_poplin__ConvPartial1x4SLIC___half_half4 + 172:   -  m @ 8903
#
sim_line_match = re.compile(r'^t\[(.+)\].+\.text\.__runCodelet_(.+)\s\+\s\d+:(.+)\sm\s@\s(\d+)$')

sim_code_data = collections.namedtuple('sim_code_data', ['tile', 'codelet', 'command', 'cycles_counter'])


# -----------------------------------------------------------------------------
# Colouring class
# -----------------------------------------------------------------------------
class bcolors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'


# -----------------------------------------------------------------------------
# Output function
# -----------------------------------------------------------------------------
def custom_log(message, print_allowed, colour=bcolors.RESET):
   if print_allowed:
      print(colour + message + bcolors.RESET)


# -----------------------------------------------------------------------------
# Colour stalls
# -----------------------------------------------------------------------------
def highlight_stalls(file_path, target_codelet, target_tile, print_allowed=False,):
    found_stall = False
    highlight_command = False
    count_up_empty_command = 0

    if not target_tile and not target_codelet:
      logging.err('At least one out of following options shall be set: codelet, tile')
      exit(1)

    with open(file_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            codelet_matched = False
            tile_matched = False
            match = sim_line_match.match(lines[i])

            if match:
                data_fields = sim_code_data._make(x for x in match.groups())
                if not target_tile and data_fields.tile.find(target_tile) != -1:
                    tile_matched = True

                if not target_codelet and data_fields.codelet.find(target_codelet) != -1:
                    codelet_matched = True

                if codelet_matched or tile_matched:
                    if data_fields.command.strip() is '-':
                        highlight_command = True
                        count_up_empty_command += 1
                        custom_log(lines[i][:-1], print_allowed, bcolors.RED)
                    elif highlight_command is True:
                        highlight_command = False
                        # Presence of the 6 empty commands is a marker for a register stall
                        if count_up_empty_command is 6:
                            found_stall = True
                            count_up_empty_command = 0
                        custom_log(lines[i][:-1], print_allowed, bcolors.YELLOW)
                    else:
                        custom_log(lines[i][:-1], print_allowed)

    return found_stall


# -----------------------------------------------------------------------------
# Execute test
# -----------------------------------------------------------------------------
def execute_test(binary, workspace, device_type):
    """Reads output of a simulator and highlights any registers stall.
       To correctly detect the stalls a poplar program needs to run with
       POPLAR_ENGINE_OPTIONS=\'{\"debug.outputAllSymbols\":true}\' and
       POPLAR_SIMULATOR_OPTIONS=\'{\"debug.trace\":true,\"sim.accurateTiming\":true,\"sim.traceTiming\":true}\'.
    """

    env_options = {"POPLAR_ENGINE_OPTIONS":"{\"debug.outputAllSymbols\":true}",\
                   "POPLAR_SIMULATOR_OPTIONS":"{\"debug.trace\":true,\"sim.accurateTiming\":true,\"sim.traceTiming\":true}"}

    for key, item in env_options.items():
        os.environ[key] = item

    file_path = os.path.join(workspace, 'stall_reg.log')


    cmd = [binary, "--input-channels=1", "--output-channels=1", "--convolution-options={\"partialsType\":\"half\"}", "--tiles-per-ipu=1", \
                "--single-phase=fwd", "--fwd-plan-constraints={\"method\":\"SLIC\"}", "--field={4,4}", "--kernel-size={4,4}", \
                f'--device-type={device_type}', "--conv-groups=16"]

    logging.debug('Executing:')
    print_cmd = " ".join(cmd)
    for key, item in env_options.items():
        os.environ[key] = item
        print_cmd = f'{key}={item} ' + print_cmd
    logging.debug(print_cmd)

    # Prevent colour codes as they cause problems with the regex matches
    env = os.environ.copy()
    env["CLICOLOR_FORCE"] = "0"
    env["CLICOLOR"] = "0"
    with open(file_path, mode="w", encoding='utf-8') as log_file:
        subprocess.call(cmd, stdout=log_file, stderr=log_file, env=env)

    return file_path


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Highlight registers stalls (cycle bubbles) emitted in supervisor/worker code. This script can be run locally with a --ci-test option enabled to obtain a command test example')
    parser.add_argument('--workspace', default=os.getcwd(), help='Absolute path to a workspace. Default path is a current folder')
    parser.add_argument('--test-binary', help='Provides a path to the single_conv_layer tool')
    parser.add_argument('--sim-trace', help='Simulator execution trace file')
    parser.add_argument('--ci-test', default=False, action='store_true', help='Predefined CI test to ensure regex continues matching sim output')
    parser.add_argument('--codelet-name', default='', help='Specify codelet name to filter for. For example: ConvPartial')
    parser.add_argument('--tile-id', default='0.0', help='Specify tile.worker pair. For example: Tile 1, SUP code -> 1.0; Tile 5, WORKER 5 code -> 5.6')
    parser.add_argument("--device-type", choices=("Sim2",), default="Sim2", help="Only Sim2 is supported")
    args = parser.parse_args()

    # Make sure workspace folder  exists
    if not os.path.exists(args.workspace):
        os.makedirs(args.workspace)

    # Setup logging so that debug messages will be logged only to a file
    # Although, for running on CI it's beneficial to stdout everything
    stdout_logger = logging.StreamHandler()
    if args.ci_test:
        stdout_logger.setLevel(logging.DEBUG)
    else:
        stdout_logger.setLevel(logging.INFO)

    log_file_path = os.path.join(args.workspace, 'sup_code_analyzer.log')
    file_logger = logging.FileHandler(log_file_path, mode='w')
    file_logger.setLevel(logging.DEBUG)

    logging.basicConfig(
        level=os.environ.get("SUP_CODE_ANALYZER_LOG_LEVEL", default='DEBUG'),
        format="[%(levelname)s] %(asctime)s: %(message)s",
        handlers=[
            file_logger,
            stdout_logger
        ]
    )

    if args.ci_test:
        logging.info('Running a ci test (Coloured output disabled)')
        if not args.test_binary:
            logging.error('test-binary options was not provided.')
            exit(1)
        sim_log_file = execute_test(args.test_binary, args.workspace, args.device_type)
    elif args.sim_trace:
        logging.info('Standalone run. Reading from a file (Coloured output enabled)')
        sim_log_file = os.path.join(args.workspace, args.sim_trace)
    else:
        logging.error('At least on of the options shall be provided: --sim-trace or --ci-test ')
        exit(1)

    # Read a file and highlight stalls if not in CI mode
    logging.info('Checking for stalls')
    stalls_state = highlight_stalls(sim_log_file, args.codelet_name, args.tile_id, not args.ci_test)

    if stalls_state:
        logging.warn('Found registers stalls')

    # For the purpose on CI testing - appearance of stalls considered as
    # success for a reason to test functionality of this script and poplar
    # simulator options
    if args.ci_test:
        stalls_state = stalls_state > 0

        logging.info('Removing temp files')
        if os.path.exists(sim_log_file):
            os.remove(sim_log_file)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

    return stalls_state


# -----------------------------------------------------------------------------
# Super MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
