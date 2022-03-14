#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import collections
import csv
import itertools
import logging
import numpy
import math
from multiprocessing import Pool, TimeoutError
import re
import os
import subprocess
import sys
from progress.bar import Bar


NEW_LINE_CHAR = r'(?:\n|\r\n?)'
LOG_FILE_EXT = r'.conv.log'
NUMBER_OF_MATCHES = 6 # See capture_logs_info for an explanation
group_to_analyse = ['single_conv_layer']
phases_dict = {'fwd':0, 'bwd':1, 'wu':2}

# Monsto regex to capture planner output:
# 16:52:46.800 55378 PL [W] Found best plan using {"type":"AMP"}: Cost{cycles=30748, memory=51072, tiles=1120}.
# 16:52:46.800 55378 PL [D]   for input {14,14}x(256x1x4), kernel {14,14}, output = {3,3}x(256x1x256), pass=TRAINING_WU
# 16:52:46.800 55378 PL [D]   breakdown of memory and cycle estimates:
# 16:52:46.800 55378 PL [D]    - total parallel split: 1120
# 16:52:46.800 55378 PL [D]    - total serial split: 1
# 16:52:46.800 55378 PL [D]    - broadcast operands before loop: 0 copy cycles, 5012 exchange cycles, 0 bytes
# 16:52:46.800 55378 PL [D]    - rearrangement before slice: 0 cycles, 0 bytes (0 overhead, 0 per-loop iteration)
# 16:52:46.800 55378 PL [D]    - dynamic slice: 0 cycles, unknown bytes
# 16:52:46.800 55378 PL [D]    - transform: 33281 copy cycles, 9984 exchange cycles, 54784 bytes
# 16:52:46.800 55378 PL [D]    - exchange: 8233 cycles, n/a bytes. (Input 4178, Weight 359, Reduce 3696 + 0)
# 16:52:46.800 55378 PL [D]    - tile level transform: 0 cycles, 0 bytes
# 16:52:46.800 55378 PL [D]    - compute: 17338 cycles, 51072 bytes
# 16:52:46.800 55378 PL [D]    - reduction: 3756 cycles, 0 bytes
# 16:52:46.800 55378 PL [D]    - dynamic update: 0 cycles, unknown bytes
# 16:52:46.800 55378 PL [D]    - add in-place: 0 cycles, 0 bytes
# 16:52:46.800 55378 PL [D]    - cast: 0 cycles, 0 bytes
# 16:52:46.800 55378 PL [D]    - total: 30748 cycles, 51072 bytes
re_planner_info = re.compile(r'^.+Found best plan using {"type":"([A-Z]+_?[A-Z]+?)".*}: Cost\{cycles=(\d+), memory=(\d+), tiles=(\d+)\}.' + NEW_LINE_CHAR +
                           ".+pass=([A-Z]+)_([A-Z]+)"  + NEW_LINE_CHAR +
                          ".+" + NEW_LINE_CHAR +
                          ".+total parallel split:\s(\d+)" + NEW_LINE_CHAR +
                          ".+total serial split:\s(\d+)" + NEW_LINE_CHAR +
                          ".+broadcast operands before loop:\s(\d+)\scopy\scycles,\s(\d+)\sexchange\scycles,\s(\d+).+" + NEW_LINE_CHAR +
                          ".+rearrangement before slice:\s(\d+).+" + NEW_LINE_CHAR +
                          ".+dynamic slice:\s(\d+).+" + NEW_LINE_CHAR +
                          ".+transform:\s(\d+)\scopy\scycles,\s(\d+)\sexchange\scycles,\s(\d+)\sbytes\s\(input\s(\d+),\sweights\s(\d+).+" + NEW_LINE_CHAR +  #764 bytes (input 360, weights 22)
                          ".+exchange:\s(\d+).+Input\s(\d+),\sWeight\s(\d+),\sReduce\s(\d+)\s\+\s(\d+).+" + NEW_LINE_CHAR +
                          ".+tile level transform:\s(\d+).+" + NEW_LINE_CHAR +
                          ".+compute:\s(\d+).+" + NEW_LINE_CHAR +
                          ".+reduction:\s(\d+).+" + NEW_LINE_CHAR +
                          ".+dynamic update:\s(\d+).+" + NEW_LINE_CHAR +
                          ".+add in-place:\s(\d+).+" + NEW_LINE_CHAR +
                          ".+cast:\s(\d+).+$",
                          re.MULTILINE)

planner_info_fields_names = ['method', 'cost', 'memory', 'tiles', 'training', 'phase',
                 'parallelSplit', 'serialSplit', 'broadcastInputsCycles', 'broadcastExchangeCycles',
                 'broadcastBytes', 'rearrangeBeforeSlice', 'dynamicSlice', 'transformsCopyCycles',
                 'transformsExchangeCycles', 'transformsBytes', 'transformsCopyInputBytes',
                 'transformsCopyWeightsBytes', 'exchange', 'inputExchange', 'weightsExchange',
                 'reduceExchange', 'reduceExchangePlus', 'tileTransforms', 'compute', 'reduce',
                 'dynamicUpdate', 'addInPlace', 'cast']

PlannerInfoFields = collections.namedtuple(
    'PlannerInfoFields', planner_info_fields_names
)


# Regex to capture transforms info
# transform #0
#   Transform: extraFieldDims          0
#              dilatePostConv          {}
#              swapOperands            false
#              expandDims              {1}
#              outChanFlattenDims      {}
#              flattenDims             {0,2}
#              combineConvGroupsFactor 1
re_transform_info = re.compile(r'^transform\s\#0' + NEW_LINE_CHAR +
                             '\s+Transform:\sextraFieldDims\s+(\d+)' + NEW_LINE_CHAR +
                             '\s+dilatePostConv\s+(.+)' + NEW_LINE_CHAR +
                             '\s+swapOperands\s+([a-z]+)' + NEW_LINE_CHAR +
                             '\s+expandDims\s+(.+)' + NEW_LINE_CHAR +
                             '\s+outChanFlattenDims\s+(.+)' + NEW_LINE_CHAR +
                             '\s+flattenDims\s+(.+)' + NEW_LINE_CHAR +
                             '\s+combineConvGroupsFactor\s+(\d+)$',
                            re.MULTILINE)

transform_info_fields_names = ['extraFieldDims', 'dilatePostConv', 'swapOperands',
    'expandDims', 'outChanFlattenDims', 'flattenDims', 'combineConvGroupsFactor']
TransformInfoFields = collections.namedtuple(
    'TransformInfoFields', transform_info_fields_names
)

# Regex to capture partitions info
# partition #0
#   Partition: fieldSplit            {5,1}
#              batchSplit            1
#              outChanSplit.serial   1
#              outChanSplit.parallel 22
#              kernelSplit           {1,1}
#              inChanSplit.serial    1
#              inChanSplit.parallel  11
#              convGroupSplit        1
#              fieldAxisGrainSize    {1,1}
#              inChanGrainSize       16
#              outChanGrainSize      8
re_partition_info = re.compile(r'^partition\s\#0' + NEW_LINE_CHAR +
                             '\s+Partition:\sfieldSplit\s+(.+)' + NEW_LINE_CHAR +
                             '\s+batchSplit\s+(\d+)' + NEW_LINE_CHAR +
                             '\s+outChanSplit.serial\s+(\d+)' + NEW_LINE_CHAR +
                             '\s+outChanSplit.parallel\s+(\d+)' + NEW_LINE_CHAR +
                             '\s+kernelSplit\s+(.+)' + NEW_LINE_CHAR +
                             '\s+inChanSplit.serial\s+(\d+)' + NEW_LINE_CHAR +
                             '\s+inChanSplit.parallel\s+(\d+)' + NEW_LINE_CHAR +
                             '\s+convGroupSplit\s+(\d+)' + NEW_LINE_CHAR +
                             '\s+fieldAxisGrainSize\s+(.+)' + NEW_LINE_CHAR +
                             '\s+inChanGrainSize\s+(\d+)' + NEW_LINE_CHAR +
                             '\s+outChanGrainSize\s+(\d+)$',
                            re.MULTILINE)

partition_info_fields_names = ['fieldSplit', 'batchSplit', 'outChanSplit_serial',
    'outChanSplit_parallel', 'kernelSplit', 'inChanSplit_serial', 'inChanSplit_parallel',
    'convGroupSplit', 'fieldAxisGrainSize', 'inChanGrainSize', 'outChanGrainSize']
PartitionInfoFields = collections.namedtuple(
    'PartitionInfoFields', partition_info_fields_names
)


# Plan information
#        convGroupsPerGroup      1
#        inChansPerGroup         16
#        partialChansPerGroup    8
#        method                  {"type":"AMP"}
#        isJointPlan             0
#        startTile               0
#        linearizeTileDirection  ASCENDING
#        totalTiles              24
re_plan_info = re.compile(r'^\s+convGroupsPerGroup\s+(\d+)' + NEW_LINE_CHAR +
                          '\s+inChansPerGroup\s+(\d+)' + NEW_LINE_CHAR +
                           '\s+partialChansPerGroup\s+(\d+)' + NEW_LINE_CHAR +
                           '\s+method\s+{"type":\s*"([A-Z]+)".*' + NEW_LINE_CHAR +
                           '\s+isJointPlan\s+(\d+)' + NEW_LINE_CHAR +
                           '\s+startTile\s+(\d+)' + NEW_LINE_CHAR +
                           '\s+linearizeTileDirection\s+([A-Z]+)' + NEW_LINE_CHAR +
                           '\s+totalTiles\s+(\d+)$',
                           re.MULTILINE)

plan_info_fields_names = ['inChansPerGroup', 'convGroupsPerGroup', 'partialChansPerGroup',
    'method', 'isJointPlan', 'startTile', 'linearizeTileDirection', 'totalTiles']
PlanInfoFields = collections.namedtuple(
    'PlanInfoFields', plan_info_fields_names
)


# Another monsto regex to capture conv params:
#   Params:
#         inputType                  half
#         outputType                 half
#         batchSize                  256
#         numConvGroups              1
#         inputFieldShape            {14,14}
#         kernelShape                {14,14}
#         inputChannelsPerConvGroup  4
#         outputChannelsPerConvGroup 256
#         inputTruncationLower       {0,0}
#         inputTruncationUpper       {0,0}
#         inputDilation              {1,1}
#         inputPaddingLower          {1,1}
#         inputPaddingUpper          {1,1}
#         flipInput                  {0,0}
#         kernelTruncationLower      {0,0}
#         kernelTruncationUpper      {0,0}
#         kernelDilation             {1,1}
#         kernelPaddingLower         {0,0}
#         kernelPaddingUpper         {0,0}
#         flipKernel                 {0,0}
#         outputTruncationLower      {0,0}
#         outputTruncationUpper      {0,0}
#         stride                     {1,1}
#         outputPaddingLower         {0,0}
#         outputPaddingUpper         {0,0}
#         outputFieldShape           {3,3}
re_conv_params_info = re.compile(r"^\s+Params:" + NEW_LINE_CHAR +
                             "\s+inputType\s+([a-z]+)" + NEW_LINE_CHAR +
                             "\s+outputType\s+([a-z]+)" + NEW_LINE_CHAR +
                             "\s+batchSize\s+(\d+)" + NEW_LINE_CHAR +
                             "\s+numConvGroups\s+(\d+)" + NEW_LINE_CHAR +
                             "\s+inputFieldShape\s+(.+)" + NEW_LINE_CHAR +
                             "\s+kernelShape\s+(.+)" + NEW_LINE_CHAR +
                             "\s+inputChannelsPerConvGroup\s+(\d+)" + NEW_LINE_CHAR +
                             "\s+outputChannelsPerConvGroup\s+(\d+)" + NEW_LINE_CHAR +
                             "\s+inputTruncationLower\s+(.+)" + NEW_LINE_CHAR +
                             "\s+inputTruncationUpper\s+(.+)" + NEW_LINE_CHAR +
                             "\s+inputDilation\s+(.+)" + NEW_LINE_CHAR +
                             "\s+inputPaddingLower\s+(.+)" + NEW_LINE_CHAR +
                             "\s+inputPaddingUpper\s+(.+)" + NEW_LINE_CHAR +
                             "\s+flipInput\s+(.+)" + NEW_LINE_CHAR +
                             "\s+kernelTruncationLower\s+(.+)" + NEW_LINE_CHAR +
                             "\s+kernelTruncationUpper\s+(.+)" + NEW_LINE_CHAR +
                             "\s+kernelDilation\s+(.+)" + NEW_LINE_CHAR +
                             "\s+kernelPaddingLower\s+(.+)" + NEW_LINE_CHAR +
                             "\s+kernelPaddingUpper\s+(.+)" + NEW_LINE_CHAR +
                             "\s+flipKernel\s+(.+)" + NEW_LINE_CHAR +
                             "\s+outputTruncationLower\s+(.+)" + NEW_LINE_CHAR +
                             "\s+outputTruncationUpper\s+(.+)" + NEW_LINE_CHAR +
                             "\s+stride\s+(.+)" + NEW_LINE_CHAR +
                             "\s+outputPaddingLower\s+(.+)" + NEW_LINE_CHAR +
                             "\s+outputPaddingUpper\s+(.+)" + NEW_LINE_CHAR +
                             "\s+outputFieldShape\s+(.+)$",
                             re.MULTILINE)

conv_params_info_fields_names = ['inputType', 'outputType', 'batchSize', 'numConvGroups', 'inputFieldShape',
                             'kernelShape', 'inputChannelsPerConvGroup', 'outputChannelsPerConvGroup',
                             'inputTruncationLower', 'inputTruncationUpper', 'inputDilation', 'inputPaddingLower',
                             'inputPaddingUpper', 'flipInput', 'kernelTruncationLower', 'kernelTruncationUpper',
                             'kernelDilation', 'kernelPaddingLower', 'kernelPaddingUpper', 'flipKernel',
                             'outputTruncationLower', 'outputTruncationUpper', 'stride', 'outputPaddingLower',
                             'outputPaddingUpper', 'outputFieldShape']
ConvParamsInfoFields = collections.namedtuple(
    'ConvParamsInfoFields', conv_params_info_fields_names
)


# Execution (Profile) capture
re_execution_profile = re.compile(r"^\s+([a-zA-Z]+): (.+)" + NEW_LINE_CHAR +
                               "\s+Cycles:\s+IPU 0:\s+(.+) \(.+" + NEW_LINE_CHAR +
                               "\s+Active Tiles:"
                              , re.MULTILINE)

execution_fields_names = ['weightsTransposeSeq', 'transformPreSerialSeq', 'transformPre0',
                          'transformPre1', 'transformPost1', 'transformPost0', 'transformPostSerialSeq']
execution_step_cycles = ['DoExchange', 'OnTileExecute']

# Planner to Profile diff
profile_2_planner_ratio_fields_names = ['Exch_PrByPl', 'OnTile_PrByPl']
Profile2PlannerRatioFields = collections.namedtuple(
    'Profile2PlannerRatioFields', profile_2_planner_ratio_fields_names
)


# Benchmark names and params capture
# 407: Test command: /usr/bin/python3 "/scratch/oleksiik/poplar/poplibs/tools/bench.py" "--name" "resnet50_tr_bs1_cnv_reduce" "--config" "default"
#                                      "--expected_csv" "/scratch/oleksiik/poplar/poplibs/tests/benchmark_results.csv" "/scratch/oleksiik/poplar/build_release/build/poplibs/tools/reduce_op"
#                                      "--shape=4,25088,8" "--dims=1" "--type=half" "--scale=1.0" "--update=false" "--operation=SQUARE_ADD" "--ignore-data" "--device-type=IpuModel"
# Labels: benchmarks python3
#   Test #407: IpuModel_default_resnet50_tr_bs1_cnv_reduce_benchmark
benchmarks_info = re.compile(r"^(\d+).+\"--name\"\s\"(\S+)\"\s\"--config\"\s\"([a-zA-z]+)\"\s\"--expected_csv\"\s\"(\S+)\"\s\"(\S+)\"\s(.+)$")


# -----------------------------------------------------------------------------
# Find if <--convolution-options> is already present and amend it
# -----------------------------------------------------------------------------
def amend_conv_options(test_cmd):
    #
    convOptions = r'"insertTransformsCycleCountProgs":true'
    try:
        co_index = test_cmd.index('--convolution-options') + 1
        test_cmd[co_index] = test_cmd[co_index][:-1] + ', ' + convOptions + '}'
    except ValueError:
        test_cmd.append('--convolution-options={' + convOptions + '}')
    return test_cmd


# -----------------------------------------------------------------------------
# Find <--device-type> and amend with desired target
# -----------------------------------------------------------------------------
def amend_device_type(test_cmd, device_type):
    try:
        co_index = test_cmd.index('--device-type') + 1
        test_cmd[co_index] = device_type
    except ValueError:
        test_cmd.append(f'--device-type={device_type}')
    return test_cmd


# -----------------------------------------------------------------------------
# Find if <--use-create-input> is already present and amend and amend it to 0
# -----------------------------------------------------------------------------
def amend_create_input(test_cmd):
    state = '0'
    try: # Reset create input to False
        index = test_cmd.index('--use-create-input') + 1
        test_cmd[index] = state
    except ValueError:
        test_cmd.append(f'--use-create-input={state}')
    return test_cmd


# -----------------------------------------------------------------------------
# Benchmarks collector
# -----------------------------------------------------------------------------
def update_test_cmd(test_cmd, device_type):

    test_cmd = amend_conv_options(test_cmd)
    test_cmd = amend_device_type(test_cmd, device_type)
    test_cmd.append('--profile')
    test_cmd.append('--enable-convolution-reuse=false')
    test_cmd = amend_create_input(test_cmd)

    return test_cmd


# -----------------------------------------------------------------------------
# Benchmarks collector
# -----------------------------------------------------------------------------
def amend_conv_options(test_cmd):
    # Find if <--convolution-options> is already present and amend it
    try:
        co_index = test_cmd.index('--convolution-options') + 1
        test_cmd[co_index] = test_cmd[co_index][:-1] + ', "insertTransformsCycleCountProgs":true, "partialsType":"float"}'
    except ValueError:
        test_cmd.append(r'--convolution-options={"insertTransformsCycleCountProgs":true}')
    return test_cmd


def amend_device_type(test_cmd, device_type):
    # Find <--device-type> and amend with desired target
    try:
        co_index = test_cmd.index('--device-type') + 1
        test_cmd[co_index] = device_type
    except ValueError:
        test_cmd.append(f'--device-type={device_type}')
    return test_cmd


def amend_create_input(test_cmd):
    state = '0'
    try: # Reset create input to False
        index = test_cmd.index('--use-create-input') + 1
        test_cmd[index] = state
    except ValueError:
        test_cmd.append(r'--use-create-input=' + state)
    return test_cmd


def get_list_of_tests(cmd, device_type):
    nproc = f'-j{os.cpu_count()}'
    cmd.append(nproc)
    tests_dict = {}
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        for line in proc.stdout:
            dline = line.decode('utf-8')
            match = benchmarks_info.match(dline)
            if match:
                tests_group = match.group(5).split("/")[-1]
                if tests_group in group_to_analyse:
                    test_cmd = match.group(6).replace('"','').replace('=',' ').split(' ')
                    # Change batch size to 2 (make the data collection a bit faster)
                    test_cmd[test_cmd.index('--batch-size') + 1] = '2'
                    test_cmd.insert(0, match.group(5))
                    test_cmd = update_test_cmd(test_cmd, device_type)
                    test_cmd.append('--preplan=0')
                    tests_dict[match.group(2)] = test_cmd

    return tests_dict


def collect_standard_benchmarks(names, device_type):
    cmd = [r'./test.sh', 'poplibs', '-L', 'benchmarks', '-N', '-V'] # dry run
    all_benchmarks = get_list_of_tests(cmd, device_type)

    tests_dict = {}
    if names:
        list_of_names = names.strip(' ').split(',')
        benchmarks_names = all_benchmarks.keys()
        for name in list_of_names:
            if name in benchmarks_names:
                tests_dict[name] = all_benchmarks[name]
        if len(tests_dict) == 0:
            logging.error('No tests found for the given names - %s', names)
            sys.exit(1)
    else:
        tests_dict = all_benchmarks

    return tests_dict


# -----------------------------------------------------------------------------
# Get benchmarks from file
# -----------------------------------------------------------------------------
def parse_test_file(file_path, device_type):
    tests_dict = {}
    tests_match = re.compile(r'^name:(.+),\scommand:(.+)$')
    # with open(file_path, mode='r', encoding='utf-8') as f:
    with open(file_path, mode='r') as f:
        for line in f:
            match = tests_match.match(line)
            if match:
                test_cmd = match.group(2).replace('=','\' \'').split('\' \'')
                # This is a bit of hack to  get rid of leading apostrophy
                test_cmd[0]  = test_cmd[0][1:]
                test_cmd = update_test_cmd(test_cmd, device_type)
                for phase in phases_dict:
                    tests_dict[f'random{match.group(1)}_{phase}'] =\
                        test_cmd + [f'--single-phase={phase}']

    return tests_dict


# -----------------------------------------------------------------------------
# Benchmarks permutation
# -----------------------------------------------------------------------------
def transform_constraints(phase, so, ed, ocfd, ccgf):
    constraints = []
    if so:
        constraints.append(f'"swapOperands":{so}')
    if ed:
        constraints.append(f'"expandDims":{ed}')
    if ocfd:
        constraints.append(f'"outChanFlattenDims":{ocfd}')
    if ccgf:
        constraints.append(f'"combineConvGroupsFactor":[{ccgf}]')

    phase_constraints = ''
    if constraints:
        phase_constraints = f'--{phase.lower()}-plan-constraints='
        phase_constraints += r'{"0": {"transform": {'
        for c in constraints[:-1]:
            phase_constraints += c + ','
        phase_constraints += constraints[-1]
        phase_constraints += r'}}}'

    return phase_constraints


def powerset(dims):
    x = len(dims)
    for i in range(1 << x):
        combo = [dims[j] for j in range(x) if i & (1 << j)]
        yield combo[::-1]


def add_permutations(standard_test):
    swap_operands = ['true', 'false']
    combine_conv_groups_factor = ['1', '2', '4', '8']

    all_tests_dic = {}
    for name, cmd in standard_test.items():
        num_dimensions = len(cmd[2].split(','))

        # Redefinitions below are useful for debug purposes
        # so that one can easily eliminate one or more dimensions

        # Example of debug params
        # p_list = phases_dict
        # so_list = ['false']
        # ed_list =  ['[1,0]']
        # ocfd_list = ['[]']
        # ccgf_list = ['1']

        # Release params
        p_list = phases_dict
        so_list = swap_operands
        ed_list =  powerset(range(num_dimensions))
        ocfd_list = powerset(range(num_dimensions))
        ccgf_list = combine_conv_groups_factor

        parameters = itertools.product(p_list, so_list, ed_list, ocfd_list, ccgf_list)

        for phase, so, ed, ocfd, ccgf in parameters:
            edStr = str(ed).replace(', ', '_')
            ocfdStr = str(ocfd).replace(', ', '_')
            final_test_name = f'{name}_{so}_{edStr}_{ocfdStr}_[{ccgf}]_{phase}'

            all_tests_dic[final_test_name] = standard_test[name] +\
                    [f'--single-phase={phase}'] +\
                    [transform_constraints(phase, so, ed, ocfd, ccgf)]

    return all_tests_dic


# -----------------------------------------------------------------------------
# Run benchmarks
# -----------------------------------------------------------------------------
class BenchmarksBar(Bar):
    suffix = '%(index)d/%(max)d - %(elapsed)ds'


def open_proc(file_path, cmd):
    # Skip run if log file already exists
    if not os.path.exists(file_path):
        # Prevent colour codes as they cause problems with the regex matches
        env = os.environ.copy()
        env["CLICOLOR_FORCE"] = "0"
        env["CLICOLOR"] = "0"
        with open(file_path, mode="w", encoding='utf-8') as log_file:
            subprocess.call(cmd, stdout=log_file, stderr=log_file, env=env)

def run_tests(tests_dict, output_path, runtime_timeout):
    os.environ["POPLIBS_LOG_LEVEL"] = "DEBUG"
    nproc = os.cpu_count()

    nr_of_bench = len(tests_dict)
    procs = {}
    killed_tests = []

    logging.info('Starting %d benchmarks (Runtime timeout for a one test is %dsecs).', nr_of_bench, runtime_timeout)
    logging.fatal('Meanwhile go and grab some brew!')

    nproc = int(nproc / 2) # gives same execution results as nproc
    progress_bar =  BenchmarksBar('Running', max=nr_of_bench)
    pool = Pool(processes = nproc)

    for name, cmd in tests_dict.items():
        file_path = os.path.join(output_path, name + LOG_FILE_EXT)
        procs[name] = pool.apply_async(open_proc, args=(file_path, cmd))

    for name, process in procs.items():
        progress_bar.next() # dry run to display progress bar
        try:
            process.get(runtime_timeout)
        except TimeoutError:
            killed_tests.append(name)

    pool.terminate()
    pool.join()

    # New line required after progress bar has finished
    logging.info('')
    if killed_tests:
        logging.info('--------------------------------------------------------------------------------')
        logging.info('These tests were removed due to timeout %dsec:', runtime_timeout)
        for k in killed_tests:
            logging.info(k)
        logging.info('--------------------------------------------------------------------------------')


# -----------------------------------------------------------------------------
# Capture information from profile output
# -----------------------------------------------------------------------------
def capture_exec_cycles(log_output):
    # Current format:
    #    debugPrefix + "/timeBeforeCS_" + id
    #    debugPrefix + "/timeAfterCS_" + id
    exec_start_capture = re.compile(r'^(.+)/timeBeforeCS_(\d+)$')
    exec_end_capture = re.compile(r'^(.+)/timeAfterCS_(\d+)$')
    ExecStepsInfo = collections.namedtuple('ExecStepsInfo', ['location', 'cs', 'cycles'])

    valid = False
    execution_dict = list(0 for x in range(2 * len(execution_fields_names)))

    compute_set_id = 0
    execution_step_name = ''
    for step in log_output:
        exec_steps_named = ExecStepsInfo._make(x for x in step)
        start_match = exec_start_capture.match(exec_steps_named.cs)
        if start_match:
            cs_name = start_match.group(1).split('/')[-1]
            if cs_name in execution_fields_names:
                execution_step_name = cs_name
                compute_set_id = start_match.group(2)
                continue
        end_match = exec_end_capture.match(exec_steps_named.cs)
        if end_match:
            cs_name = end_match.group(1).split('/')[-1]
            if cs_name in execution_fields_names:
                if compute_set_id != end_match.group(2):
                    logging.error('Expected timeAfterCS_%s. Got timeAfterCS_%s', compute_set_id, end_match.group(2))
                    raise 'Couldn\'t find timeAfterCS_'
                compute_set_id = 0 # reset
                execution_step_name = ''
                # At least  one valid pair of cycle count found
                valid = True
                continue

        if compute_set_id != 0 and exec_steps_named.location in execution_step_cycles:
            index = execution_fields_names.index(execution_step_name)
            next_step_cycles = int(exec_steps_named.cycles.replace(',',''))
            offset = execution_step_cycles.index(exec_steps_named.location)
            index = index * len(execution_step_cycles) + offset
            execution_dict[index] += next_step_cycles

    return valid, tuple(execution_dict)


# -----------------------------------------------------------------------------
# Calculate differences between planner and profile
# -----------------------------------------------------------------------------
def get_diffs(planner_cycles,  profile_cycles):
    try:
        diff = '{:.2f}'.format(abs(float(profile_cycles) / float(planner_cycles)))
    except ZeroDivisionError:
        diff = '0'

    return diff


def get_execution_offset(field_name, step_name):
    if field_name not in execution_fields_names:
        logging.error('Requested filed (%s) doesn\'t exist in %s', field_name, str(execution_fields_names))
        sys.exit(1)
    if step_name not in execution_step_cycles:
        logging.error('Requested step (%s) doesn\'t exist in %s', step_name, str(execution_step_cycles))
        sys.exit(1)

    transforms_offset = execution_fields_names.index(field_name) * len(execution_step_cycles)
    return transforms_offset + execution_step_cycles.index(step_name)


def calculate_diffs(planner_data, profile_data):
    # At the moment only do diffs  for transfroms but can be exteded to  any other fields
    te_index = get_execution_offset('transformPre0', 'DoExchange')
    tot_index = get_execution_offset('transformPre0', 'OnTileExecute')

    #TODO: Need to take into account broadcastInputs and broadcastExchange cycles (transform parts)
    transforms_exchange_diff = get_diffs(planner_data.transformsExchangeCycles, profile_data[te_index])
    transforms_on_tile_diff = get_diffs(planner_data.transformsCopyCycles, profile_data[tot_index])

    return Profile2PlannerRatioFields._make(x for x in [transforms_exchange_diff, transforms_on_tile_diff])


# -----------------------------------------------------------------------------
# Capture information from logs
# -----------------------------------------------------------------------------
def capture_logs_info(tests_dict, output_path, remove_log_files):
    all_results = {}
    for name in tests_dict.keys():
        phase_dict = {}
        filepath = os.path.join(output_path, name + LOG_FILE_EXT)
        with open(filepath, mode='r', encoding='utf-8') as f:
            all_of_it = f.read()
        if remove_log_files is True:
            os.remove(filepath)

        # NOTE: When adding new match need to update NUMBER_OF_MATCHES define to allow
        #       CI test successfully validate number of matches for each test
        match_planner_info = re_planner_info.findall(all_of_it)
        match_transform_info = re_transform_info.findall(all_of_it)
        match_partition_info = re_partition_info.findall(all_of_it)
        match_plan_info = re_plan_info.findall(all_of_it)
        match_conv_params_info = re_conv_params_info.findall(all_of_it)
        match_execution_profile = re_execution_profile.findall(all_of_it)

        # Check if test was successful
        if all_of_it.find('terminate called') != -1:
            logging.debug('%s - was terminated. Mission aborted... (No record added into a  file)', name)
            continue

        # Find phase related info and record index to get params and transform infos
        index = -1
        phase = name.split('_')[-1]
        try:
            for p_info in match_planner_info:
                index += 1
                if p_info[5].lower() == phase:
                    planner_data = PlannerInfoFields._make(x for x in p_info)
                    phase_dict[name] = [planner_data]
                    break
        except IndexError:
            logging.debug('%s - No best plan info. Most likely test had a timeout', name)
            continue

        try:
            phase_dict[name].append(TransformInfoFields._make(x for x in match_transform_info[index]))
        except IndexError:
            logging.debug('%s - No transforms info. Possible incorrect test params...', name)
            continue

        try:
            phase_dict[name].append(PartitionInfoFields._make(x for x in match_partition_info[index]))
        except IndexError:
            logging.debug('%s - No partition info. Planner failed...', name)
            continue

        try:
            plan_info = PlanInfoFields._make(x for x in match_plan_info[index])
        except IndexError:
            logging.debug('%s - No plan details. Planner failed...', name)
            continue

        try:
            conv_params = ConvParamsInfoFields._make(x for x in match_conv_params_info[index])
            phase_dict[name].append(conv_params)
        except IndexError:
            logging.debug('%s - No convolution params info. Planner failed...', name)
            continue

        # Get transforms execution cycles from profile output
        if match_execution_profile:
            valid, profile_data = capture_exec_cycles(match_execution_profile)
            if valid:
                phase_dict[name].append(profile_data)
            else:
                logging.debug('%s - No transfroms markers found in profile output. Make sure you use next option:\
                               --convolution-options={{\"insertTransformsCycleCountProgs\":true}}', name)
                continue
        else:
            logging.debug('%s - No profile info. Planner failed...', name)
            continue

        # Generate planner vs profiles diffs
        phase_dict[name].append(calculate_diffs(planner_data, profile_data))

        # Debug printouts
        printout_step = False
        if printout_step is True:
            logging.error("Execution:")
            for profile_step in match_execution_profile:
                logging.error(profile_step)

        if printout_step is True:
            logging.error("Planner info:")
            for phase, phase_info in phase_dict.items():
                logging.error('%s:', phase)
                for info in phase_info:
                    logging.error('%s', info)

        all_results.update(phase_dict)

    return all_results


# -----------------------------------------------------------------------------
# Dump results
# -----------------------------------------------------------------------------
def dump_results(all_results, file_path):

    header1 = ['Info Groups']
    header2 = ['Info Fields']

    header1 += ['plannerInfo'] * len(planner_info_fields_names)
    header2 += planner_info_fields_names

    header1 += ['transform0'] * len(transform_info_fields_names)
    header2 += transform_info_fields_names

    header1 += ['Partition0'] * len(partition_info_fields_names)
    header2 += partition_info_fields_names

    header1 += ['convParams'] * len(conv_params_info_fields_names)
    header2 += conv_params_info_fields_names

    for e_name in execution_fields_names:
        for e_step in execution_step_cycles:
            header1.append('profileCycles')
            header2.append(e_name + '_' + e_step)

    for p_name in profile_2_planner_ratio_fields_names:
        header1.append('Diff ratio')
        header2.append(p_name)

    if len(header1) != len(header2):
        raise 'Headers lengths don\'t match'

    with open(file_path, 'w') as results_file:
        results_writer = csv.writer(results_file,
                                delimiter=',',
                                lineterminator=os.linesep)

        results_writer.writerow(header2)
        results_writer.writerow(header1)
        for key, infos in all_results.items():
            data = tuple([key])
            for info in infos:
                data += info
            results_writer.writerow(data)


# -----------------------------------------------------------------------------
# Generate CI tests
# -----------------------------------------------------------------------------
def generate_ci_tests(workspace, test_binary, device_type):
    ci_test_dict = {}

    if len(test_binary) == 0:
        logging.error('--test-binary can\'t be empty. See help for more details')
        sys.exit(1)
    elif os.path.exists(test_binary) is False:
        logging.error('Invalid path to test-binary - <%s>', test_binary)
        sys.exit(1)

    for phase in phases_dict:
        test_name = f'ci_test_{phase}'
        ci_test_dict[test_name] = [test_binary,
                        '--field', '{7,7}', '--kernel-size', '3', '--padding', '1', '--input-channels', '1',
                        '--output-channels', '1', '--conv-groups', '64', '--batch-size', '2', '--bias', '0',
                        '--ignore-data', f'--device-type={device_type}',
                        '--profile', f'--single-phase={phase}', '--tiles-per-ipu=2',
                        '--convolution-options={"insertTransformsCycleCountProgs":true}',
                        '--preplan', '0',
                        transform_constraints(phase, 'true', '[0]', '[1]', '1')]

        # Remove ci-test logs to guarantee tests run
        file_to_remove = os.path.join(workspace, test_name + LOG_FILE_EXT)
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)

    return ci_test_dict


# -----------------------------------------------------------------------------
# Validate CI tests results
# -----------------------------------------------------------------------------
def validate_ci_test(all_results):
    # Successful capture shall have next tuples:
    #    - planner(0)
    #    - transform0(1)
    #    - partition0(2)
    #    - conv params(3)
    #    - profiles infos(4)
    #    - diff ratio(5)
    if len(all_results) != len(phases_dict):
        logging.error(' No valid test result for all phases')
        logging.error('Only following present: %s', all_results.keys())
        sys.exit(1)
    else:
        for k in all_results:
            number_of_matches = len(all_results[k])
            # See capture_logs_info method for an explanation on NUMBER_OF_MATCHES
            if number_of_matches != NUMBER_OF_MATCHES:
                logging.error(f'Not all test data found for {k} phase. Expected {NUMBER_OF_MATCHES} but got {number_of_matches}')
                sys.exit(1)

        # Check so specific transfroms cycles fields
        execution_data_index = len(all_results['ci_test_fwd']) - 2
        tp0_index = get_execution_offset('transformPre0', 'OnTileExecute')
        tp1_index = get_execution_offset('transformPre1', 'OnTileExecute')

        # fwd pass check
        if all_results['ci_test_fwd'][execution_data_index][tp0_index] == 0:
            logging.error('FWD pass. transformPre0::OnTileExecute shall be greater than 0')
            sys.exit(1)

        # bwd pass check
        if all_results['ci_test_bwd'][execution_data_index][tp1_index] == 0:
            logging.error('BWD pass. transformPre1::OnTileExecute shall be greater than 0')
            sys.exit(1)

        # wu pass check
        if all_results['ci_test_wu'][execution_data_index][tp0_index] == 0:
            logging.error('WU pass. transformPre0::OnTileExecute shall be greater than 0')
            sys.exit(1)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Collect convolution transforms data')
    parser.add_argument('--workspace', default=os.getcwd(), help='Absolute path to where to store tests log files and report. Default path is a current folder')
    parser.add_argument('--results-file', default='transforms.csv', help='Filename to store results. Default: transfroms.csv')
    parser.add_argument('--remove-files', default=False, action='store_true', help='If specified - log capture files WILL BE removed after being processed')
    parser.add_argument('--test-names', default='', help='Benchmarks names separated by commas')
    parser.add_argument('--ci-test', default=False, action='store_true', help='Uses predefined test to assest regex parsers')
    parser.add_argument('--execution-timeout', type=int, default=120, help='Defines timeout for a single test execution')
    parser.add_argument('--test-file', default='', help='Shall contain list of '
         'single_conv_layers tests. File format shall be next: name:<test_name>, command:<test executable>. One command per line.')
    parser.add_argument('--test-binary', default='', help='Provides a path to the single_conv_layer tool')
    parser.add_argument("--device-type", choices=("Hw"), default="Hw", help="Only Hw is supported")
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

    log_file_path = os.path.join(args.workspace, 'transforms_tool.log')
    file_logger = logging.FileHandler(log_file_path, mode='w')
    file_logger.setLevel(logging.DEBUG)

    logging.basicConfig(
        level=os.environ.get("TRANSFORMS_TOOL_LOG_LEVEL", default='DEBUG'),
        format="[%(levelname)s] %(asctime)s: %(message)s",
        handlers=[
            file_logger,
            stdout_logger
        ]
    )

    report_file = os.path.join(args.workspace, args.results_file)

    # Get
    if args.ci_test is True:
        logging.info('Starting a test run')
        tests_dict = generate_ci_tests(args.workspace, args.test_binary, args.device_type)

    elif os.path.exists(args.test_file):
        logging.info('Collecting benchmarks from a file')
        tests_dict = parse_test_file(args.test_file, args.device_type)

    else:
        logging.info('Collecting existent benchmarks')
        tests_dict = collect_standard_benchmarks(args.test_names, args.device_type)

        # Speaks for itself
        logging.info('Generating plan constarints for the given benchmarks')
        tests_dict = add_permutations(tests_dict)

    # Run benchmarks
    logging.info('Processing...')
    run_tests(tests_dict, args.workspace, args.execution_timeout)

    # Capture logs
    logging.info('Capturing results')
    all_results = capture_logs_info(tests_dict, args.workspace, args.remove_files)

    # Dump results
    if args.ci_test is True:
        validate_ci_test(all_results)
    else:
        logging.info('Storing results into a file(s)')
        dump_results(all_results, report_file)


# -----------------------------------------------------------------------------
# Super MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
