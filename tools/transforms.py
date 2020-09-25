#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import collections
import csv
import itertools
import logging
from multiprocessing import Pool, TimeoutError
from progress.bar import Bar
import re
import os
import subprocess
import sys


newline = r'(?:\n|\r\n?)'
extension = r'.conv.log'
groupToAnalyse = ['single_conv_layer']
phasesDict = {'fwd':0, 'bwd':1, 'wu':2}

# Monsto regex to capture planner output:
# 16:52:46.800 55378 PL [W] Found best plan using AMP: Cost{cycles=30748, memory=51072, tiles=1120}.
# 16:52:46.800 55378 PL [D]   for input {14,14}x(256x1x4), kernel {14,14}, output = {3,3}x(256x1x256), pass=TRAINING_WU
# 16:52:46.800 55378 PL [D]   breakdown of memory and cycle estimates:
# 16:52:46.800 55378 PL [D]    - total parallel split: 1120
# 16:52:46.800 55378 PL [D]    - total serial split: 1
# 16:52:46.800 55378 PL [D]    - rearrangement before slice: 0 cycles, 0 bytes (0 overhead, 0 per-loop iteration)
# 16:52:46.800 55378 PL [D]    - memsetZeroBeforeAddInPlace: 0 cycles, unknown bytes
# 16:52:46.800 55378 PL [D]    - dynamic slice: 0 cycles, unknown bytes
# 16:52:46.800 55378 PL [D]    - transform: 1421 cycles, 2868 bytes
# 16:52:46.800 55378 PL [D]    - exchange: 8233 cycles, n/a bytes. (Input 4178, Weight 359, Reduce 3696 + 0)
# 16:52:46.800 55378 PL [D]    - tile level transform: 0 cycles, 0 bytes
# 16:52:46.800 55378 PL [D]    - compute: 17338 cycles, 51072 bytes
# 16:52:46.800 55378 PL [D]    - reduction: 3756 cycles, 0 bytes
# 16:52:46.800 55378 PL [D]    - dynamic update: 0 cycles, unknown bytes
# 16:52:46.800 55378 PL [D]    - add in-place: 0 cycles, 0 bytes
# 16:52:46.800 55378 PL [D]    - cast: 0 cycles, 0 bytes
# 16:52:46.800 55378 PL [D]    - total: 30748 cycles, 51072 bytes
plannerInfo = re.compile(r"^.+Found best plan using ([A-Z]+): Cost\{cycles=(\d+), memory=(\d+), tiles=(\d+)\}." + newline +
                           ".+pass=([A-Z]+)_([A-Z]+)"  + newline +
                          ".+" + newline +
                          ".+total parallel split:\s(\d+)" + newline +
                          ".+total serial split:\s(\d+)" + newline +
                          ".+rearrangement before slice:\s(\d+).+" + newline +
                          ".+memsetZeroBeforeAddInPlace:\s(\d+).+" + newline +
                          ".+dynamic slice:\s(\d+).+" + newline +
                          ".+transform:\s(\d+)\scycles,\s(\d+).+" + newline +
                          ".+exchange:\s(\d+).+Input\s(\d+),\sWeight\s(\d+),\sReduce\s(\d+)\s\+\s(\d+).+" + newline +
                          ".+tile level transform:\s(\d+).+" + newline +
                          ".+compute:\s(\d+).+" + newline +
                          ".+reduction:\s(\d+).+" + newline +
                          ".+dynamic update:\s(\d+).+" + newline +
                          ".+add in-place:\s(\d+).+" + newline +
                          ".+cast:\s(\d+).+$",
                          re.MULTILINE)

plannerInfoFieldsNames = ['method', 'cost', 'memory', 'tiles', 'training', 'phase',
                 'parallelSplit', 'serialSplit', 'rearrangeBeforeSlice', 'memsetZeroBeforeAddInPlace',
                 'dynamicSlice', 'transformsCycles', 'transformsBytes', 'exchange', 'inputExchange',
                 'weightsExchange', 'reduceExchange', 'reduceExchangePlus', 'tileTransforms',
                 'compute', 'reduce', 'dynamicUpdate', 'addInPlace', 'cast']

plannerInfoFields = collections.namedtuple(
    'plannerInfoFields', plannerInfoFieldsNames
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
transformInfo = re.compile(r'^transform\s\#0' + newline +
                             '\s+Transform:\sextraFieldDims\s+(\d+)' + newline +
                             '\s+dilatePostConv\s+(.+)' + newline +
                             '\s+swapOperands\s+([a-z]+)' + newline +
                             '\s+expandDims\s+(.+)' + newline +
                             '\s+outChanFlattenDims\s+(.+)' + newline +
                             '\s+flattenDims\s+(.+)' + newline +
                             '\s+combineConvGroupsFactor\s+(\d+)$',
                            re.MULTILINE)

transformInfoFieldsNames = ['extraFieldDims', 'dilatePostConv', 'swapOperands',
    'expandDims', 'outChanFlattenDims', 'flattenDims', 'combineConvGroupsFactor']
transformInfoFileds = collections.namedtuple(
    'transformInfoFileds', transformInfoFieldsNames
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
partitionInfo = re.compile(r'^partition\s\#0' + newline +
                             '\s+Partition:\sfieldSplit\s+(.+)' + newline +
                             '\s+batchSplit\s+(\d+)' + newline +
                             '\s+outChanSplit.serial\s+(\d+)' + newline +
                             '\s+outChanSplit.parallel\s+(\d+)' + newline +
                             '\s+kernelSplit\s+(.+)' + newline +
                             '\s+inChanSplit.serial\s+(\d+)' + newline +
                             '\s+inChanSplit.parallel\s+(\d+)' + newline +
                             '\s+convGroupSplit\s+(\d+)' + newline +
                             '\s+fieldAxisGrainSize\s+(.+)' + newline +
                             '\s+inChanGrainSize\s+(\d+)' + newline +
                             '\s+outChanGrainSize\s+(\d+)$',
                            re.MULTILINE)

partitionInfoFieldsNames = ['fieldSplit', 'batchSplit', 'outChanSplit_serial',
    'outChanSplit_parallel', 'kernelSplit', 'inChanSplit_serial', 'inChanSplit_parallel',
    'convGroupSplit', 'fieldAxisGrainSize', 'inChanGrainSize', 'outChanGrainSize']
partitionInfoFields = collections.namedtuple(
    'partitionInfoFields', partitionInfoFieldsNames
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
convParamsInfo = re.compile(r"^\s+Params:" + newline +
                             "\s+inputType\s+([a-z]+)" + newline +
                             "\s+outputType\s+([a-z]+)" + newline +
                             "\s+batchSize\s+(\d+)" + newline +
                             "\s+numConvGroups\s+(\d+)" + newline +
                             "\s+inputFieldShape\s+(.+)" + newline +
                             "\s+kernelShape\s+(.+)" + newline +
                             "\s+inputChannelsPerConvGroup\s+(\d+)" + newline +
                             "\s+outputChannelsPerConvGroup\s+(\d+)" + newline +
                             "\s+inputTruncationLower\s+(.+)" + newline +
                             "\s+inputTruncationUpper\s+(.+)" + newline +
                             "\s+inputDilation\s+(.+)" + newline +
                             "\s+inputPaddingLower\s+(.+)" + newline +
                             "\s+inputPaddingUpper\s+(.+)" + newline +
                             "\s+flipInput\s+(.+)" + newline +
                             "\s+kernelTruncationLower\s+(.+)" + newline +
                             "\s+kernelTruncationUpper\s+(.+)" + newline +
                             "\s+kernelDilation\s+(.+)" + newline +
                             "\s+kernelPaddingLower\s+(.+)" + newline +
                             "\s+kernelPaddingUpper\s+(.+)" + newline +
                             "\s+flipKernel\s+(.+)" + newline +
                             "\s+outputTruncationLower\s+(.+)" + newline +
                             "\s+outputTruncationUpper\s+(.+)" + newline +
                             "\s+stride\s+(.+)" + newline +
                             "\s+outputPaddingLower\s+(.+)" + newline +
                             "\s+outputPaddingUpper\s+(.+)" + newline +
                             "\s+outputFieldShape\s+(.+)$",
                             re.MULTILINE)

convParamsInfoFieldsNames = ['inputType', 'outputType', 'batchSize', 'numConvGroups', 'inputFieldShape',
                             'kernelShape', 'inputChannelsPerConvGroup', 'outputChannelsPerConvGroup',
                             'inputTruncationLower', 'inputTruncationUpper', 'inputDilation', 'inputPaddingLower',
                             'inputPaddingUpper', 'flipInput', 'kernelTruncationLower', 'kernelTruncationUpper',
                             'kernelDilation', 'kernelPaddingLower', 'kernelPaddingUpper', 'flipKernel',
                             'outputTruncationLower', 'outputTruncationUpper', 'stride', 'outputPaddingLower',
                             'outputPaddingUpper', 'outputFieldShape']
convParamsInfoFields = collections.namedtuple(
    'convParamsInfoFields', convParamsInfoFieldsNames
)

# Execution (Profile) capture
executionProfile = re.compile(r"^\s+([a-zA-Z]+): (.+)" + newline +
                               "\s+Cycles:\s+(.+):.+" + newline +
                               "\s+Active Tiles:"
                              , re.MULTILINE)

executionFieldsNames = ['weightsTranspose', 'transformPreSerial', 'transformPre0', 'transformPost0', 'transformPostSerial']
executionStepCycles = ['DoExchange', 'OnTileExecute']

# Planner to Profile diff
planner2ProfileRatioFieldsNames = ['Exch_Pl2Pr', 'OnTile_Pl2Pr']
planner2ProfileRatioFields = collections.namedtuple(
    'planner2ProfileRatioFields', planner2ProfileRatioFieldsNames
)


# Benchmark names and params capture
# 407: Test command: /usr/bin/python3 "/scratch/oleksiik/poplar/poplibs/tools/bench.py" "--name" "resnet50_tr_bs1_cnv_reduce" "--config" "default"
#                                      "--expected_csv" "/scratch/oleksiik/poplar/poplibs/tests/benchmark_results.csv" "/scratch/oleksiik/poplar/build_release/build/poplibs/tools/reduce_op"
#                                      "--shape=4,25088,8" "--dims=1" "--type=half" "--scale=1.0" "--update=false" "--operation=SQUARE_ADD" "--ignore-data" "--use-unstable-format" "--device-type=IpuModel"
# Labels: benchmarks python3
#   Test #407: IpuModel_default_resnet50_tr_bs1_cnv_reduce_benchmark
benchmarksInfo = re.compile(r"^(\d+).+\"--name\"\s\"(\S+)\"\s\"--config\"\s\"([a-zA-z]+)\"\s\"--expected_csv\"\s\"(\S+)\"\s\"(\S+)\"\s(.+)$")


# -----------------------------------------------------------------------------
# Benchmarks collector
# -----------------------------------------------------------------------------
def get_list_of_tests(cmd):
    nproc = f'-j{os.cpu_count()}'
    cmd.append(nproc)
    testsDict = {}
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        for line in proc.stdout:
            dline = line.decode('utf-8')
            match = benchmarksInfo.match(dline)
            if match:
                testsGroup = match.group(5).split("/")[-1]
                if testsGroup in groupToAnalyse:
                    testCmd = match.group(6).replace('"','').split(" ")[:-1]
                    # Change batch size to 2 (make the data collection a bit faster)
                    testCmd[testCmd.index('--batch-size') + 1] = '2'
                    # NOTE: Ideally need to search through the matches in case a test already has  conv options and append it
                    testCmd.append(r'--convolution-options={"insertTransformsCycleCountProgs":true}')
                    testCmd.insert(0, match.group(5))
                    testCmd.append(r'--device-type=Hw')
                    testCmd.append(r'--profile')
                    testsDict[match.group(2)] = testCmd
    return testsDict


def collect_standard_benchmarks(names):
    cmd = [r'./test.sh', 'poplibs', '-L', 'benchmarks', '-N', '-V'] # dry run
    allBenchmarks = get_list_of_tests(cmd)

    testsDict = {}
    if names:
        listOfNames = names.strip(' ').split(',')
        benchmarksNames = allBenchmarks.keys()
        for name in listOfNames:
            if name in benchmarksNames:
                testsDict[name] = allBenchmarks[name]
        if len(testsDict) == 0:
            logging.error(f'No tests found for the given names - {names}')
            sys.exit(1)
    else:
       testsDict = allBenchmarks

    return testsDict


# -----------------------------------------------------------------------------
# Get benchmarks from file
# -----------------------------------------------------------------------------
def get_test_from_file(filepath):
    testsDict = {}
    testsMatch = re.compile(r'^name:(.+),\scommand:(.+)$')
    with open(filepath, mode='r', encoding='utf-8') as f:
        match = testsMatch.match(line.decode('utf-8'))
        if match:
            testsDict[match.group(1)] = match.group(2)

    return testsDict


# -----------------------------------------------------------------------------
# Benchmarks permutation
# -----------------------------------------------------------------------------
def transform_constraints(phase, so, ed, ocfd, ccgf):
    swapOperands = f'"swapOperands":{so}'
    expandDims = f'"expandDims":{ed}'
    outChanFlattenDims = f'"outChanFlattenDims":{ocfd}'
    combineConvGroupsFactor = f'"combineConvGroupsFactor":[{ccgf}]'
    phaseConstraintsPrefix = r'{"0": {"transform": {'
    phaseConstraintsSuffix = r'}}}'
    phaseConstraints =  phaseConstraintsPrefix +\
                        swapOperands + ',' +\
                        expandDims + ',' +\
                        outChanFlattenDims + ',' +\
                        combineConvGroupsFactor +\
                        phaseConstraintsSuffix
    return f'--{phase.lower()}-plan-constraints=' + phaseConstraints


def powerset(dims):
    x = len(dims)
    p = []
    for i in range(1 << x):
        combo = [dims[j] for j in range(x) if (i & (1 << j))]
        yield combo[::-1]


def add_permutations(standardTest):
    swapOperands = ['true', 'false']
    combineConvGroupsFactor = ['1', '2', '4', '8']

    finalDict = {}
    for name, cmd in standardTest.items():
        numDims = len(cmd[2].split(','))

        # Redefinitions below are useful for debug purposes
        # so that one can easily eliminate one or more dimensions

        # Example of debug params
        # p_list = phasesDict
        # so_list = ['false']
        # ed_list =  ['[]']
        # ocfd_list = ['[]']
        # ccgf_list = ['1']

        # Release params
        p_list = phasesDict
        so_list = swapOperands
        ed_list =  powerset(range(numDims))
        ocfd_list = powerset(range(numDims))
        ccgf_list = combineConvGroupsFactor

        parameters = itertools.product(p_list, so_list, ed_list, ocfd_list, ccgf_list)

        for p, so, ed, ocfd, ccgf in parameters:
            edStr = str(ed).replace(', ', '_')
            ocfdStr = str(ocfd).replace(', ', '_')
            pname = f'{name}_{so}_{edStr}_{ocfdStr}_[{ccgf}]_{p}'
            finalDict[pname] = standardTest[name] +\
                    [f'--single-phase={p}'] +\
                    [transform_constraints(p, so, ed, ocfd, ccgf)]

    return finalDict


# -----------------------------------------------------------------------------
# Run benchmarks
# -----------------------------------------------------------------------------
class BenchmarksBar(Bar):
    suffix = '%(index)d/%(max)d - %(elapsed)ds'

def open_proc(filePath, cmd):
    # Skip run if log file already exists
    if not os.path.exists(filePath):
        with open(filePath, mode="w", encoding='utf-8') as logFile:
            subprocess.call(cmd, stdout=logFile, stderr=logFile)

def run_tests(testsDict, outputPath, timeOut):
    os.environ["POPLIBS_LOG_LEVEL"] = "DEBUG"
    nproc = os.cpu_count()

    numOfBench = len(testsDict)
    procs = {}
    killedTests = []

    logging.info(f'Starting {numOfBench} benchmarks (Runtime timeout for a one test is {timeOut}secs).')
    logging.fatal('Meanwhile go and grab some brew!')

    nproc = int(nproc / 2) # gives same execution results as nproc
    progressBar =  BenchmarksBar('Running', max=numOfBench)
    pool = Pool(processes = nproc)

    for name, cmd in testsDict.items():
        filePath = os.path.join(outputPath, name + extension)
        procs[name] = pool.apply_async(open_proc, args=(filePath, cmd))

    for name, process in procs.items():
        progressBar.next() # dry run to display progress bar
        try:
            process.get(timeOut)
        except TimeoutError:
            killedTests.append(name)

    pool.terminate()
    pool.join()

    # New line required after progress bar has finished
    if killedTests:
        logging.info('--------------------------------------------------------------------------------')
        logging.info(f'These tests were removed due to timeout {timeOut}sec:')
        for k in killedTests:
            logging.info(k)
        logging.info('--------------------------------------------------------------------------------')


# -----------------------------------------------------------------------------
# Capture information from profile output
# -----------------------------------------------------------------------------
def capture_exec_cycles(logOutput):
#   //  - weightsTranspose (optional)
#   //  - transformPreSerial
#   //  - repeat(loopCount)
#   //    - slice
#   //    - transformPre[level=0]
#   //    - transformPre[level=1]
#   //    - convolve[level=1]
#   //    - transformPost[level=1]
#   //    - reduce[level=0]
#   //    - transformPost[level=0]
#   //    - update/addInPlace
#   //    - loopPost
#   //  - transformPostSerial
#   //  - finalizeProg

    execStartCapture = re.compile(r'^([a-zA-Z]+\d)/timeBeforeCS_(\d+)$')
    execEndCapture = re.compile(r'^([a-zA-Z]+\d)/timeAfterCS_(\d+)$')
    execStepsInfo = collections.namedtuple('execStepsInfo', ['location', 'cs', 'cycles'])

    valid = False
    execDB = list(0 for x in range(2 * len(executionFieldsNames)))

    csId = 0
    stepName = ''
    for step in logOutput:
        execStepsNamed = execStepsInfo._make(x for x in step)
        startMatch = execStartCapture.match(execStepsNamed.cs)
        if startMatch and startMatch.group(1) in executionFieldsNames:
            stepName = startMatch.group(1)
            csId = startMatch.group(2)
            continue
        endMatch = execEndCapture.match(execStepsNamed.cs)
        if endMatch and endMatch.group(1) in executionFieldsNames:
            if csId != endMatch.group(2):
                logging.debug(f'Expected timeAfterCS_{csId}. Got timeAfterCS_{endMatch.group(2)}')
                raise 'Couldn\'t find timeAfterCS_'
            csId = 0 # reset
            stepName = ''
            # At least  one valid pair of cycle count found
            valid = True
            continue

        if csId != 0 and execStepsNamed.location in executionStepCycles:
            index = executionFieldsNames.index(stepName)
            nextStepCycles = int(execStepsNamed.cycles.replace(',',''))
            offset = executionStepCycles.index(execStepsNamed.location)
            index = index * len(executionStepCycles) + offset
            execDB[index] += nextStepCycles

    return valid, tuple(execDB)


# -----------------------------------------------------------------------------
# Calculate differences between planner and profile
# -----------------------------------------------------------------------------
def get_diffs(plannerCycles,  profileCycles):
    try:
        diff = '{:.2f}'.format(abs(float(plannerCycles) / float(profileCycles)))
    except ZeroDivisionError:
        diff = '0'

    return diff


def calculate_diffs(plannerData, profileData):

    # At the moment only do diffs  for transfroms but can be exteded to  any other fields
    transformsOffset = executionFieldsNames.index('transformPre0') * len(executionStepCycles)
    transformsExchange = transformsOffset + executionStepCycles.index('DoExchange')
    transformsOnTile = transformsOffset + executionStepCycles.index('OnTileExecute')

    transformsExchangeDiff = get_diffs(plannerData.exchange, profileData[transformsExchange])
    transformsOnTileDiff = get_diffs(plannerData.transformsCycles, profileData[transformsOnTile])

    return planner2ProfileRatioFields._make(x for x in [transformsExchangeDiff, transformsOnTileDiff])


# -----------------------------------------------------------------------------
# Capture information from logs
# -----------------------------------------------------------------------------
def capture_logs_info(testsDict, outputPath, removeFiles):
    megaDB = {}
    for name in testsDict.keys():
        phaseDB = {}
        fileName = os.path.join(outputPath, name + extension)
        with open(fileName, mode='r', encoding='utf-8') as f:
            all_of_it = f.read()
        if removeFiles is True:
            os.remove(fileName)

        plannerInfoMatch = plannerInfo.findall(all_of_it)
        transformInfoMatch = transformInfo.findall(all_of_it)
        partitionInfoMatch = partitionInfo.findall(all_of_it)
        convParamsInfoMatch = convParamsInfo.findall(all_of_it)
        executionProfileMatch = executionProfile.findall(all_of_it)

        # Check if test was successful
        if all_of_it.find('terminate called') != -1:
            logging.debug(f'{name} - was terminated. Mission aborted... (No record added into a  file)')
            continue

        # Find phase related info and record index to get params and transform infos
        index = -1
        phase = name.split('_')[-1]
        try:
            for pi in plannerInfoMatch:
                index += 1
                if pi[5].lower() == phase:
                    plannerData = plannerInfoFields._make(x for x in pi)
                    phaseDB[name] = [plannerData]
                    break
        except IndexError:
            logging.debug(f'{name} - No best plan info. Most likely test had a timeout')
            continue

        # Make sure we got plan info for a       correct phase
        if index != phasesDict[phase]:
            logging.debug(f'{name} - No best plan info for a {phase} pass')
            continue

        try:
            phaseDB[name].append(transformInfoFileds._make(x for x in transformInfoMatch[index]))
        except IndexError:
            logging.debug(f'{name} - No transforms info. Possible incorrect test params...')
            continue

        try:
            phaseDB[name].append(partitionInfoFields._make(x for x in partitionInfoMatch[index]))
        except IndexError:
            logging.debug(f'{name} - No partition info. Planner failed...')
            continue

        try:
            phaseDB[name].append(convParamsInfoFields._make(x for x in convParamsInfoMatch[index]))
        except IndexError:
            logging.debug(f'{name} - No convolution params info. Planner failed...')
            continue

        # Get transforms execution cycles from profile output
        if executionProfileMatch:
            valid, profileData = capture_exec_cycles(executionProfileMatch)
            if valid:
                phaseDB[name].append(profileData)
            else:
                logging.debug(f'{name} - No transfroms markers found in profile output. Make sure you use next option: --convolution-options={{\"insertTransformsCycleCountProgs\":true}}')
                continue
        else:
            logging.debug(f'{name} - No profile info. Planner failed...')
            continue

        # Generate planner vs profiles diffs
        phaseDB[name].append(calculate_diffs(plannerData, profileData))

        # Debug printouts
        if False:
            logging.error("Execution:")
            for p in executionProfileMatch:
                logging.error(p)

        if False:
            logging.error("Planner info:")
            for phase, params in phaseDB.items():
                logging.error(f'{phase}:')
                for p in params:
                    logging.error(f'{p}')

        megaDB.update(phaseDB)

    return megaDB


# -----------------------------------------------------------------------------
# Dump results
# -----------------------------------------------------------------------------
def dump_results(megaDB, filePath):

    header1 = ['Info Groups']
    header2 = ['Info Fields']

    for p in plannerInfoFieldsNames:
        header1.append('plannerInfo')
    header2 += plannerInfoFieldsNames

    for t in transformInfoFieldsNames:
        header1.append('transform0')
    header2 += transformInfoFieldsNames

    for t in partitionInfoFieldsNames:
        header1.append('Partition0')
    header2 += partitionInfoFieldsNames

    for c in convParamsInfoFieldsNames:
        header1.append('convParams')
    header2 += convParamsInfoFieldsNames

    for e in executionFieldsNames:
        for s in executionStepCycles:
            header1.append('profileCycles')
            header2.append(e + '_' + s)

    for p in planner2ProfileRatioFieldsNames:
        header1.append('Diff ratio')
        header2.append(p)

    if  len(header1) != len(header2):
        raise ('Headers lengths don\'t match')

    with open(filePath, 'w') as resultsFile:
        resultsWriter = csv.writer(resultsFile,
                                delimiter=',',
                                lineterminator=os.linesep)

        resultsWriter.writerow(header2)
        resultsWriter.writerow(header1)
        for key, infos in megaDB.items():
            data = tuple([key])
            for info in infos:
                data += info
            resultsWriter.writerow(data)


# -----------------------------------------------------------------------------
# Generate CI tests
# -----------------------------------------------------------------------------
def generate_ci_tests(workspace, test_binary):
    ci_test_dict = {}
    for p in phasesDict:
        test_name = f'ci_test_{p}'
        ci_test_dict[test_name] = [test_binary,
                        '--field', '{7,7}', '--kernel-size', '3', '--padding', '1', '--input-channels', '1',
                        '--output-channels', '1', '--conv-groups', '64', '--batch-size', '2', '--bias', '0',
                        '--ignore-data', '--use-unstable-format', '--device-type=Hw', '--profile',
                        f'--single-phase={p}', '--tiles-per-ipu=2',
                        '--convolution-options={"insertTransformsCycleCountProgs":true}',
                        transform_constraints(p, 'true', '[]', '[]', '1')]

        # Remove ci-test logs to guarantee tests run
        file_to_remove = os.path.join(workspace, test_name + extension)
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)

    return ci_test_dict


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
    parser.add_argument('--execution-timeout', type=int, default=45, help='Defines timeout for a single test execution')
    parser.add_argument('--test-file', default='', help='Shall contain list of '
         'single_conv_layers tests. File format shall be next: name:<test_name>, command:<test executable>. One command per line.')
    parser.add_argument('--test-binary', default='', help='')
    args = parser.parse_args()


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

    reportFile = os.path.join(args.workspace, args.results_file)

    # Make sure workspace folder  exists
    if not os.path.exists(args.workspace):
        os.makedirs(args.workspace)

    # Get
    if args.ci_test is True:
        logging.info('Starting a test run')
        testsDict = generate_ci_tests(args.workspace, args.test_binary)

    elif os.path.exists(args.test_file):
        logging.info('Collecting benchmarks from a file')
        testsDict = parse_test_file(args.test_file)

    else:
        logging.info('Collecting existent benchmarks')
        testsDict = collect_standard_benchmarks(args.test_names)

        # Speaks for itself
        logging.info('Generating plan constarints for the given benchmarks')
        testsDict = add_permutations(testsDict)

    # Run benchmarks
    logging.info('Processing...')
    run_tests(testsDict, args.workspace, args.execution_timeout)

    # Capture logs
    logging.info('Capturing results')
    megaDB = capture_logs_info(testsDict, args.workspace, args.remove_files)

    # Dump results
    if args.ci_test is True:
        # Successful capture shall have - planner, transform0, partition0, conv params, profiles infos and a diff ratio
        if len(megaDB) != len(phasesDict):
            logging.error(' No valid test result for all phases')
            logging.error(f'Only following present: {megaDB.keys()}')
            sys.exit(1)
        else:
            for k in megaDB.keys():
                if len(megaDB[k]) != 6:
                    logging.error(f'Not all test data found for {k} phase')
                    sys.exit(1)
    else:
        logging.info('Storing results into a file(s)')
        dump_results(megaDB, reportFile)


# -----------------------------------------------------------------------------
# Super MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
