#!/usr/bin/env python

import os, sys
import getopt

test_rnn_dict = [
  {"Name":"1760x16x50",       "Output-Size":1760,      "Batch-Size":16,         "Time-Steps":50 },
  {"Name":"1760x32x50",       "Output-Size":1760,      "Batch-Size":32,         "Time-Steps":50 },
  {"Name":"1760x64x50",       "Output-Size":1760,      "Batch-Size":64,         "Time-Steps":50 },
  {"Name":"1760x128x50",      "Output-Size":1760,      "Batch-Size":128,        "Time-Steps":50 },

  {"Name":"2048x16x50",       "Output-Size":2048,      "Batch-Size":16,         "Time-Steps":50 },
  {"Name":"2048x32x50",       "Output-Size":2048,      "Batch-Size":32,         "Time-Steps":50 },
  {"Name":"2048x64x50",       "Output-Size":2048,      "Batch-Size":64,         "Time-Steps":50 },
  {"Name":"2048x128x50",      "Output-Size":2048,      "Batch-Size":128,        "Time-Steps":50 },

  {"Name":"2560x16x50",       "Output-Size":2560,      "Batch-Size":16,         "Time-Steps":50 },
  {"Name":"2560x32x50",       "Output-Size":2560,      "Batch-Size":32,         "Time-Steps":50 },
  {"Name":"2560x64x50",       "Output-Size":2560,      "Batch-Size":64,         "Time-Steps":50 },
  {"Name":"2560x128x50",      "Output-Size":2560,      "Batch-Size":128,        "Time-Steps":50 },
  ]

def run_rnn(data_type, methods, tests):
  directory = "./deepbench"
  if not os.path.exists(directory):
    os.makedirs(directory)

  for j in range(0, len(methods)):
    # @TODO: change this to the correct method
    for i in range(tests[0], tests[1]):
      test = test_rnn_dict[i];

      exec_str =  "tools/rnn_layer " \
                  " --output-size=" + str(test["Output-Size"]) + \
                  " --batch-size=" + str(test["Batch-Size"]) + \
                  " --sequence-size=" + str(test["Time-Steps"]) + \
                  " --data-type=" + data_type

      exec_str = exec_str + ' > ' + "./deepbench/rnn_" + test["Name"] + methods[j] + '.res'

      print("Running...   " + exec_str)
      os.system(exec_str)


def process_results(tests, methods):
  print('RESULTS:')
  for j in range(0, len(methods)):
    for i in range(tests[0], tests[1]):
      fname = "./deepbench/rnn_" + test_rnn_dict[i]["Name"] + methods[j] + '.res'
      try:
        with open(fname, 'r') as inF:
          for line in inF:
            if 'Number of cycles:' in line:
              print(fname + " : " + line)
      except IOError:
        print(fname + " not found")

def print_test_list():
  for i in range(0, len(test_rnn_dict)):
    print("Test " + str(i+1) + '   :    ' + test_rnn_dict[i]["Name"])


def usage():
  print('usage: deepbench_rnn.py')
  print('Options:')
  print('        -h               | --help                       produce this message')
  print('        -d <float|half>  | --data-type=<float|half>     data type')
  print('        -p <all|fwd|bwd> | --pass= <all|fwd|bwd>        pass to use, all = fwd && bwd')
  print('        -l               | --list                       print list of tests')
  print('        -t <number>      | --test=<number>              run specific test')
  print('        -r               | --report                     produce report')

def main(argv):
  try:
    opts, args = getopt.getopt(argv, 'hlrd:p:t:', ['help', 'list', 'report', 'data-type=','pass=', 'test='])
  except getopt.GetoptError:
    usage()
    sys.exit()

  data_type = 'float'
  tests = (0, len(test_rnn_dict))
  report_results = 0

  # @TODO: change this to ('fwd', 'bwd')
  methods = ['fwd']
  for o, a in opts:
    if o in ('-h', '--help'):
      usage()
      sys.exit()
    elif o in ('-d', '--data-type'):
      print('....')
      if a in ('float', 'half'):
        data_type = a  
      else:
        usage()
        sys.exit()
    elif o in ('-p', '--pass'):
      if a in ('all', 'fwd', 'bwd'):
        if a == 'all':
          methods = ['fwd', 'bwd']
        else:
          methods = [a]
      else:
        usage()
        sys.exit()
    elif o in ('-t', '--test'):
      test = int(a)
      u_bound = len(test_rnn_dict) + 1
      if 1 <= test < u_bound:
        tests = (test - 1, test)
      else:
        print('Test not supported')
        sys.exit()
    elif o in ('-l', '--list'):
      print_test_list()
      sys.exit()
    elif o in ('-r', '--report'):
      report_results = 1;
    else:
      usage()
      sys.exit()

  run_rnn(data_type, methods, tests)

  if report_results:
    process_results(tests, methods)

if __name__ == '__main__':
  main(sys.argv[1:])
