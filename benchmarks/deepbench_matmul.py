#!/usr/bin/env python

import os, sys
import getopt

test_matmul_dict = [
    {"Name":"2560x16x2560NN",  "M":2560,  "N":16,  "K":2560,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"2560x32x2560NN",  "M":2560,  "N":32,  "K":2560,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"2560x64x2560NN",  "M":2560,  "N":64,  "K":2560,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"2560x128x2560NN", "M":2560,  "N":128, "K":2560,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"2560x7000x2560NN","M":2560,  "N":7000,"K":2560,  "Left-op":"normal", "Right-op":"normal" },

    {"Name":"1760x16x1760NN",  "M":1760,  "N":16,  "K":1760,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"1760x32x1760NN",  "M":1760,  "N":32,  "K":1760,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"1760x64x1760NN",  "M":1760,  "N":64,  "K":1760,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"1760x128x1760NN", "M":1760,  "N":128, "K":1760,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"1760x7000x1760NN","M":1760,  "N":7000,"K":1760,  "Left-op":"normal", "Right-op":"normal" },

    {"Name":"2048x16x2048NN",  "M":2048,  "N":16,  "K":2048,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"2048x32x2048NN",  "M":2048,  "N":32,  "K":2048,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"2048x64x2048NN",  "M":2048,  "N":64,  "K":2048,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"2048x128x2048NN", "M":2048,  "N":128, "K":2048,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"2048x7000x2048NN","M":2048,  "N":7000,"K":2048,  "Left-op":"normal", "Right-op":"normal" },

    {"Name":"4096x16x4096NN",  "M":4096,  "N":16,  "K":4096,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"4096x32x4096NN",  "M":4096,  "N":32,  "K":4096,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"4096x64x4096NN",  "M":4096,  "N":64,  "K":4096,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"4096x128x4096NN", "M":4096,  "N":128, "K":4096,  "Left-op":"normal", "Right-op":"normal" },
    {"Name":"4096x7000x4096NN","M":4096,  "N":7000,"K":4096,  "Left-op":"normal", "Right-op":"normal" },

    {"Name":"2560x16x2560TN",  "M":2560,  "N":16,  "K":2560,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"2560x32x2560TN",  "M":2560,  "N":32,  "K":2560,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"2560x64x2560TN",  "M":2560,  "N":64,  "K":2560,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"2560x128x2560TN", "M":2560,  "N":128, "K":2560,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"2560x7000x2560TN","M":2560,  "N":7000,"K":2560,  "Left-op":"transpose", "Right-op":"normal" },

    {"Name":"1760x16x1760TN",  "M":1760,  "N":16,  "K":1760,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"1760x32x1760TN",  "M":1760,  "N":32,  "K":1760,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"1760x64x1760TN",  "M":1760,  "N":64,  "K":1760,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"1760x128x1760TN", "M":1760,  "N":128, "K":1760,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"1760x7000x1760TN","M":1760,  "N":7000,"K":1760,  "Left-op":"transpose", "Right-op":"normal" },

    {"Name":"2048x16x2048TN",  "M":2048,  "N":16,  "K":2048,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"2048x32x2048TN",  "M":2048,  "N":32,  "K":2048,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"2048x64x2048TN",  "M":2048,  "N":64,  "K":2048,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"2048x128x2048TN", "M":2048,  "N":128, "K":2048,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"2048x7000x2048TN","M":2048,  "N":7000,"K":2048,  "Left-op":"transpose", "Right-op":"normal" },

    {"Name":"4096x16x4096TN",  "M":4096,  "N":16,  "K":4096,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"4096x32x4096TN",  "M":4096,  "N":32,  "K":4096,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"4096x64x4096TN",  "M":4096,  "N":64,  "K":4096,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"4096x128x4096TN", "M":4096,  "N":128, "K":4096,  "Left-op":"transpose", "Right-op":"normal" },
    {"Name":"4096x7000x4096TN","M":4096,  "N":7000,"K":4096,  "Left-op":"transpose", "Right-op":"normal" },

    {"Name":"2560x7133x2560NT","M":2560,  "N":7133,"K":2560,  "Left-op":"normal", "Right-op":"transpose" },
    {"Name":"1760x6574x1760NT","M":1760,  "N":6574,"K":1760,  "Left-op":"normal", "Right-op":"transpose" },
    {"Name":"2048x10376x2048NT","M":2048, "N":10376,"K":2048, "Left-op":"normal", "Right-op":"transpose" },
    {"Name":"4096x8935x4096NT","M":4096,  "N":8935,"K":4096,  "Left-op":"normal", "Right-op":"transpose" }
    ]

def run_matmul(data_type, tests):

  directory = "./deepbench"
  if not os.path.exists(directory):
    os.makedirs(directory)

  for i in range(tests[0], tests[1]):
    test = test_matmul_dict[i];

    exec_str =  "tools/general_matrix_multiply " \
                " --m=" + str(test["M"]) + \
                " --n=" + str(test["N"]) + \
                " --k=" + str(test["K"]) + \
                " --left-matrix-op=" + test["Left-op"] + \
                " --right-matrix-op=" + test["Right-op"] + \
                " --data-type=" + data_type

    exec_str = exec_str + ' > ' + "./deepbench/matmul_" + test["Name"] + '.res'

    print("Running...   " + exec_str)
    os.system(exec_str)

def process_results(tests):
  print('RESULTS:')
  for i in range(tests[0], tests[1]):
    fname = "./deepbench/matmul_" + test_matmul_dict[i]["Name"]+'.res'

    try:
      with open(fname, 'r') as inF:
        for line in inF:
          if 'Number of cycles:' in line:
            print(fname + " : " + line)
    except IOError:
      print(fname + " not found")


def print_test_list():
  for i in range(0, len(test_matmul_dict)):
    print("Test " + str(i+1) + '   :    ' + test_matmul_dict[i]["Name"])


def usage():
  print('usage: deepbench_matmul.py')
  print('Options:')
  print('        -h              | --help                       produce this message')
  print('        -d <float|half> | --data-type=<float|half>     data type')
  print('        -l              | --list                       print list of tests')
  print('        -t <number>     | --test=<number>              run specific test')
  print('        -r              | --report                     produce report')

def main(argv):
  try:
    opts, args = getopt.getopt(argv, 'hlrd:t:', ['help', 'list', 'report', 'data-type=', 'test='])
  except getopt.GetoptError:
    usage()
    sys.exit()

  data_type = 'float'
  tests = (0, len(test_matmul_dict));
  report_results = 0;
  method = 'all'
  for o, a in opts:
    if o in ('-h', '--help'):
      usage()
      sys.exit()
    elif o in ('-d', '--data-type'):
      if a in ('float', 'half'):
        data_type = a  
      else:
        usage()
        sys.exit()
    elif o in ('-t', '--test'):
      test = int(a)
      u_bound = len(test_matmul_dict) + 1
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

  run_matmul(data_type, tests)
  if report_results:
    process_results(tests)

if __name__ == '__main__':
  main(sys.argv[1:])