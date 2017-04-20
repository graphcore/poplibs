#!/usr/bin/env python

import os, sys
import getopt



# Tests are numbered from 1 to length of dictionary
test_conv_dict =  [ 
{"Name":"DS-700x161x1_batch4",       "Width":700, "Height":161, "Batch-Size":4,   "Input-Channels":1,   "Output-Channels":32,  "Kernel-Width":20, "Kernel-Height":5,  "Pad-Width":0, "Pad-Height":0, "Vertical-Stride":2, "Horizontal-Stride":2},
{"Name":"DS-700x161x1_batch8",       "Width":700, "Height":161, "Batch-Size":8,   "Input-Channels":1,   "Output-Channels":32,  "Kernel-Width":20, "Kernel-Height":5,  "Pad-Width":0, "Pad-Height":0, "Vertical-Stride":2, "Horizontal-Stride":2},
{"Name":"DS-700x161x1_batch16",      "Width":700, "Height":161, "Batch-Size":16,  "Input-Channels":1,   "Output-Channels":32,  "Kernel-Width":20, "Kernel-Height":5,  "Pad-Width":0, "Pad-Height":0, "Vertical-Stride":2, "Horizontal-Stride":2},
{"Name":"DS-700x161x1_batch32",      "Width":700, "Height":161, "Batch-Size":32,  "Input-Channels":1,   "Output-Channels":32,  "Kernel-Width":20, "Kernel-Height":5,  "Pad-Width":0, "Pad-Height":0, "Vertical-Stride":2, "Horizontal-Stride":2},
{"Name":"DS-341x79x1_batch4",        "Width":341, "Height":79,  "Batch-Size":4,   "Input-Channels":32,  "Output-Channels":32,  "Kernel-Width":10, "Kernel-Height":5,  "Pad-Width":0, "Pad-Height":0, "Vertical-Stride":2, "Horizontal-Stride":2},
{"Name":"DS-341x79x1_batch8",        "Width":341, "Height":79,  "Batch-Size":8,   "Input-Channels":32,  "Output-Channels":32,  "Kernel-Width":10, "Kernel-Height":5,  "Pad-Width":0, "Pad-Height":0, "Vertical-Stride":2, "Horizontal-Stride":2},
{"Name":"DS-341x79x1_batch16",       "Width":341, "Height":79,  "Batch-Size":16,  "Input-Channels":32,  "Output-Channels":32,  "Kernel-Width":10, "Kernel-Height":5,  "Pad-Width":0, "Pad-Height":0, "Vertical-Stride":2, "Horizontal-Stride":2},
{"Name":"DS-341x79x1_batch32",       "Width":341, "Height":79,  "Batch-Size":32,  "Input-Channels":32,  "Output-Channels":32,  "Kernel-Width":10, "Kernel-Height":5,  "Pad-Width":0, "Pad-Height":0, "Vertical-Stride":2, "Horizontal-Stride":2},


{"Name":"OCR-480x48",                "Width":480, "Height":48,  "Batch-Size":16,  "Input-Channels":1,   "Output-Channels":16,  "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"OCR-240x24",                "Width":240, "Height":24,  "Batch-Size":16,  "Input-Channels":16,  "Output-Channels":32,  "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"OCR-120x12",                "Width":120, "Height":12,  "Batch-Size":16,  "Input-Channels":32,  "Output-Channels":64,  "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"OCR-60x6",                  "Width":60,  "Height":6,   "Batch-Size":16,  "Input-Channels":64,  "Output-Channels":128, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},

{"Name":"Face-Recog-108x108",        "Width":108, "Height":108, "Batch-Size":8,   "Input-Channels":3,   "Output-Channels":64,  "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":2, "Horizontal-Stride":2},
{"Name":"Face-Recog-54x54",          "Width":54,  "Height":54,  "Batch-Size":8,   "Input-Channels":64,  "Output-Channels":64,  "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Face-Recog-27x27",          "Width":27,  "Height":27,  "Batch-Size":8,   "Input-Channels":128, "Output-Channels":128, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Face-Recog-14x14",          "Width":14,  "Height":14,  "Batch-Size":8,   "Input-Channels":128, "Output-Channels":256, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Face-Recog-7x7",            "Width":7,   "Height":7,   "Batch-Size":8,   "Input-Channels":256, "Output-Channels":512, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},

{"Name":"Vision-224x224x3_batch8",   "Width":224, "Height":224, "Batch-Size":8,   "Input-Channels":3,   "Output-Channels":64,  "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-112x112x64_batch8",  "Width":112, "Height":112, "Batch-Size":8,   "Input-Channels":64,  "Output-Channels":128, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-56x56x128_batch8",   "Width":56,  "Height":56,  "Batch-Size":8,   "Input-Channels":128, "Output-Channels":256, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-28x28x256_batch8",   "Width":28,  "Height":28,  "Batch-Size":8,   "Input-Channels":256, "Output-Channels":512, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-14x14x512_batch8",   "Width":14,  "Height":14,  "Batch-Size":8,   "Input-Channels":512, "Output-Channels":512, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-7x7x512_batch8",     "Width":7,   "Height":7,   "Batch-Size":8,   "Input-Channels":512, "Output-Channels":512, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},

{"Name":"Vision-224x224x3_batch16",  "Width":224, "Height":224, "Batch-Size":16,  "Input-Channels":3,   "Output-Channels":64,  "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-112x112x64_batch16", "Width":112, "Height":112, "Batch-Size":16,  "Input-Channels":64,  "Output-Channels":128, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-56x56x128_batch16",  "Width":56,  "Height":56,  "Batch-Size":16,  "Input-Channels":128, "Output-Channels":256, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-28x28x256_batch16",  "Width":28,  "Height":28,  "Batch-Size":16,  "Input-Channels":256, "Output-Channels":512, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-14x14x512_batch16",  "Width":14,  "Height":14,  "Batch-Size":16,  "Input-Channels":512, "Output-Channels":512, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-7x7x512_batch16",    "Width":7,   "Height":7,   "Batch-Size":16,  "Input-Channels":512, "Output-Channels":512, "Kernel-Width":3,  "Kernel-Height":3,  "Pad-Width":1, "Pad-Height":1, "Vertical-Stride":1, "Horizontal-Stride":1},

{"Name":"Vision-224x224x3_kernel7",  "Width":224, "Height":224, "Batch-Size":16,  "Input-Channels":3,   "Output-Channels":64,  "Kernel-Width":7,  "Kernel-Height":7,  "Pad-Width":3, "Pad-Height":3, "Vertical-Stride":2, "Horizontal-Stride":2},
{"Name":"Vision-28x28x192_kernel5",  "Width":28,  "Height":28,  "Batch-Size":16,  "Input-Channels":192, "Output-Channels":32,  "Kernel-Width":5,  "Kernel-Height":5,  "Pad-Width":2, "Pad-Height":2, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-28x28x192_kernel1",  "Width":28,  "Height":28,  "Batch-Size":16,  "Input-Channels":192, "Output-Channels":64,  "Kernel-Width":1,  "Kernel-Height":1,  "Pad-Width":0, "Pad-Height":0, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-14x14x512_kernel5",  "Width":14,  "Height":14,  "Batch-Size":16,  "Input-Channels":512, "Output-Channels":48,  "Kernel-Width":5,  "Kernel-Height":5,  "Pad-Width":2, "Pad-Height":2, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-14x14x512_kernel1",  "Width":14,  "Height":14,  "Batch-Size":16,  "Input-Channels":512, "Output-Channels":192, "Kernel-Width":1,  "Kernel-Height":1,  "Pad-Width":0, "Pad-Height":0, "Vertical-Stride":1, "Horizontal-Stride":1},

{"Name":"Vision-7x7x832_kernel1",    "Width":7,   "Height":7,   "Batch-Size":16, "Input-Channels":832,  "Output-Channels":256, "Kernel-Width":1,  "Kernel-Height":1,  "Pad-Width":0, "Pad-Height":0, "Vertical-Stride":1, "Horizontal-Stride":1},
{"Name":"Vision-7x7x832_kernel5",    "Width":7,   "Height":7,   "Batch-Size":16, "Input-Channels":832,  "Output-Channels":128, "Kernel-Width":5,  "Kernel-Height":5,  "Pad-Width":2, "Pad-Height":2, "Vertical-Stride":1, "Horizontal-Stride":1},

]

def run_conv(data_type, methods, tests):
  directory = "./deepbench"
  if not os.path.exists(directory):
    os.makedirs(directory)

  for j in range(0, len(methods)):
    for i in range(tests[0], tests[1]):
      test = test_conv_dict[i];


      fwd_out_chans_per_group = min(test["Output-Channels"], 16)

      exec_str =  "tools/single_conv_layer " \
                  " --height=" + str(test["Height"]) + \
                  " --width=" + str(test["Width"]) + \
                  " --input-channels=" + str(test["Input-Channels"])  + \
                  " --output-channels=" + str(test["Output-Channels"]) + \
                  " --kernel-width=" + str(test["Kernel-Width"]) + \
                  " --kernel-height=" + str(test["Kernel-Height"]) + \
                  " --fwd-out-chans-per-group=" + str(fwd_out_chans_per_group) + \
                  " --padding-height=" + str(test["Pad-Height"]) + \
                  " --padding-width=" + str(test["Pad-Width"]) + \
                  " --stride-height=" + str(test["Vertical-Stride"]) + \
                  " --stride-width=" + str(test["Horizontal-Stride"]) + \
                  " --batch-size=" + str(test["Batch-Size"]) + \
                  " --single-phase=" + methods[j] + \
                  " --data-type=" + data_type 

      exec_str = exec_str + ' > ' + "./deepbench/conv_" + test["Name"] + methods[j] + '.res'

      print("Running...   " + exec_str)
      os.system(exec_str)


def process_results(tests, methods):
  print('RESULTS:')
  for j in range(0, len(methods)):
    for i in range(tests[0], tests[1]):
      fname = "./deepbench/conv_" + test_conv_dict[i]["Name"] + methods[j] + '.res'
      try:
        with open(fname, 'r') as inF:
          for line in inF:
            if 'Number of cycles:' in line:
              print(fname + " : " + line)
      except IOError:
        print(fname + " not found")

def print_test_list():
  for i in range(0, len(test_conv_dict)):
    print("Test " + str(i+1) + '   :    ' + test_conv_dict[i]["Name"])


def usage():
  print('usage: deepbench_conv.py')
  print('Options:')
  print('        -h               | --help                       produce this message')
  print('        -d <float|half>  | --data-type=<float|half>     data type')
  print('        -p <all|fwd|bwd> | --pass= <all|fwd|bwd|wu>     pass to use, all = fwd && bwd && wu')
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
  tests = (0, len(test_conv_dict))
  report_results = 0

  methods = ['fwd', 'bwd', 'wu']
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
    elif o in ('-p', '--pass'):
      if a in ('all', 'fwd', 'bwd', 'wu'):
        if a == 'all':
          methods = ['fwd', 'bwd', 'wu']
        else:
          methods = [a]
      else:
        usage()
        sys.exit()
    elif o in ('-t', '--test'):
      test = int(a)
      u_bound = len(test_conv_dict) + 1
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

  run_conv(data_type, methods, tests)

  if report_results:
    process_results(tests, methods)

if __name__ == '__main__':
  main(sys.argv[1:])
