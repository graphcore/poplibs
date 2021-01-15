# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import subprocess
import sys

# This script just runs whatever it is passed on the command line, and, if its
# return code denotes an error, prints an additional message.

# It is a wrapper for executing the tests that verify that popc generates RPT
# loops, which can fail if a couple of seemingly unrelated options are
# not set in the cmake configuration, so it is useful to have a message
# explaining the issue.

result = subprocess.run(sys.argv[1:])

msg = """

Cycle test used to verify generation of RPT loop by the compiler has failed.
      
This could be due to T32758. If so, make sure your cmake configuration 
contains the following:

    -DCOLOSSUS-CORE_ARCH_CMAKE_ARGS=-DENABLED_IPU_ARCH_NAMES=ipu1,ipu2
    -DIPU_ARCH_INFO_CMAKE_ARGS=-DENABLED_IPU_ARCH_NAMES=ipu1,ipu2

"""
if result.returncode != 0:
  print(msg, file=sys.stderr)

sys.exit(result.returncode)
