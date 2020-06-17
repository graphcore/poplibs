# Building PopLibs Externally (e.g. from Github Repo)

Note that only Ubuntu 18.04 is supported for building PopLibs externally.

Note that only Ubuntu 18.04 is supported for building PopLibs externally.

## Build Requirements

### Poplar SDK

In order to build PopLibs, the Poplar SDK must be downloaded and installed. Please see https://www.graphcore.ai/developer for details.

### CMake Version 3.10.2 or greater

On Ubuntu 18.04:

    apt install cmake

### Boost Version 1.70.0 (or compatible with)

Download Boost 1.70 source from here: https://www.boost.org/users/history/version_1_70_0.html

For build and installation instructions, see: https://www.boost.org/doc/libs/1_70_0/more/getting_started/unix-variants.html

### Spdlog Version 0.16.3 (or compatible with)

On Ubuntu 18.04:

    $ apt install libspdlog-dev

### Python 3 (optional)

Python 3 is an optional dependency. If installed, additional convolution unit
tests will become available.

Ubuntu 18.04 ships with Python 3 already installed.

### Ninja Version 1.8.2 (optional)

These instructions use Ninja to build PopLibs. However, you may choose to use an alternative build system.

On Ubuntu 18.04:

    $ apt install ninja-build

## Building PopLibs

Source the Poplar and Driver enable scripts:

    $ . <poplar_sdk_directory>/gc_drivers-<platform><version>/enable.sh
    $ . <poplar_sdk_directory>/poplar-<platform><version>/enable.sh

Create 'build' and 'install' directories within your PopLibs source directory:

    $ mkdir build install
    $ cd build

Run cmake then build with Ninja:

    $ cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -GNinja
    $ ninja

Install with Ninja then source the enable script:

    $ ninja install
    $ . ../install/enable.sh
