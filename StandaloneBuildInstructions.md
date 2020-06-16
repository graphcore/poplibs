# Building Poplibs Externally (e.g. from Github Repo)

Note that only Ubuntu 18.04 is supported for building PopLibs externally.

## Build Requirements

### Poplar SDK

In order to build Poplibs standalone, the latest version of the Poplar SDK must be downloaded and installed. Please see https://support.graphcore.ai/hc/en-us/articles/360001118534-Poplar-SDK-TensorFlow-Installation-Instructions

### CMake Version 3.10.2 or greater

On Ubuntu 18.04:

    apt install cmake

### Boost Version 1.65.1 (or compatible with)

On Ubuntu 18.04:

    apt install libboost-all-dev

### Spdlog Version 0.16.3 (or compatible with)

On Ubuntu 18.04:

    apt install libspdlog-dev

### Python 3 (optional)

Python 3 is an optional dependency. If installed, additional convolution unit
tests will become available.

Ubuntu 18.04 ships with Python 3 already installed.

## Building Poplibs

Source the Poplar and Driver enable scripts:

    . <poplar_sdk_directory>/gc_drivers-<platform><version>/enable.sh
    . <poplar_sdk_directory>/poplar-<platform><version>/enable.sh

Clone the Poplibs Github repository

    git clone https://github.com/graphcore/poplibs.git

Create 'build' and 'install' directories:

    cd poplibs
    mkdir build install
    cd build

Run cmake then build with Ninja:

    cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -GNinja
    ninja

Install with Ninja then source the enable script:

    ninja install
    . ../install/enable.sh
