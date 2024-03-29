# Overview

The Poplar SDK is a complete software stack for graph programming on the IPU. It includes the graph compiler and supporting libraries.

The PopLibs libraries contain higher-level mathematical and machine-learning functions. These underlie the Graphcore implementation of industry-standard ML frameworks such as TensorFlow and PyTorch. If you are programming the IPU using these high-level frameworks, you don't need to be familiar with Poplar and PopLibs.

PopLibs is provided as open source so you can use the code to understand how functions are implemented on the IPU. You can also extend PopLibs to create your own custom operations. This document describes how to build PopLibs from source.

PopLibs consists of the following libraries:

* **poplin:** Linear algebra functions (matrix multiplications, convolutions)
* **popnn:** Functions used in neural networks (for example, non-linearities, pooling and loss functions)
* **popops:** Operations on tensors in control programs (elementwise functions and reductions)
* **poprand:** Functions for populating tensors with random numbers
* **popsparse:** Functions for operating on sparse tensors
* **popfloat** Supporting functions
* **poputil:** General utility functions for building graphs

For more information, refer to the [Poplar and PopLibs API Reference](https://docs.graphcore.ai/projects/poplar-api/).

## Building PopLibs Externally

The following description is for building PopLibs from the source on the GitHub repository.

Note that only Ubuntu 20.04 is supported for building PopLibs externally.

## Build Requirements

### Poplar SDK

In order to build PopLibs, you must download and install the Poplar SDK.
Please see the Getting Started guides available at https://docs.graphcore.ai/ for details.

The Poplar SDK version must match the version specified in the name of the PopLibs branch.
For example, `sdk-release-3.2` requires Poplar SDK 3.2.x.

These instructions are for Poplar SDK 3.2 and later.

### CMake Version 3.16.0 or later

PopLibs requires CMake 3.12.0 or later. The package manager in Ubuntu 20.04 will install CMake 3.16.
You can install a recent version using pip:

    $ pip3 install cmake

### Boost Version 1.70.0 (or compatible with)

Download Boost 1.70 source from here: https://www.boost.org/users/history/version_1_70_0.html

Within a suitable directory, extract the archive and run:

    $ cd boost_1_70_0
    $ mkdir install # Make a note of this path for later.
    $ ./bootstrap.sh --prefix=install
    $ ./b2 link=static runtime-link=static --abbreviate-paths variant=release toolset=gcc "cxxflags= -fno-semantic-interposition -fPIC" cxxstd=14 --with=all install

Note: Consider using `-j8` (or similar) with `./b2` to reduce build time by increasing concurrency.

For more information, see: https://www.boost.org/doc/libs/1_70_0/more/getting_started/unix-variants.html

### Spdlog Version 1.8.0 (or compatible with)

    $ git clone --branch v1.8.0 https://github.com/gabime/spdlog.git
    $ mkdir -p spdlog/build/install
    $ cd spdlog/build
    $ cmake .. -DCMAKE_INSTALL_PREFIX=./install
    $ make
    $ make install

### Zoltan Version 3.83 (or compatible with, optional but needed to build popsparse)

The sparsity support in PopLibs (popsparse) comes in two flavours: static and dynamic. Zoltan (https://cs.sandia.gov/Zoltan/) is a third party computational graph partitioning tool that is used to generate efficient parallel graphs for static sparsity problems. Even if you are only using the dynamic sparsity support in popsparse, you will still need to acquire and build Zoltan to build popsparse.

Acquire Zoltan from https://github.com/sandialabs/Zoltan/releases/tag/v3.83 then build and install it as follows:

On Ubuntu 20.04:

    $ tar xvf v3.83.tar.gz
    $ cd Zoltan-3.83
    $ mkdir build
    $ cd build
    $ ../configure --disable-mpi --disable-zoltan-cppdriver --with-cflags='-fPIC' --with-cxxflags='-fPIC' --disable-tests  --disable-zoltan-tests
    $ make -j$(nproc)
    $ make install

You will use the install path later when configuring the PopLibs build.

### Python 3 (optional)

Python 3 is an optional dependency. If installed, additional convolution unit tests will become available.

Ubuntu 20.04 ships with Python 3 already installed.

### Ninja Version 1.8.2 (optional)

These instructions use Ninja to build PopLibs. However, you may choose to use an alternative build system.

On Ubuntu 20.04:

    $ apt install ninja-build

## Building PopLibs

Source the Poplar enable script:

    $ . <poplar_sdk_directory>/poplar-<platform><version>/enable.sh

Create `build` and `install` directories within your PopLibs source directory:

    $ mkdir build install
    $ cd build

Run cmake then build with Ninja:

    $ cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -GNinja -DCMAKE_PREFIX_PATH="<absolute path Boost was installed to>;<absolute path Spdlog was installed to>"
    $ ninja

Note: if you intend to use the popsparse library you will need to have Zoltan installed as described in [Build Requirements](#build-requirements) then tell cmake where to find it by adding `path Zoltan was installed to` in `-DCMAKE_PREFIX_PATH` to the cmake command above.

Note: CMake warnings similar to the following can be ignored:

> New Boost version may have incorrect or missing dependencies and imported targets

Install with Ninja:

    $ ninja install

To start using this build of PopLibs in your current shell you must source the enable script:

    $ . ../install/enable.sh

## Using PopLibs

To use this build of PopLibs in a new shell, source the Poplar
enable script and then source the PopLibs enable script:

    $ . <poplar_sdk_directory>/poplar-<platform><version>/enable.sh
    $ . <path to poplibs install directory>/enable.sh
