# Building PopLibs Externally

This applies when building from the PopLibs GitHub Repository.

Note that only Ubuntu 18.04 is supported for building PopLibs externally.

## Build Requirements

### Poplar SDK

In order to build PopLibs, you must download and install the Poplar SDK.
Please see the Getting Started guide available at https://www.graphcore.ai/developer for details.

The Poplar SDK version must match the version specified in the name of the PopLibs branch.
For example, `release-1.3` requires Poplar SDK 1.3.x.

These instructions are for Poplar SDK 1.3 and later.

### CMake Version 3.12.0 or later

PopLibs requires CMake 3.12.0 or later. The package manager in Ubuntu 18.04 will install CMake 3.10.
If you see errors about your CMake version you may need to uninstall the apt version using:

    $ sudo apt-get remove cmake

Then install a recent version using pip:

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

PopLibs' sparsity support (popsparse) comes in two flavours: static and dynamic. Zoltan (http://www.cs.sandia.gov/Zoltan/) is a third party computational graph partitioning tool that is used to generate efficient parallel graphs for static sparsity problems. Even if you are only using popsparse's dynamic sparsity support you will still need to acquire and build Zoltan to build popsparse.

Acquire Zoltan from http://www.cs.sandia.gov/Zoltan/ under the LGPL license then build and install it as follows:

On Ubuntu 18.04:

    $ cd zoltan
    $ mkdir build
    $ cd build
    $ ../configure --prefix install --disable-mpi --disable-zoltan-cppdriver --with-cflags='-fPIC' --with-cxxflags='-fPIC' --disable-tests  --disable-zoltan-tests
    $ make
    $ make install

You will use the install path later when configuring the PopLibs build.

### Python 3 (optional)

Python 3 is an optional dependency. If installed, additional convolution unit tests will become available.

Ubuntu 18.04 ships with Python 3 already installed.

### Ninja Version 1.8.2 (optional)

These instructions use Ninja to build PopLibs. However, you may choose to use an alternative build system.

On Ubuntu 18.04:

    $ apt install ninja-build

## Building PopLibs

Source the Poplar enable script:

    $ . <poplar_sdk_directory>/poplar-<platform><version>/enable.sh

Note: from Poplar SDK 1.3 there is no additional enable script for the drivers.

Create `build` and `install` directories within your PopLibs source directory:

    $ mkdir build install
    $ cd build

Run cmake then build with Ninja:

    $ cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DBOOST_ROOT=<absolute path Boost was installed to> -GNinja -DCMAKE_PREFIX_PATH=<absolute path Spdlog was installed to>
    $ ninja

Note: if you intend to use the popsparse library you will need to have Zoltan installed as described in Build Requirements then tell cmake where to find it by adding the following to the cmake command above:

    -DZOLTAN_ROOT=<path Zoltan was installed to>

Note: There are some warnings that can be ignored:
 * CMake Warnings of the form "New Boost version may have incorrect or missing dependencies and imported targets"

Install with Ninja:

    $ ninja install

To start using this build of PopLibs in your current shell you must source the enable script:

    $ . ../install/enable.sh

## Using PopLibs

To use this build of PopLibs in a new shell, source the Poplar
enable script and then source the PopLibs enable script:

    $ . <poplar_sdk_directory>/poplar-<platform><version>/enable.sh
    $ . <path to poplibs install directory>/enable.sh
