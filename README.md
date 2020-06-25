# Building PopLibs Externally (e.g. from Github Repo)

Note that only Ubuntu 18.04 is supported for building PopLibs externally.

## Build Requirements

### Poplar SDK

In order to build PopLibs, the Poplar SDK must be downloaded and installed. Please see https://www.graphcore.ai/developer for details.

### CMake Version 3.10.2 or greater

On Ubuntu 18.04:

    apt install cmake

### Boost Version 1.70.0 (or compatible with)

Download Boost 1.70 source from here: https://www.boost.org/users/history/version_1_70_0.html

Within a suitable directory, extract the archive and run:

    $ mkdir install # Make a note of this path for later.
    $ ./bootstrap.sh --prefix=install
    $ ./b2 link=static runtime-link=static --abbreviate-paths variant=release toolset=gcc "cxxflags= -fno-semantic-interposition -fPIC" cxxstd=14 --with-all install

Note: Consider using '-j8' (or similar) with './b2' to reduce build time by increasing concurrency.

For more information, see: https://www.boost.org/doc/libs/1_70_0/more/getting_started/unix-variants.html

### Spdlog Version 0.16.3 (or compatible with)

On Ubuntu 18.04:

    $ apt install libspdlog-dev

### Zoltan Version 3.83 (or compatible with, optional but needed to build popsparse)

PopLibs' sparsity support (popsparse) comes in two flavours: static and dynamic. Zoltan (http://www.cs.sandia.gov/Zoltan/) is a third party computational graph partitioning tool that is used to generate efficient parallel graphs for static sparsity problems. Even if you are only using popsparse's dynamic sparsity support you will still need to acquire and build Zoltan to build popsparse.

Acquire Zoltan from http://www.cs.sandia.gov/Zoltan/ under the LGPL license then build and install it as follows:

On Ubuntu 18.04:

    $ cd zoltan/
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

Source the Poplar and Driver enable scripts:

    $ . <poplar_sdk_directory>/gc_drivers-<platform><version>/enable.sh
    $ . <poplar_sdk_directory>/poplar-<platform><version>/enable.sh

Create 'build' and 'install' directories within your PopLibs source directory:

    $ mkdir build install
    $ cd build

Run cmake then build with Ninja:

    $ cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DBOOST_ROOT=<path Boost was installed to> -GNinja
    $ ninja

Note: if you intend to use the popsparse library you will need to have Zoltan installed as described in Build Requirements then tell cmake where to find it by adding the following to the cmake command above:

    -DZOLTAN_ROOT=<path Zoltan was installed to>

Install with Ninja then source the enable script:

    $ ninja install
    $ . ../install/enable.sh
