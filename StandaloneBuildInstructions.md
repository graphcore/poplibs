# Building Poplibs Externally (e.g. from Github Repo)

## Build Requirements

In order to build poplibs standalone, the latest version of the Poplar SDK must be downloaded and installed. Please see https://support.graphcore.ai/hc/en-us/articles/360001118534-Poplar-SDK-TensorFlow-Installation-Instructions

## Building Poplibs

Source the Poplar and Driver enable scripts:

    . <poplar_sdk_directory>/gc_drivers-<platform><version>/enable.sh
    . <poplar_sdk_directory>/poplar-<platform><version>/enable.sh

Clone the Poplibs Github repository

    git clone https://github.com/graphcore/poplibs.git

Create a build directory:

    cd poplibs
    mkdir build
    cd build

Run cmake:

    cmake ../ -DCMAKE_BUILD_TYPE=Release -GNinja

Build with ninja:
    
    ninja

## Linking to a custom-built Poplibs

If you have customized Poplibs and built it by linking to the Poplar SDK, you
may want to build another project that links to your custom Poplibs build. To
do this you must first install your customized Poplibs (by using `ninja install`
for example) then source the `enable.sh` script inside the install directory.
