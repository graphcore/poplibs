#ifndef __Winograd_hpp__
#define __Winograd_hpp__

#include <poplar/Program.hpp>

namespace poplin {

struct WinogradParams {
  WinogradParams(std::vector<unsigned> inputTransformPaddingLower,
                 std::vector<unsigned> inputTransformPaddingUpper,
                 std::vector<unsigned> outputTransformStride)
  : inputTransformPaddingLower(std::move(inputTransformPaddingLower)),
    inputTransformPaddingUpper(std::move(inputTransformPaddingUpper)),
    outputTransformStride(std::move(outputTransformStride)) {}

  std::vector<unsigned> inputTransformPaddingLower;
  std::vector<unsigned> inputTransformPaddingUpper;
  std::vector<unsigned> outputTransformStride;
};

struct WinogradOptions {
  WinogradOptions(unsigned numIPUs, unsigned tilesPerIPU)
  : numIPUs(numIPUs), tilesPerIPU(tilesPerIPU) {}

  unsigned numIPUs;
  unsigned tilesPerIPU;

  unsigned getNumTiles() const {
    return numIPUs * tilesPerIPU;
  }
};

poplar::program::Program
winogradConvolution(poplar::Graph &graph,
                    const WinogradParams &params,
                    const WinogradOptions &options,
                    const poplar::Tensor &in,
                    const poplar::Tensor &weights,
                    const poplar::Tensor &out,
                    unsigned patchSizeX,
                    unsigned patchSizeY,
                    const poplar::Type &partialsType,
                    const std::string &debugPrefix = "");
}

#endif //__Winograd_hpp__
