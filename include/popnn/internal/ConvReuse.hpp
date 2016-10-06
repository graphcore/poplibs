#ifndef __ConvReuse_hpp__
#define __ConvReuse_hpp__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include "popnn/ConvDef.hpp"
#include "popnn/NonLinearityDef.hpp"

struct ReusableLayer {
  poplar::program::Program prog;
  std::vector<poplar::Tensor> inputs;
  std::vector<poplar::Tensor> outputs;
  ReusableLayer(poplar::program::Program prog,
                std::vector<poplar::Tensor> inputs,
                std::vector<poplar::Tensor> outputs) :
    prog(prog),
    inputs(std::move(inputs)),
    outputs(std::move(outputs)) {}

  poplar::program::Program
  apply(const std::vector<poplar::Tensor> &in,
        const std::vector<poplar::Tensor> &out);
};

class ConvImplSpec {
public:
  std::vector<std::vector<size_t>> tensorDims;
  unsigned kernelSizeY, kernelSizeX, strideY, strideX;
  unsigned paddingY, paddingX;
  NonLinearityType nonLinearityType;
  ResidualMethod resMethod;
  ConvImplSpec(std::vector<std::vector<size_t>> tensorDims,
               unsigned kernelSizeY, unsigned kernelSizeX,
               unsigned strideY, unsigned strideX,
               unsigned paddingY, unsigned paddingX,
               NonLinearityType nonLinearityType,
               ResidualMethod resMethod) :
    tensorDims(std::move(tensorDims)),
    kernelSizeY(kernelSizeY), kernelSizeX(kernelSizeX),
    strideY(strideY), strideX(strideX),
    paddingY(paddingY), paddingX(paddingX),
    nonLinearityType(nonLinearityType), resMethod(resMethod) {}

  bool operator<(const ConvImplSpec &other) const;
};


#endif // __ConvReuse_hpp__
