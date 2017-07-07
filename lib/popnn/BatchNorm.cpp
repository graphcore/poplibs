#include "popstd/VertexTemplates.hpp"
#include "popstd/TileMapping.hpp"
#include "popstd/Util.hpp"
#include "popstd/Operations.hpp"
#include "popnn/BatchNorm.hpp"
#include "popreduce/Reduce.hpp"
#include "popstd/Add.hpp"
#include "popconv/Convolution.hpp"
#include "popstd/exceptions.hpp"
#include <poplar/Program.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <cassert>
#include <numeric>
#include <functional>
#include <map>

using namespace poplar;
using namespace poplar::program;
using namespace popstd;
using namespace popreduce;

namespace popnn {
namespace bn {

static void check(Tensor acts) {
  const auto rank = acts.rank();
  if (rank != 2 && rank != 4 ) {
    throw popstd::poplib_error("Batch norm supported for tensors of rank 2 or "
                               "4");
  }
}

std::size_t numChannels(Tensor acts) {
  return acts.dim(acts.rank() - 1);
}

std::size_t numActsPerChannel(Tensor acts) {
  if (acts.rank() > 1) {
    const auto shape = acts.shape();
    return std::accumulate(shape.begin(), shape.end()-1,
                           1, std::multiplies<std::size_t>());
  }
  return 0;
}


std::pair<Tensor, Tensor>
createBatchNormParams(Graph &graph, const Tensor acts) {
  const auto rank = acts.rank();
  check(acts);
  if (rank == 4) {
    return popconv::createBatchNormParams(graph, acts);
  } else {
    const unsigned numActs = acts.shape()[1];
    const auto dType = acts.elementType();
    auto gamma = graph.addTensor(dType, {numActs}, "gamma");
    mapTensorLinearly(graph, gamma);
    auto beta = graph.addTensor(dType, {numActs}, "beta");
    mapTensorLinearly(graph, beta);
    return std::make_pair(gamma, beta);
  }
}

std::pair<Tensor, Tensor>
batchNormEstimates(Graph &graph, const Tensor acts,
                   float eps,
                   Sequence &prog,
                   const std::string &partialsTypeStr,
                   const std::string &debugPrefix) {

  const auto rank = acts.rank();
  check(acts);
  if (rank == 4) {
    return popconv::batchNormEstimates(graph, acts, eps, prog, partialsTypeStr,
                                       debugPrefix);
  } else {

    const auto fnPrefix = debugPrefix + "/BN/Estimates";
    ComputeSet cs = graph.addComputeSet(fnPrefix);

    const auto &deviceInfo = graph.getDevice().getDeviceInfo();
    const auto dataPathWidth = deviceInfo.dataPathWidth;

    assert(acts.rank() == 2);
    const unsigned numChans = numChannels(acts);
    const auto dType = acts.elementType();

    auto mean = graph.addTensor(dType, {numChans}, "mean");
    mapTensorLinearly(graph, mean);
    auto stdDev = graph.addTensor(dType, {numChans}, "stdDev");
    mapTensorLinearly(graph, stdDev);

    auto actsShuf = acts.dimShuffle({1, 0});

    // Both mean and stdDev share the same mapping, so use mapping of one
    const auto mapping = graph.getTileMapping(mean);

    for (auto tile = 0U; tile != mapping.size(); ++tile) {
      const auto vertexRegions =
          splitRegionsBetweenWorkers(deviceInfo, mapping[tile], 1);

      for (const auto &regions : vertexRegions) {
        unsigned inpIdx = 0;
        unsigned num = 0;
        auto v = graph.addVertex(cs, templateVertex("popnn::BatchNormEstimates",
                                                    dType, partialsTypeStr));

        for (const auto &interval : regions) {
          const auto begin = interval.begin();
          const auto end = interval.end();

          for (auto i = begin; i != end; ++i) {
            graph.connect(v["acts"][inpIdx++], actsShuf[i].flatten());
          }
          graph.connect(v["mean"][num], mean.slice(interval));
          graph.connect(v["stdDev"][num], stdDev.slice(interval));
          ++num;
        }
        graph.setFieldSize(v["acts"], inpIdx);
        graph.setFieldSize(v["mean"], num);
        graph.setFieldSize(v["stdDev"], num);
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
        graph.setInitialValue(v["eps"], eps);
        graph.setTileMapping(v, tile);
      }
    }
    prog.add(Execute(cs));
    return std::make_pair(mean, stdDev);
  }
}

std::pair<Tensor, Tensor>
batchNormalise(Graph &graph,
               const Tensor &acts,
               const Tensor &gamma,
               const Tensor &beta,
               const Tensor &mean,
               const Tensor &stdDev,
               Sequence &prog,
               const std::string &debugPrefix) {
  const auto rank = acts.rank();
  check(acts);
  if (rank == 4) {
    return popconv::batchNormalise(graph, acts, gamma, beta, mean, stdDev, prog,
                                   debugPrefix);
  } else {
    const auto fnPrefix = debugPrefix + "/BN/batchNormalise";
    const auto actsShape = acts.shape();
    const auto numChans = numChannels(acts);
    const auto actsPerChan = numActsPerChannel(acts);

    auto bMean =
        mean.broadcast(actsPerChan, 0).reshape({actsPerChan, numChans});
    auto bStdDev =
        stdDev.broadcast(actsPerChan, 0).reshape({actsPerChan, numChans});
    auto actsZeroMean = popstd::sub(graph, acts, bMean, prog, fnPrefix);
    auto actsWhitened = popstd::div(graph, actsZeroMean, bStdDev, prog,
                                    fnPrefix);

    auto actsOut = mul(graph, actsWhitened,
                       gamma.broadcast(actsPerChan, 0).reshape(actsShape),
                       prog, fnPrefix);
    addTo(graph, actsOut, beta.broadcast(actsPerChan, 0).reshape(actsShape),
          1.0, prog, fnPrefix);
    return std::make_pair(actsOut, actsWhitened);
  }
}

std::pair<Tensor, Tensor>
batchNormDeltas(Graph &graph,
                const Tensor &actsWhitened,
                const Tensor &gradsIn,
                Sequence &prog,
                const std::string &partialsTypeStr,
                const std::string &debugPrefix) {
  check(actsWhitened);
  const auto rank = actsWhitened.rank();
  if (rank == 4) {
    return popconv::batchNormDeltas(graph, actsWhitened, gradsIn, prog,
                                    partialsTypeStr, debugPrefix);
  } else {
    const auto fnPrefix = debugPrefix + "/BN/deltas";
    const auto betaDelta = reduce(graph, gradsIn, prog, fnPrefix);
    const auto gammaDelta =
      reduce(graph,
             mul(graph, gradsIn, actsWhitened, prog, fnPrefix), prog, fnPrefix);
    return std::make_pair(gammaDelta, betaDelta);
  }
}

Tensor batchNormGradients(Graph &graph,
                          const Tensor &actsWhitened,
                          const Tensor &gradsIn,
                          const Tensor &gammaDelta,
                          const Tensor &betaDelta,
                          const Tensor &stdDev,
                          const Tensor &gamma,
                          Sequence &prog,
                          const std::string &partialsTypeStr,
                          const std::string &debugPrefix) {
  const auto rank = actsWhitened.rank();
  check(actsWhitened);
  if (rank == 4) {
    return popconv::batchNormGradients(graph, actsWhitened, gradsIn, gammaDelta,
                                       betaDelta, stdDev, gamma, prog,
                                       partialsTypeStr, debugPrefix);
  } else {
    const auto fnPrefix = debugPrefix + "/BN/gradients";
    const auto actsShape = actsWhitened.shape();
    const auto numElements = numActsPerChannel(actsWhitened);
    const float rScale = 1.0 / numElements;

    const auto gammaDeltaMulAct =
        mul(graph, actsWhitened,
            gammaDelta.broadcast(numElements, 0).reshape(actsShape), prog,
            fnPrefix);

    auto gradient = graph.clone(actsWhitened);
    prog.add(Copy(gradsIn, gradient));
    addTo(graph, gradient, gammaDeltaMulAct, -rScale, prog, fnPrefix);
    addTo(graph, gradient,
          betaDelta.broadcast(numElements, 0).reshape(actsShape),
          -rScale, prog, fnPrefix);

    const auto scale = div(graph, gamma, stdDev, prog, fnPrefix);

    return mul(graph, gradient,
               scale.broadcast(numElements, 0).reshape(actsShape),
               prog, fnPrefix);
  }
}

void batchNormParamUpdate(Graph &graph,
                          const Tensor &gammaDelta,
                          const Tensor &betaDelta,
                          float learningRate,
                          Tensor &gamma,
                          Tensor &beta,
                          Sequence &prog,
                          const std::string &debugPrefix) {
  const std::string fnPrefix = debugPrefix + "/BN/paramUpdate";
  addTo(graph, beta, betaDelta, -learningRate, prog, fnPrefix);
  addTo(graph, gamma, gammaDelta, -learningRate, prog, fnPrefix);
}

uint64_t getFwdFlops(uint64_t numChannels, uint64_t actsPerChannel,
                     bool computeEstimates) {
  // Acts per channel:
  // - for fc layers is the total number of batches.
  // - for conv layers it is the field size per channel * batch size
  //
  // Number of channels:
  // - for fc layers is the total number of activations in a batch
  // - for conv layers is the total number of channels

  uint64_t flopsForEstimates =
      (actsPerChannel - 1) * numChannels   // sum for mean
      + numChannels                        // divide by actsPerChannel
      + actsPerChannel * numChannels       // square
      + (actsPerChannel - 1) * numChannels // sum of squares
      + numChannels                        // divide by actsPerChannel
      + numChannels                        // mean square
      + numChannels                        // sub
      + numChannels                        // add eps
      + numChannels;                       // sqrt: revisit this
  uint64_t flopsForActs =
      + actsPerChannel * numChannels       // sub mean
      + actsPerChannel * numChannels       // divide by std dev
      + actsPerChannel * numChannels       // multiply by gamma
      + actsPerChannel * numChannels;      // add beta
  return (computeEstimates ? flopsForEstimates : 0) + flopsForActs;
}


uint64_t getBwdFlops(uint64_t numChannels, uint64_t actsPerChannel) {
  // assumes whitened activations are available
  uint64_t flopsReduceGrads =
      (actsPerChannel - 1) * numChannels   // Reduce
      + numChannels;                       // Divide by actsPerChannel
  uint64_t flopsReduceProd =
      actsPerChannel * numChannels         // product of whitenedActs * grads
      + (actsPerChannel - 1) * numChannels // reduce
      + numChannels                        // divide by actsPerChannel
      + actsPerChannel * numChannels;      // reduced multiply by whitened acts

  uint64_t finalComp =
      actsPerChannel * numChannels         // add the two parts above
      + numChannels                        // gamma divide by standard dev
      + actsPerChannel * numChannels;      // scale by (gamma/stdDev
  return flopsReduceGrads + flopsReduceProd + finalComp;
}


uint64_t getWuFlops(uint64_t numChannels, uint64_t actsPerChannel) {
  uint64_t flopsBeta =
    (actsPerChannel - 1) * numChannels     // Reduce
    + numChannels                          // multiply learning rate
    + numChannels;                         // update beta

  uint64_t flopsGamma =
    actsPerChannel * numChannels           // product of grads and activations
    + (actsPerChannel - 1) * numChannels   // reduce
    + numChannels                          // multiply learning rate
    + numChannels;                         // update gamma
  return flopsBeta + flopsGamma;
}

} // namespace bn
} // namespace popnn
