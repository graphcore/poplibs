#include "Residual.hpp"
#include "popnn/ActivationMapping.hpp"
#include "popnn/Convolution.hpp"
#include "popnn/exceptions.hpp"
#include "VertexTemplates.hpp"
#include "Regroup.hpp"
#include "Pad.hpp"
#include <gcd.hpp>

using namespace poplar;
using namespace poplar::program;

namespace residual {

std::uint64_t getNumberOfAdds(unsigned inDimY, unsigned inDimX,
                              unsigned inNumChans,
                              bool forwardOnly) {
  // An addition is required to add the residual information
  return inNumChans * inDimX * inDimY;
}

uint64_t getFlops(unsigned batchSize,
                  unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                  bool forwardOnly) {
  auto flopsPerItem = getNumberOfAdds(inDimY, inDimX, inNumChans, forwardOnly);
  return batchSize * flopsPerItem;
}

double getPerfectCycleCount(const Graph &graph,
                            std::string dType, unsigned batchSize,
                            unsigned inDimY, unsigned inDimX,
                            unsigned numChannels,
                            bool forwardOnly) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  unsigned dTypeSize = dType == "float" ? 4 : 2;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numFLOPs = getFlops(batchSize, inDimY, inDimX, numChannels,
                                 forwardOnly);
  const auto vectorWidth = deviceInfo.dataPathWidth / (8 * dTypeSize);
  return static_cast<double>(numFLOPs) / (vectorWidth * numTiles);
}


Tensor
arrangeResidualInput(Graph &graph,
                     Tensor resIn,
                     std::vector<size_t> outDims,
                     std::string dType,
                     ResidualMethod resMethod) {
  auto resDimY = resIn.dim(2), outDimY = outDims[2];
  auto resDimX = resIn.dim(3), outDimX = outDims[3];
  if (resDimX < outDimX || resDimY < outDimY) {
    throw popnn::popnn_error("Residual layers must use previous layers "
                             "with X and Y dimensions that are larger"
                             "than the current layer's output.");
  }
  unsigned resStride = resDimX / outDimX;
  if (resDimY / outDimY != resStride) {
    throw popnn::popnn_error("Only residual layers with the same X/Y stride"
                             "are supported");
  }
  auto resNumChanGroups = resIn.dim(1), outNumChanGroups = outDims[1];
  auto resChansPerGroup = resIn.dim(4), outChansPerGroup = outDims[4];
  auto resNumChans = resNumChanGroups * resChansPerGroup;
  auto outNumChans = outNumChanGroups * outChansPerGroup;
  if (resMethod == RESIDUAL_PAD &&
      resNumChans == outNumChans &&
      resNumChanGroups == outNumChanGroups) {
    // We can directly add the output of the previous layer to this
    // layer's output.
    return resIn;
  }

  Tensor residual;

  switch (resMethod) {
  case RESIDUAL_PAD:
    {
      Tensor resampled = resIn.subSample(resDimY / outDimY, 2)
                              .subSample(resDimX / outDimX, 3);
      // pad channel depth to match out then regroup to match it
      Tensor zPadded = pad(graph,
                           regroup(resampled, resNumChans),
                                   {resampled.dim(0), 1, resampled.dim(2),
                                    resampled.dim(3), outNumChans},
                                   {0, 0, 0, 0, 0});
      residual = regroup(zPadded, outNumChans / outNumChanGroups);
    }
    break;
  case RESIDUAL_CONCATENATE:
    assert(0 && "Weighted calculation of residual input not implemented");
    break;
  default:
    assert(0 && "Unknown residual calculation method");
  }
  return residual;
}

Program
joinResidual(Graph &graph,
             Tensor in0,
             Tensor in1,
             Tensor out) {
  assert(in0.getDimensionality() == 5); //[batch][nCG][y][g][chan]
  assert(in0.dims() == out.dims());
  assert(in1.dims() == in0.dims());

  const auto outType = graph.getTensorElementType(out);
  const auto inType = graph.getTensorElementType(in0);
  assert(inType == graph.getTensorElementType(in1));
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  ComputeSet cs = graph.createComputeSet("JoinResidual");
  Program prog = Execute(cs);
  const auto batchSize = out.dim(0);
  for (unsigned b = 0; b < batchSize; b++) {
    const auto &activationMapping =
      computeActivationsMapping(graph, out[b], b, batchSize);
    Tensor flattenedIn0 = in0[b].flatten();
    Tensor flattenedIn1 = in1[b].flatten();
    Tensor flattenedOut = out[b].flatten();
    buildTransform(activationMapping, graph, [&](unsigned deltaBegin,
                                                 unsigned deltaEnd,
                                                 unsigned tile)
      {
        auto v =
          graph.addVertex(
              cs,
              templateVertex("popnn::AddTensors", inType, outType),
              {{"in0", flattenedIn0.slice(deltaBegin, deltaEnd)},
               {"in1", flattenedIn1.slice(deltaBegin, deltaEnd)},
               {"out", flattenedOut.slice(deltaBegin, deltaEnd)}});
        graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
        graph.setTileMapping(v, tile);
      });
  }
  return prog;
}

// Combine deltas that have different shapes
// @param graph The graph to be amended
// @param outIn0 The vector that will be updated
// @param in1 The incoming/bypass deltas
//
// Typically \a in1 will have smaller X/Y dimensions and deeper channel
// dimensions. The updates to \a outIn0 will be sparse in X&Y; excess channel
// values in \a in1 are ignored
static Program
joinStridedDeltas(Graph &graph,
                    Tensor outIn0,
                    Tensor in1)
{
  assert(outIn0.dim(0) == in1.dim(0));
  const auto outIn0DType = graph.getTensorElementType(outIn0);
  const auto in1DType = graph.getTensorElementType(in1);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numBatches = outIn0.dim(0);
  const auto deltasPerBatch = outIn0.numElements() / numBatches;
  const unsigned yStride = outIn0.dim(2) / in1.dim(2);
  const unsigned xStride = outIn0.dim(3) / in1.dim(3);
  const auto zOutIn0 = outIn0.dim(1) * outIn0.dim(4);
  const auto zIn1 = in1.dim(1) * in1.dim(4);
  const auto chunkSize = gcd(outIn0.dim(4), in1.dim(4));
  const auto chunksPerX = outIn0.dim(4) / chunkSize;
  assert(zOutIn0 <= zIn1); // we can discard some input Z values

  ComputeSet joinCS = graph.createComputeSet("JoinDeltas.Bwd");
  unsigned tile = 0;
  // iterate across the output deltas. We must handle subsampling in Y and
  // X and excess values in Z
 for (unsigned b = 0; b != numBatches; b++) {
    // deltas are mapped in the same way as activations
    // indexing is over outIn0. Indexing for in1 must be scaled appropriately
    // In general the two layers map to different numbers of tiles so some
    // exchange of in1 is required
    const auto outMapping = computeActivationsMapping(graph, outIn0[b], b,
                                                      numBatches);
    for (unsigned g = 0; g != outIn0.dim(1); g++) {
      for (unsigned y = 0; y < outIn0.dim(2); y += yStride) {
        for (unsigned x = 0; x < outIn0.dim(3); x += xStride) {
          unsigned chunkOffset = b * deltasPerBatch
                                 + ((g * outIn0.dim(2)
                                     + y) * outIn0.dim(3)
                                    + x) * outIn0.dim(4);
          for (unsigned chunk = 0; chunk != chunksPerX; chunk++) {
            while (outMapping[tile] < chunkOffset)
              tile++;
            auto v = graph.addVertex(joinCS,
                       templateVertex("popnn::Accumulate",
                                      in1DType, outIn0DType));
            Tensor chunkOutIn0
              = outIn0[b][g][y][x].slice(chunk * chunkSize,
                                         (chunk + 1) * chunkSize);
            auto chan = g * outIn0.dim(4) + chunk;
            auto in1Group = chan / in1.dim(4);
            auto in1Offset = chan - in1Group * in1.dim(4);

            Tensor chunkIn1
              = in1[b][in1Group][y / yStride][x / xStride]
                .slice(in1Offset, in1Offset + chunkSize);
            graph.connect(chunkOutIn0, v["outIn0"]);
            graph.connect(chunkIn1, v["in1"]);
            graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
            graph.setTileMapping(v, tile);
            chunkOffset += chunkSize;
          }
        }
      }
    }
  }
  return Execute(joinCS);
}

Program
joinDeltas(Graph &graph,
           Tensor outIn0,
           Tensor in1) {
  const auto outType = graph.getTensorElementType(outIn0);
  const auto inType = graph.getTensorElementType(in1);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  ComputeSet cs = graph.createComputeSet("JoinResidual");
  Program prog = Execute(cs);
  const auto batchSize = outIn0.dim(0);
  if (outIn0.dims() == in1.dims()) {
    for (unsigned b = 0; b < batchSize; b++) {
      const auto &activationMapping =
        computeActivationsMapping(graph, outIn0[b], b, batchSize);
      Tensor flattenedIn1 = in1[b].flatten();
      Tensor flattenedOutIn0 = outIn0[b].flatten();
      buildTransform(activationMapping, graph, [&](unsigned deltaBegin,
                                                   unsigned deltaEnd,
                                                   unsigned tile)
        {
          auto v =
            graph.addVertex(
                cs,
                templateVertex("popnn::AddTensors", inType, outType),
                {{"in0", flattenedOutIn0.slice(deltaBegin, deltaEnd)},
                 {"in1", flattenedIn1.slice(deltaBegin, deltaEnd)},
                 {"out", flattenedOutIn0.slice(deltaBegin, deltaEnd)}});
          graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
          graph.setTileMapping(v, tile);
        });
    }
    return prog;
  } else {
    return joinStridedDeltas(graph, outIn0, in1);
  }
}

}
