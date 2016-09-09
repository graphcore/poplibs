#include "DimShuffle.hpp"

#include <algorithm>
#include <cstdlib>
#include <poplar/Graph.hpp>
#include "VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;

static bool isIdentityPermutation(const std::vector<unsigned> &permutation) {
  for (std::size_t i = 0, e = permutation.size(); i != e; ++i) {
    if (permutation[i] != i)
      return false;
  }
  return true;
}

static void validateDimShuffleArgs(poplar::Graph &graph,
                                   poplar::Tensor in, poplar::Tensor out,
                                   std::vector<unsigned> permutation) {
  // Check the number of dimensions match.
  const auto numDims = in.getDimensionality();
  if (out.getDimensionality() != numDims) {
    std::abort();
  }
  if (permutation.size() != numDims) {
    std::abort();
  }
  // Check the data types match.
  if (graph.getTensorElementType(in) != graph.getTensorElementType(out)) {
    std::abort();
  }
  // Check permutation is a valid permutation.
  auto sorted = permutation;
  std::sort(sorted.begin(), sorted.end());
  if (!isIdentityPermutation(sorted)) {
    std::abort();
  }
  // Check the size of the permuted dimensions match.
  const auto &inDims = in.dims();
  const auto &outDims = out.dims();
  for (std::size_t i = 0; i != numDims; ++i) {
    if (outDims[i] != inDims[permutation[i]])
      std::abort();
  }
}

poplar::program::Program
dimShuffle(poplar::Graph &graph,
           poplar::Tensor in, poplar::Tensor out,
           const std::vector<unsigned> &permutation,
           const std::vector<unsigned> &outTileMapping) {
  validateDimShuffleArgs(graph, in, out, permutation);
  if (isIdentityPermutation(permutation)) {
    return Copy(out, in);
  }
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dType = graph.getTensorElementType(in);
  const auto &inDims = in.dims();
  const auto numDims = inDims.size();
  const auto &outDims = out.dims();
  unsigned chunkSize = 1;
  if (permutation.back() == numDims - 1) {
    chunkSize = inDims.back();
  }
  const auto numTiles = deviceInfo.getNumTiles();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto workersPerTile = deviceInfo.numWorkerContexts;
  ComputeSet cs = graph.createComputeSet("dimShuffle");
  std::vector<unsigned> inIndices(numDims);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileChunkBegin = outTileMapping[tile] / chunkSize;
    const auto tileChunkEnd = outTileMapping[tile + 1] / chunkSize;
    const auto tileNumChunks = tileChunkEnd - tileChunkBegin;
    if (tileNumChunks == 0)
      continue;
    const auto maxChunksPerWorker =
        (tileNumChunks + workersPerTile - 1) / workersPerTile;
    // Choose the number of vertices such that each vertex is reponsible for
    // at most maxChunksPerWorker groups.
    const auto verticesToCreate =
        (tileNumChunks + maxChunksPerWorker - 1) / maxChunksPerWorker;
    for (unsigned vertex = 0; vertex != verticesToCreate; ++vertex) {
      const auto chunkBegin =
          (vertex * tileNumChunks) / verticesToCreate + tileChunkBegin;
      const auto chunkEnd =
          ((vertex + 1) * tileNumChunks) / verticesToCreate + tileChunkBegin;
      if (chunkBegin == chunkEnd)
        continue;
      // Create a vertex for this worker to process a number of output chunks.
      const auto numChunks = chunkEnd - chunkBegin;
      auto v = graph.addVertex(cs,
                               templateVertex("DimShuffle", dType));
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
      graph.connect(out.flatten().slice(chunkBegin * chunkSize,
                                        chunkEnd * chunkSize),
                    v["out"]);
      graph.setFieldSize(v["in"], numChunks);
      for (unsigned chunk = chunkBegin; chunk != chunkEnd; ++chunk) {
        auto outOffset = chunk * chunkSize;
        // Compute the indices in the input tensor.
        for (int dim = numDims - 1; dim >= 0; --dim) {
          inIndices[permutation[dim]] = outOffset % outDims[dim];
          outOffset /= outDims[dim];
        }
        // Compute the offset in the input tensor.
        auto inOffset = 0;
        for (unsigned i = 0; i != numDims; ++i) {
          inOffset *= inDims[i];
          inOffset += inIndices[i];
        }
        graph.connect(in.flatten().slice(inOffset, inOffset + chunkSize),
                      v["in"][chunk - chunkBegin]);
      }
    }
  }
  return Execute(cs);
}
