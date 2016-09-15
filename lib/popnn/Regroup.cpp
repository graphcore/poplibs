#include "Regroup.hpp"
#include "DimShuffle.hpp"
#include "gcd.hpp"
#include "VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;

void
regroup(Graph &graph, const ComputeSet &cs,
        const std::string &inType, const std::string &outType,
        const std::vector<unsigned> &outTileMapping,
        Tensor in, Tensor out) {
  const auto outNumChanGroups = out.dim(0);
  const auto dimY = out.dim(1);
  const auto dimX = out.dim(2);
  const auto outChansPerGroup = out.dim(3);
  const auto numChans = outNumChanGroups * outChansPerGroup;
  const auto inNumChanGroups = in.dim(0);
  assert(in.dim(1) == dimY);
  assert(in.dim(2) == dimX);
  const auto inChansPerGroup = in.dim(3);
  assert(inNumChanGroups * inChansPerGroup == numChans);
  // Try to implement regrouping using a dimshuffle.
  if (inType == outType && inChansPerGroup % outChansPerGroup == 0) {
    out = out.reshape({inNumChanGroups, inChansPerGroup / outChansPerGroup,
                       dimY,
                       dimX,
                       outChansPerGroup});
    in = in.reshape({inNumChanGroups,
                     dimY,
                     dimX,
                     inChansPerGroup / outChansPerGroup, outChansPerGroup});
    std::vector<unsigned> permutation = {0, 3, 1, 2, 4};
    dimShuffle(graph, cs, in, out, permutation, outTileMapping);
    return;
  }
  if (inType == outType && outChansPerGroup % inChansPerGroup == 0) {
    out = out.reshape({outNumChanGroups,
                       dimY,
                       dimX,
                       outChansPerGroup / inChansPerGroup, inChansPerGroup});
    in = in.reshape({outNumChanGroups, outChansPerGroup / inChansPerGroup,
                     dimY,
                     dimX,
                     inChansPerGroup});
    std::vector<unsigned> permutation = {0, 2, 3, 1, 4};
    dimShuffle(graph, cs, in, out, permutation, outTileMapping);
    return;
  }
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numTiles = deviceInfo.getNumTiles();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto workersPerTile = deviceInfo.numWorkerContexts;
  const auto chunkSize = gcd<unsigned>(outChansPerGroup, inChansPerGroup);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileActivationsBegin = outTileMapping[tile];
    const auto tileActivationsEnd = outTileMapping[tile + 1];
    assert(tileActivationsBegin % outChansPerGroup == 0);
    assert(tileActivationsEnd % outChansPerGroup == 0);
    const auto tileGroupBegin = tileActivationsBegin / outChansPerGroup;
    const auto tileGroupEnd = tileActivationsEnd / outChansPerGroup;
    const auto tileNumGroups = tileGroupEnd - tileGroupBegin;
    if (tileNumGroups == 0)
      continue;
    const auto maxGroupsPerWorker =
        (tileNumGroups + workersPerTile - 1) / workersPerTile;
    // Choose the number of vertices such that each vertices is reponsible for
    // at most maxGroupsPerWorker groups.
    const auto verticesToCreate =
        (tileNumGroups + maxGroupsPerWorker - 1) / maxGroupsPerWorker;
    for (unsigned vertex = 0; vertex != verticesToCreate; ++vertex) {
      const auto groupBegin =
          (vertex * tileNumGroups) / verticesToCreate + tileGroupBegin;
      const auto groupEnd =
          ((vertex + 1) * tileNumGroups) / verticesToCreate + tileGroupBegin;
      if (groupBegin == groupEnd)
        continue;
      // Create a vertex for this worker to process a number of output channel
      // groups.
      const auto numGroups = groupEnd - groupBegin;
      auto v = graph.addVertex(cs,
                               templateVertex("RegroupChans",
                                              inType, outType));
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
      graph.setInitialValue<unsigned>(v["outChans"], outChansPerGroup);
      // Connect the output channel groups and inputs from the partial sums.
      graph.setFieldSize(v["out"], numGroups);
      graph.setFieldSize(v["in"],
                         numGroups * outChansPerGroup / chunkSize);
      unsigned numIn = 0;
      for (auto group = groupBegin; group != groupEnd; ++group) {
        auto outChanGroup = group / (dimX * dimY);
        auto y = group % (dimX * dimY) / dimX;
        auto x = group % dimX;
        auto groupOut = out[outChanGroup][y][x];
        graph.connect(v["out"][group - groupBegin], groupOut);
        Tensor inChans = in.slice(
           {0, y, x, 0},
           {in.dim(0), y + 1, x + 1, inChansPerGroup}
        ).flatten();
        Tensor inByChanGroup =
            inChans.reshape({outNumChanGroups,
                             outChansPerGroup / chunkSize,
                             chunkSize});
        Tensor in = inByChanGroup[outChanGroup];
        for (unsigned i = 0; i < in.dim(0); ++i) {
          graph.connect(in[i], v["in"][numIn++]);
        }
      }
    }
  }
}
