#include "ExprOpUtil.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

#include <cassert>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

namespace {

// Returns a pair with second element set to true if a single channel group is
// assigned in all the intervals assigned to a tile. The first element is the
// group number
std::pair<std::size_t, bool>
getAssignedGroupForTile(const std::vector<Interval> &tileMap,
                        const std::vector<std::size_t> &firstInGroupShape) {
  if (tileMap.empty())
    return {0, false};
  // Number of elements except the outermost dimension
  const std::size_t numElemsInInnerDims =
      std::accumulate(firstInGroupShape.begin() + 1, firstInGroupShape.end(),
                      1U, std::multiplies<std::size_t>());

  std::size_t singleGroupIndex = tileMap[0].begin() / numElemsInInnerDims;
  bool singleGroup = true;

  for (std::size_t i = 0; i < tileMap.size() && singleGroup; ++i) {
    auto bIndex = tileMap[i].begin() / numElemsInInnerDims;
    auto eIndex = (tileMap[i].end() - 1) / numElemsInInnerDims;
    singleGroup = singleGroupIndex == bIndex && singleGroupIndex == eIndex;
  }
  return {singleGroupIndex, singleGroup};
}

// Add a VectorInner vertex to implement the addToChannel function, but if acts
// is too long, use multiple vertices.
// This is to simplify the codelet assembly because `rpt` can only
// loop up to 4095 times (the actual number is hardware-dependent).
// This uses a supervisor vertex that splits up the work between workers
// according to actsBlockCountPacked.
void addVectorInnerAddSupervisorVertex(Graph &graph, ComputeSet &cs,
                                       const std::string &templateVertexName,
                                       const Tensor &acts, const Tensor &addend,
                                       float scale, unsigned tile) {
  if (graph.getTarget().getNumWorkerContexts() != 6)
    throw poplibs_error("not implemented for IPUs without 6 worker contexts");

  auto addendLen = addend.numElements();
  auto actsBlockCount = acts.numElements() / addendLen;

  auto actsFlat = acts.flatten();

  auto maxBlockCount = (graph.getTarget().getRptCountMax() & ~1UL) * 6;

  std::size_t consumedBlocks = 0;

  while (consumedBlocks < actsBlockCount) {
    // The number of blocks is limited by the maximum repeat count for
    // `rpt` (getRptCountMax()). For example if it is 4095 We can do up to
    // 6 * 4094 blocks per vertex. We can't do 6 * 4094 + 2 blocks because
    // that would result in workers having the following numbers of blocks:
    //
    // 4095 4095 4094 4094 4094 4094
    //
    // Which would then be rounded to make them even to:
    //
    // 4094 4096 4094 4094 4094 4094
    //
    // And the second worker has too many blocks for the `rpt`. In fact
    // we can do 6 * 4094 + 1 blocks but let's keep things simple.
    auto thisBlockCount =
        std::min(actsBlockCount - consumedBlocks, maxBlockCount);

    auto actsSlice =
        actsFlat.slice(consumedBlocks * addendLen,
                       (consumedBlocks + thisBlockCount) * addendLen);

    auto v = graph.addVertex(cs, templateVertexName,
                             {{"data", actsSlice}, {"B", addend}});
    auto actsBlockCountPacked =
        ((thisBlockCount / 6) << 3) | (thisBlockCount % 6);

    uint16_t actsBlockCountPacked16 = actsBlockCountPacked;
    assert(actsBlockCountPacked16 == actsBlockCountPacked);

    graph.setInitialValue(v["dataBlockCountPacked"], actsBlockCountPacked16);

    if (scale != 1.0)
      graph.setInitialValue(v["scale"], scale);

    graph.setTileMapping(v, tile);

    consumedBlocks += thisBlockCount;
  }
};

// Add an VectorInner2D vertex to implement the addToChannel function, but if
// any of various length constraints are violated, use multiple vertices.
//
// acts             - The grouped activations
// firstInGroup     - The slice of `acts` that contains the first element from
//                    each channel group.
// addendByGroup    - The addend as a 2D tensor grouped into channel groups.
// outChansPerGroup - The size of each channel group (e.g. 8 or 16).
// groupsForWorker  - The groups assigned to this worker. These are
//                    intervals on firstInGroup.
// maxBlockCount    - The maximum block count for a vertex. This is only used
//                    in an assertion. The blocks should be split between
//                    multiple vertices outside this function if necessary.
// maxAddendLen     - The maximum length for an addend. If the addend is
//                    longer than this it will be split into multiple vertices
//                    inside this function.
//
void addVectorInnerAdd2DVertex(
    Graph &graph, ComputeSet &cs, const std::string &templateVertexName,
    const Tensor &acts, const Tensor &firstInGroup, const Tensor &addendByGroup,
    const std::vector<poplar::Interval> &groupsForWorker, float scale,
    unsigned tile, std::size_t maxBlockCount, std::size_t maxAddendLen) {
  assert(acts.rank() >= 3);
  assert(firstInGroup.rank() == 3);
  assert(addendByGroup.rank() == 2);
  assert(firstInGroup.dim(0) == acts.dim(0));
  assert(firstInGroup.dim(1) == acts.dim(1));
  assert(addendByGroup.dim(0) == acts.dim(0));
  assert(addendByGroup.dim(1) == acts.dim(acts.rank() - 1));

  const std::size_t outChansPerGroup = addendByGroup.dim(1);

  // We have the following limitations for BroadcastVectorInner2D<ADD>
  //
  // * dataBlockCount cannot be more than Target::getRptCountMax() because it
  //   is used in a `rpt` loop.
  // * BLen and dataBlockCount cannot be greater than 2^16-1 because
  //   their lengths are stored in uint16_t's.

  VertexRef v = graph.addVertex(cs, templateVertexName);
  graph.setTileMapping(v, tile);
  unsigned num = 0;

  // Add a sub-vector to operate on, handling splitting it into multiple
  // vertices as necessary. The addend length must be <= maxAddendLen,
  // and the block count must be <= maxBlockCount.
  auto addShortPiece = [&](const Tensor &acts, const Tensor &addend) {
    auto addendLen = addend.numElements();

    assert(addendLen <= maxAddendLen);
    assert(acts.numElements() % addendLen == 0);

    auto actsBlockCount = acts.numElements() / addendLen;

    assert(actsBlockCount <= maxBlockCount);

    graph.connect(v["data"][num], acts);
    graph.connect(v["B"][num], addend);

    uint16_t actsBlockCount16 = actsBlockCount;
    assert(actsBlockCount16 == actsBlockCount);

    graph.setInitialValue(v["BLen"][num], addendLen);
    graph.setInitialValue(v["dataBlockCount"][num], actsBlockCount16);

    ++num;
  };

  // Add a sub-vector to operate on, handling splitting the addend into
  // multiple pieces if necessary.
  auto addPiece = [&](const Tensor &acts, const Tensor &addend) {
    auto addendLen = addend.numElements();
    assert(acts.numElements() % addendLen == 0);

    // Group into addendLen pieces.
    auto actsRegrouped =
        acts.reshape({acts.numElements() / addendLen, addendLen});

    std::size_t consumedAddend = 0;
    while (consumedAddend < addendLen) {
      auto thisAddendLen = std::min(addendLen - consumedAddend, maxAddendLen);

      // Extract part of the addend. This could end up being inefficient.
      addShortPiece(
          actsRegrouped.slice(consumedAddend, consumedAddend + thisAddendLen, 1)
              .flatten(),
          addend.slice(consumedAddend, consumedAddend + thisAddendLen));
      consumedAddend += thisAddendLen;
    }
  };

  for (const auto &interval : groupsForWorker) {
    // The first and last groups in this interval.
    const auto begin = interval.begin();
    const auto end = interval.end();
    const auto last = end - 1;

    // Get the unflattened indices. That is: [G][C1][N*...].
    auto beginIndices = poputil::unflattenIndex(firstInGroup.shape(), begin);
    auto lastIndices = poputil::unflattenIndex(firstInGroup.shape(), last);

    // Loop through the C1 group dimension.
    for (unsigned g = beginIndices[0]; g <= lastIndices[0]; ++g) {

      // Get the first batch, which may be part way through this conv group
      // if this is the first conv group (otherwise it is just batch 0).
      unsigned batchBegin = g == beginIndices[0] ? beginIndices[1] : 0;
      // Similarly for the last batch.
      unsigned batchLast =
          g == lastIndices[0] ? lastIndices[1] : firstInGroup.dim(1) - 1;

      // The part of the addend that is added to this channel group.
      auto addendWindow = addendByGroup[g];

      // Loop through the N batch dimension.
      for (unsigned b = batchBegin; b <= batchLast; ++b) {
        // Get the first channel group index, which may be part way through
        // this batch if this is the first batch in the first group.
        unsigned begin =
            g == beginIndices[0] && b == beginIndices[1] ? beginIndices[2] : 0;
        unsigned last = g == lastIndices[0] && b == lastIndices[1]
                            ? lastIndices[2]
                            : firstInGroup.dim(2) - 1;

        auto actsWindow = acts[g][b].flatten().slice(
            begin * outChansPerGroup, (last + 1) * outChansPerGroup);

        addPiece(actsWindow, addendWindow);
      }
    }
  }

  if (scale != 1.0f) {
    graph.setInitialValue(v["scale"], scale);
  }
  graph.setInitialValue(v["n"], num);
  graph.setFieldSize(v["B"], num);
  graph.setFieldSize(v["BLen"], num);
  graph.setFieldSize(v["data"], num);
  graph.setFieldSize(v["dataBlockCount"], num);
}

// Add a VectorInner vertex to implement the ChannelMul function, but if acts
// is too long, use multiple vertices.
// This is to simplify the codelet assembly because `rpt` can only
// loop up to 4095 times (the actual number is hardware-dependent).
/// This uses a supervisor vertexthat splits up the work between workers
// according to actsBlockCountPacked.
void addVectorInnerMulSupervisorVertex(Graph &graph, ComputeSet &cs,
                                       const std::string &templateVertexName,
                                       const Tensor &acts,
                                       const Tensor &actsOut,
                                       const Tensor &scale, unsigned tile) {
  if (graph.getTarget().getNumWorkerContexts() != 6)
    throw poplibs_error("not implemented for IPUs without 6 worker contexts");

  auto scaleLen = scale.numElements();
  auto actsBlockCount = acts.numElements() / scaleLen;

  auto actsFlat = acts.flatten();
  auto actsOutFlat = actsOut.flatten();

  auto maxBlockCount = (graph.getTarget().getRptCountMax() & ~1UL) * 6;

  std::size_t consumedBlocks = 0;

  while (consumedBlocks < actsBlockCount) {
    // The number of blocks is limited by the maximum repeat count for
    // `rpt` (getRptCountMax()). For example if it is 4095 We can do up to
    // 6 * 4094 blocks per vertex. We can't do 6 * 4094 + 2 blocks because
    // that would result in workers having the following numbers of blocks:
    //
    // 4095 4095 4094 4094 4094 4094
    //
    // Which would then be rounded to make them even to:
    //
    // 4094 4096 4094 4094 4094 4094
    //
    // And the second worker has too many blocks for the `rpt`. In fact
    // we can do 6 * 4094 + 1 blocks but let's keep things simple.
    auto thisBlockCount =
        std::min(actsBlockCount - consumedBlocks, maxBlockCount);

    auto actsSlice =
        actsFlat.slice(consumedBlocks * scaleLen,
                       (consumedBlocks + thisBlockCount) * scaleLen);
    auto actsOutSlice =
        actsOutFlat.slice(consumedBlocks * scaleLen,
                          (consumedBlocks + thisBlockCount) * scaleLen);

    auto v = graph.addVertex(
        cs, templateVertexName,
        {{"data", actsSlice}, {"out", actsOutSlice}, {"B", scale}});

    auto actsBlockCountPacked =
        ((thisBlockCount / 6) << 3) | (thisBlockCount % 6);

    uint16_t actsBlockCountPacked16 = actsBlockCountPacked;
    assert(actsBlockCountPacked16 == actsBlockCountPacked);

    graph.setInitialValue(v["dataBlockCountPacked"], actsBlockCountPacked16);

    graph.setTileMapping(v, tile);

    consumedBlocks += thisBlockCount;
  }
};

// Add a VectorInner2D vertex to implement the ChannelMul function, but if any
// of various length constraints are violated, use multiple vertices.
//
// acts             - The grouped activations
// actsOut          - The output tensor which has the same shape as `acts`.
// firstInGroup     - The slice of `acts` that contains the first element from
//                    each channel group.
// scaleByGroup     - The scale as a 2D tensor grouped into channel groups.
// outChansPerGroup - The size of each channel group (e.g. 8 or 16).
// groupsForWorker  - The groups assigned to this worker. These are
//                    intervals on firstInGroup.
// maxBlockCount    - The maximum block count for a vertex. This is only used
//                    in an assertion. The blocks should be split between
//                    multiple vertices outside this function if necessary.
// maxAddendLen     - The maximum length for an addend. If the addend is
//                    longer than this it will be split into multiple vertices
//                    inside this function.
//
void addVectorInnerMul2DVertex(
    Graph &graph, ComputeSet &cs, const std::string &templateVertexName,
    const Tensor &acts, const Tensor &actsOut, const Tensor &firstInGroup,
    const Tensor &scaleByGroup,
    const std::vector<poplar::Interval> &groupsForWorker, unsigned tile,
    std::size_t maxBlockCount, std::size_t maxScaleLen) {
  assert(acts.rank() >= 3);
  assert(firstInGroup.rank() == 3);
  assert(scaleByGroup.rank() == 2);
  assert(firstInGroup.dim(0) == acts.dim(0));
  assert(firstInGroup.dim(1) == acts.dim(1));
  assert(scaleByGroup.dim(0) == acts.dim(0));
  assert(scaleByGroup.dim(1) == acts.dim(acts.rank() - 1));
  assert(acts.shape() == actsOut.shape());

  const std::size_t outChansPerGroup = scaleByGroup.dim(1);

  // We have the following limitations for BroadcastVectorInner2D<MULTIPLY>
  //
  // * dataBlockCount cannot be more than Target::getRptCountMax() because it
  //   is used in a `rpt` loop.
  // * BLen and dataBlockCount cannot be greater than 2^16-1 because
  //   their lengths are stored in uint16_t's.

  VertexRef v = graph.addVertex(cs, templateVertexName);
  graph.setTileMapping(v, tile);
  unsigned num = 0;

  // Add a sub-vector to operate on, handling splitting it into multiple
  // vertices as necessary. The addend length must be <= maxScaleLen,
  // and the block count must be <= maxBlockCount.
  auto addShortPiece = [&](const Tensor &acts, const Tensor &actsOut,
                           const Tensor &scale) {
    auto scaleLen = scale.numElements();

    assert(scaleLen <= maxScaleLen);
    assert(acts.numElements() % scaleLen == 0);
    assert(actsOut.numElements() == acts.numElements());

    auto actsBlockCount = acts.numElements() / scaleLen;

    assert(actsBlockCount <= maxBlockCount);

    graph.connect(v["data"][num], acts);
    graph.connect(v["out"][num], actsOut);
    graph.connect(v["B"][num], scale);

    uint16_t actsBlockCount16 = actsBlockCount;
    assert(actsBlockCount16 == actsBlockCount);

    graph.setInitialValue(v["BLen"][num], scaleLen);
    graph.setInitialValue(v["dataBlockCount"][num], actsBlockCount16);

    ++num;
  };

  // Add a sub-vector to operate on, handling splitting the addend into
  // multiple pieces if necessary.
  auto addPiece = [&](const Tensor &acts, const Tensor &actsOut,
                      const Tensor &scale) {
    auto scaleLen = scale.numElements();
    assert(acts.numElements() % scaleLen == 0);

    // Group into addendLen pieces.
    auto actsRegrouped =
        acts.reshape({acts.numElements() / scaleLen, scaleLen});
    auto actsOutRegrouped =
        actsOut.reshape({actsOut.numElements() / scaleLen, scaleLen});

    std::size_t consumedScale = 0;
    while (consumedScale < scaleLen) {
      auto thisScaleLen = std::min(scaleLen - consumedScale, maxScaleLen);

      // Extract part of the addend. This could end up being inefficient.
      addShortPiece(
          actsRegrouped.slice(consumedScale, consumedScale + thisScaleLen, 1)
              .flatten(),
          actsOutRegrouped.slice(consumedScale, consumedScale + thisScaleLen, 1)
              .flatten(),
          scale.slice(consumedScale, consumedScale + thisScaleLen));
      consumedScale += thisScaleLen;
    }
  };

  for (const auto &interval : groupsForWorker) {
    // The first and last groups in this interval.
    const auto begin = interval.begin();
    const auto end = interval.end();
    const auto last = end - 1;

    // Get the unflattened indices. That is: [G][C1][N*...].
    auto beginIndices = poputil::unflattenIndex(firstInGroup.shape(), begin);
    auto lastIndices = poputil::unflattenIndex(firstInGroup.shape(), last);

    // Loop through the C1 group dimension.
    for (unsigned g = beginIndices[0]; g <= lastIndices[0]; ++g) {

      // Get the first batch, which may be part way through this conv group
      // if this is the first conv group (otherwise it is just batch 0).
      unsigned batchBegin = g == beginIndices[0] ? beginIndices[1] : 0;
      // Similarly for the last batch.
      unsigned batchLast =
          g == lastIndices[0] ? lastIndices[1] : firstInGroup.dim(1) - 1;

      // The part of the addend that is added to this channel group.
      auto scaleWindow = scaleByGroup[g];

      // Loop through the N batch dimension.
      for (unsigned b = batchBegin; b <= batchLast; ++b) {
        // Get the first channel group index, which may be part way through
        // this batch if this is the first batch in the first group.
        unsigned begin =
            g == beginIndices[0] && b == beginIndices[1] ? beginIndices[2] : 0;
        unsigned last = g == lastIndices[0] && b == lastIndices[1]
                            ? lastIndices[2]
                            : firstInGroup.dim(2) - 1;

        auto actsWindow = acts[g][b].flatten().slice(
            begin * outChansPerGroup, (last + 1) * outChansPerGroup);
        auto actsOutWindow = actsOut[g][b].flatten().slice(
            begin * outChansPerGroup, (last + 1) * outChansPerGroup);
        addPiece(actsWindow, actsOutWindow, scaleWindow);
      }
    }
  }

  graph.setInitialValue(v["n"], num);
  graph.setFieldSize(v["B"], num);
  graph.setFieldSize(v["BLen"], num);
  graph.setFieldSize(v["data"], num);
  graph.setFieldSize(v["out"], num);
  graph.setFieldSize(v["dataBlockCount"], num);
}

} // anonymous namespace

void broadcastAddVectorInnermostInPlace(Graph &graph, const Tensor &acts,
                                        const Tensor &addendByGroup,
                                        const float scale, ComputeSet &cs) {

  const auto outChansPerGroup = acts.dim(acts.rank() - 1);
  const auto dType = acts.elementType();
  const auto &target = graph.getTarget();
  // Get the first element in each C2-sized group, and flatten so that the
  // final shape is [C1][N][...].
  const auto firstInGroup =
      acts.slice(0, 1, acts.rank() - 1).flatten(2, acts.rank());
  const auto firstInGroupMapping = graph.getTileMapping(firstInGroup);
  const unsigned numTiles = firstInGroupMapping.size();

  expr::BroadcastOpType op = (scale != 1.0f) ? expr::BroadcastOpType::SCALED_ADD
                                             : expr::BroadcastOpType::ADD;

  auto templateVertexName = templateVertex(
      "popops::BroadcastVectorInnerInPlaceSupervisor", op, dType);
  auto templateVertexName2D =
      templateVertex("popops::BroadcastVectorInner2DInPlace", op, dType);

  // Limits for the 2D vertex.
  const auto maxBlockCount = std::min<unsigned>(
      graph.getMaxVertexFieldValue(templateVertexName2D, "dataBlockCount"),
      target.getRptCountMax());
  const auto maxAddendLen =
      graph.getMaxVertexFieldValue(templateVertexName2D, "BLen");

  for (unsigned tile = 0; tile != numTiles; ++tile) {

    const auto singleGroup = getAssignedGroupForTile(firstInGroupMapping[tile],
                                                     firstInGroup.shape());
    // singleGroup.second is true if all elements mapped to this tile are in
    // the same channel group. I.e. they only require one addendByGroup[x].
    //
    // If this is the case, then singleGroup.first is the value of `x`, i.e.
    // the group that they are all in.
    //
    // In this case we use VectorInnerSupervisor<ADD> otherwise we use
    // VectorInner2D<ADD>.
    if (singleGroup.second) {
      std::vector<Interval> actsSlices;
      for (const auto &t : firstInGroupMapping[tile]) {
        actsSlices.emplace_back(t.begin() * outChansPerGroup,
                                t.end() * outChansPerGroup);
      }
      auto vActs = concat(acts.flatten().slices(actsSlices));
      auto vAddend = addendByGroup[singleGroup.first];

      addVectorInnerAddSupervisorVertex(graph, cs, templateVertexName, vActs,
                                        vAddend, scale, tile);

    } else {
      // We have elements from multiple groups on this tile. Split the groups
      // between workers. The size corresponds to actsBlockSize which
      // we limit to maxBlockCount.
      const auto perWorkerGroups = splitRegionsBetweenWorkers(
          target, firstInGroupMapping[tile], 1, 0, maxBlockCount);
      for (const auto &groupsForWorker : perWorkerGroups) {
        // Add a vertex for this worker.
        addVectorInnerAdd2DVertex(graph, cs, templateVertexName2D, acts,
                                  firstInGroup, addendByGroup, groupsForWorker,
                                  scale, tile, maxBlockCount, maxAddendLen);
      }
    }
  }
}

void broadcastMulVectorInnermost(Graph &graph, const Tensor &acts,
                                 const Tensor &actsOut,
                                 const Tensor &scaleByGroup, ComputeSet &cs,
                                 const std::string &fnPrefix) {

  const auto outChansPerGroup = acts.dim(acts.rank() - 1);

  if (scaleByGroup.rank() != 2)
    throw poputil::poplibs_error(
        "popopsBroadcastVectorInner<MULTIPLY> requires scale to be a"
        " rank 2 tensor");

  if (scaleByGroup.dim(0) != acts.dim(0))
    throw poputil::poplibs_error(
        "popopsBroadcastVectorInner<MULTIPLY> requires scale, acts"
        " outermost dimensions to match");

  if (scaleByGroup.dim(1) != outChansPerGroup)
    throw poputil::poplibs_error(
        "popopsBroadcastVectorInner<MULTIPLY> requires scale innermost"
        " dimension to match output outermost dimension");

  const auto dType = acts.elementType();
  const auto &target = graph.getTarget();

  const auto firstInGroup =
      acts.slice(0, 1, acts.rank() - 1).flatten(2, acts.rank());
  const auto firstInGroupMapping = graph.getTileMapping(firstInGroup);
  const unsigned numTiles = firstInGroupMapping.size();

  auto templateVertexName =
      templateVertex("popops::BroadcastVectorInnerSupervisor",
                     expr::BroadcastOpType::MULTIPLY, dType);
  auto templateVertexName2D = templateVertex(
      "popops::BroadcastVectorInner2D", expr::BroadcastOpType::MULTIPLY, dType);

  // Limits for the 2D vertex.
  const auto maxBlockCount = std::min<unsigned>(
      graph.getMaxVertexFieldValue(templateVertexName2D, "dataBlockCount"),
      target.getRptCountMax());
  const auto maxScaleLen =
      graph.getMaxVertexFieldValue(templateVertexName2D, "BLen");

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto singleGroup = getAssignedGroupForTile(firstInGroupMapping[tile],
                                                     firstInGroup.shape());
    // singleGroup.second is true if all elements mapped to this tile are in
    // the same channel group. I.e. they only require one scaleByGroup[x].
    //
    // If this is the case, then singleGroup.first is the value of `x`, i.e.
    // the group that they are all in.
    //
    // In this case we use VectorInnerSupervisor<Multiply> otherwise we use
    // VectorInner2D<MULTIPLY>.
    if (singleGroup.second) {
      std::vector<Interval> actsSlices;
      for (const auto &t : firstInGroupMapping[tile]) {
        actsSlices.emplace_back(t.begin() * outChansPerGroup,
                                t.end() * outChansPerGroup);
      }
      auto vActs = concat(acts.flatten().slices(actsSlices));
      auto vActsOut = concat(actsOut.flatten().slices(actsSlices));
      auto vScale = scaleByGroup[singleGroup.first];

      addVectorInnerMulSupervisorVertex(graph, cs, templateVertexName, vActs,
                                        vActsOut, vScale, tile);
    } else {
      // We have elements from multiple groups on this tile. Split the groups
      // between workers. The size corresponds to actsBlockSize which
      // we limit to maxBlockCount.
      const auto perWorkerGroups = splitRegionsBetweenWorkers(
          target, firstInGroupMapping[tile], 1, 0, maxBlockCount);
      for (const auto &groupsForWorker : perWorkerGroups) {
        // Add a vertex for this worker.
        addVectorInnerMul2DVertex(
            graph, cs, templateVertexName2D, acts, actsOut, firstInGroup,
            scaleByGroup, groupsForWorker, tile, maxBlockCount, maxScaleLen);
      }
    }
  }
  return;
}

} // namespace popops
