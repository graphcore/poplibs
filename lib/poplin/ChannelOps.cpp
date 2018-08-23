#include "ChannelOps.hpp"

#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

#include "ConvUtilInternal.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace poplin {

namespace {

// Returns a pair with second element set to true if a single channel group is
// assigned in all the intervals assigned to a tile. The first element is the
// group number
std::pair<std::size_t, bool>
getAssignedGroupForTile(const std::vector<Interval> &tileMap,
                        const std::vector<std::size_t> &firstInGroupShape)  {
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

// Add an AddToChannel vertex, but if acts is too long, use multiple vertices.
// This is to simplify the codelet assembly because `rpt` can only
// loop up to 4095 times (the actual number is hardware-dependent).
// AddToChannel is a supervisor vertex that splits up the work between workers
// according to actsBlockCountPacked.
void addAddToChannelSupervisorVertex(Graph &graph,
                                     ComputeSet &cs,
                                     const std::string &vertexName,
                                     const Type &dType,
                                     const Tensor &acts,
                                     const Tensor &addend,
                                     float scale,
                                     unsigned tile) {
  if (graph.getTarget().getNumWorkerContexts() != 6)
    throw poplib_error("not implemented for IPUs without 6 worker contexts");

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
    auto thisBlockCount = std::min(actsBlockCount - consumedBlocks,
                                   maxBlockCount);

    auto actsSlice = actsFlat.slice(consumedBlocks * addendLen,
                                    (consumedBlocks + thisBlockCount)
                                      * addendLen);

    auto v = graph.addVertex(cs, templateVertex(vertexName, dType),
                             {{"acts", actsSlice},
                              {"addend", addend}});

    auto actsBlockCountPacked = ((thisBlockCount / 6) << 3)
                                | (thisBlockCount % 6);

    uint16_t actsBlockCountPacked16 = actsBlockCountPacked;
    assert(actsBlockCountPacked16 == actsBlockCountPacked);

    graph.setInitialValue(v["actsBlockCountPacked"], actsBlockCountPacked16);

    if (scale != 1.0)
        graph.setInitialValue(v["scale"], scale);

    graph.setTileMapping(v, tile);

    consumedBlocks += thisBlockCount;
  }

};

// Add an AddToChannel vertex, but if any of various length constraints are
// violated, use multiple vertices.
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
void addAddToChannel2DVertex(Graph &graph,
                             ComputeSet &cs,
                             const std::string &vertexName,
                             const Type &dType,
                             const Tensor &acts,
                             const Tensor &firstInGroup,
                             const Tensor &addendByGroup,
                             const std::vector<poplar::Interval>
                                 &groupsForWorker,
                             float scale,
                             unsigned tile,
                             std::size_t maxBlockCount,
                             std::size_t maxAddendLen) {
  assert(acts.rank() >= 3);
  assert(firstInGroup.rank() == 3);
  assert(addendByGroup.rank() == 2);
  assert(firstInGroup.dim(0) == acts.dim(0));
  assert(firstInGroup.dim(1) == acts.dim(1));
  assert(addendByGroup.dim(0) == acts.dim(0));
  assert(addendByGroup.dim(1) == acts.dim(acts.rank()-1));

  const std::size_t outChansPerGroup = addendByGroup.dim(1);

  // We have the following limitations for AddToChannel2D
  //
  // * actsBlockCount cannot be more than Target::getRptCountMax() because it
  //   is used in a `rpt` loop.
  // * addendLen and actsBlockCount cannot be greater than 2^16-1 because
  //   their lengths are stored in uint16_t's.

  VertexRef v = graph.addVertex(cs, templateVertex(vertexName, dType));
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

    graph.connect(v["acts"][num], acts);
    graph.connect(v["addend"][num], addend);

    uint16_t actsBlockCount16 = actsBlockCount;
    assert(actsBlockCount16 == actsBlockCount);

    graph.setInitialValue(v["addendLen"][num], addendLen);
    graph.setInitialValue(v["actsBlockCount"][num], actsBlockCount16);

    ++num;
  };

  // Add a sub-vector to operate on, handling splitting the addend into
  // multiple pieces if necessary.
  auto addPiece = [&](const Tensor &acts, const Tensor &addend) {

    auto addendLen = addend.numElements();
    assert(acts.numElements() % addendLen == 0);

    // Group into addendLen pieces.
    auto actsRegrouped = acts.reshape({acts.numElements() / addendLen,
                                       addendLen});

    std::size_t consumedAddend = 0;
    while (consumedAddend < addendLen) {
      auto thisAddendLen = std::min(addendLen - consumedAddend, maxAddendLen);

      // Extract part of the addend. This could end up being inefficient.
      addShortPiece(actsRegrouped.slice(consumedAddend,
                                        consumedAddend + thisAddendLen,
                                        1).flatten(),
                    addend.slice(consumedAddend,
                                 consumedAddend + thisAddendLen));
      consumedAddend += thisAddendLen;
    }
  };

  for (const auto &interval : groupsForWorker) {
    // The first and last groups in this interval.
    const auto begin = interval.begin();
    const auto end = interval.end();
    const auto last = end - 1;

    // Get the unflattened indices. That is: [G][C1][N*...].
    auto beginIndices = poputil::unflattenIndex(firstInGroup.shape(),
                                                begin);
    auto lastIndices = poputil::unflattenIndex(firstInGroup.shape(),
                                               last);

    // Loop through the C1 group dimension.
    for (unsigned g = beginIndices[0]; g <= lastIndices[0]; ++g) {

      // Get the first batch, which may be part way through this conv group
      // if this is the first conv group (otherwise it is just batch 0).
      unsigned batchBegin = g == beginIndices[0] ?
                              beginIndices[1] : 0;
      // Similarly for the last batch.
      unsigned batchLast = g == lastIndices[0] ?
                             lastIndices[1] : firstInGroup.dim(1) - 1;

      // The part of the addend that is added to this channel group.
      auto addendWindow = addendByGroup[g];

      // Loop through the N batch dimension.
      for (unsigned b = batchBegin; b <= batchLast; ++b) {
        // Get the first channel group index, which may be part way through
        // this batch if this is the first batch in the first group.
        unsigned begin = g == beginIndices[0] && b == beginIndices[1] ?
                           beginIndices[2] : 0;
        unsigned last = g == lastIndices[0] && b == lastIndices[1] ?
                          lastIndices[2] : firstInGroup.dim(2) - 1;

        auto actsWindow =
            acts[g][b].flatten().slice(begin * outChansPerGroup,
                                       (last + 1) * outChansPerGroup);

        addPiece(actsWindow, addendWindow);
      }
    }
  }

  if (scale != 1.0f) {
    graph.setInitialValue(v["scale"], scale);
  }
  graph.setInitialValue(v["n"], num);
  graph.setFieldSize(v["addend"], num);
  graph.setFieldSize(v["addendLen"], num);
  graph.setFieldSize(v["acts"], num);
  graph.setFieldSize(v["actsBlockCount"], num);

}


// Add a ChannelMul vertex, but if acts is too long, use multiple vertices.
// This is to simplify the codelet assembly because `rpt` can only
// loop up to 4095 times (the actual number is hardware-dependent).
// ChannelMul is a supervisor vertex that splits up the work between workers
// according to actsBlockCountPacked.
void addChannelMulSupervisorVertex(Graph &graph,
                                   ComputeSet &cs,
                                   const std::string &vertexName,
                                   const Type &dType,
                                   const Tensor &acts,
                                   const Tensor &actsOut,
                                   const Tensor &scale,
                                   unsigned tile) {
  if (graph.getTarget().getNumWorkerContexts() != 6)
    throw poplib_error("not implemented for IPUs without 6 worker contexts");

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
    auto thisBlockCount = std::min(actsBlockCount - consumedBlocks,
                                   maxBlockCount);

    auto actsSlice = actsFlat.slice(consumedBlocks * scaleLen,
                                    (consumedBlocks + thisBlockCount)
                                      * scaleLen);
    auto actsOutSlice = actsFlat.slice(consumedBlocks * scaleLen,
                                       (consumedBlocks + thisBlockCount)
                                         * scaleLen);

    auto v = graph.addVertex(cs, templateVertex(vertexName, dType),
                             {{"actsIn", actsSlice},
                              {"actsOut", actsOutSlice},
                              {"scale", scale}});

    auto actsBlockCountPacked = ((thisBlockCount / 6) << 3)
                                | (thisBlockCount % 6);

    uint16_t actsBlockCountPacked16 = actsBlockCountPacked;
    assert(actsBlockCountPacked16 == actsBlockCountPacked);

    graph.setInitialValue(v["actsBlockCountPacked"], actsBlockCountPacked16);

    graph.setTileMapping(v, tile);

    consumedBlocks += thisBlockCount;
  }

};

// Add a ChannelMul vertex, but if any of various length constraints are
// violated, use multiple vertices.
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
void addChannelMul2DVertex(Graph &graph,
                           ComputeSet &cs,
                           const std::string &vertexName,
                           const Type &dType,
                           const Tensor &acts,
                           const Tensor &actsOut,
                           const Tensor &firstInGroup,
                           const Tensor &scaleByGroup,
                           const std::vector<poplar::Interval> &groupsForWorker,
                           unsigned tile,
                           std::size_t maxBlockCount,
                           std::size_t maxScaleLen) {
  assert(acts.rank() >= 3);
  assert(firstInGroup.rank() == 3);
  assert(scaleByGroup.rank() == 2);
  assert(firstInGroup.dim(0) == acts.dim(0));
  assert(firstInGroup.dim(1) == acts.dim(1));
  assert(scaleByGroup.dim(0) == acts.dim(0));
  assert(scaleByGroup.dim(1) == acts.dim(acts.rank()-1));
  assert(acts.shape() == actsOut.shape());

  const std::size_t outChansPerGroup = scaleByGroup.dim(1);

  // We have the following limitations for ChannelMul
  //
  // * actsBlockCount cannot be more than Target::getRptCountMax() because it
  //   is used in a `rpt` loop.
  // * scaleLen and actsBlockCount cannot be greater than 2^16-1 because
  //   their lengths are stored in uint16_t's.

  VertexRef v = graph.addVertex(cs, templateVertex(vertexName, dType));
  graph.setTileMapping(v, tile);
  unsigned num = 0;

  // Add a sub-vector to operate on, handling splitting it into multiple
  // vertices as necessary. The addend length must be <= maxScaleLen,
  // and the block count must be <= maxBlockCount.
  auto addShortPiece = [&](const Tensor &acts,
                           const Tensor &actsOut,
                           const Tensor &scale) {
    auto scaleLen = scale.numElements();

    assert(scaleLen <= maxScaleLen);
    assert(acts.numElements() % scaleLen == 0);
    assert(actsOut.numElements() == acts.numElements());

    auto actsBlockCount = acts.numElements() / scaleLen;

    assert(actsBlockCount <= maxBlockCount);

    graph.connect(v["actsIn"][num], acts);
    graph.connect(v["actsOut"][num], actsOut);
    graph.connect(v["scale"][num], scale);

    uint16_t actsBlockCount16 = actsBlockCount;
    assert(actsBlockCount16 == actsBlockCount);

    graph.setInitialValue(v["scaleLen"][num], scaleLen);
    graph.setInitialValue(v["actsBlockCount"][num], actsBlockCount16);

    ++num;
  };

  // Add a sub-vector to operate on, handling splitting the addend into
  // multiple pieces if necessary.
  auto addPiece = [&](const Tensor &acts,
                      const Tensor &actsOut,
                      const Tensor &scale) {

    auto scaleLen = scale.numElements();
    assert(acts.numElements() % scaleLen == 0);

    // Group into addendLen pieces.
    auto actsRegrouped = acts.reshape({acts.numElements() / scaleLen,
                                       scaleLen});
    auto actsOutRegrouped = actsOut.reshape({actsOut.numElements() / scaleLen,
                                             scaleLen});

    std::size_t consumedScale = 0;
    while (consumedScale < scaleLen) {
      auto thisScaleLen = std::min(scaleLen - consumedScale, maxScaleLen);

      // Extract part of the addend. This could end up being inefficient.
      addShortPiece(actsRegrouped.slice(consumedScale,
                                        consumedScale + thisScaleLen,
                                        1).flatten(),
                    actsOutRegrouped.slice(consumedScale,
                                           consumedScale + thisScaleLen,
                                           1).flatten(),
                    scale.slice(consumedScale,
                                consumedScale + thisScaleLen));
      consumedScale += thisScaleLen;
    }
  };

  for (const auto &interval : groupsForWorker) {
    // The first and last groups in this interval.
    const auto begin = interval.begin();
    const auto end = interval.end();
    const auto last = end - 1;

    // Get the unflattened indices. That is: [G][C1][N*...].
    auto beginIndices = poputil::unflattenIndex(firstInGroup.shape(),
                                                begin);
    auto lastIndices = poputil::unflattenIndex(firstInGroup.shape(),
                                               last);

    // Loop through the C1 group dimension.
    for (unsigned g = beginIndices[0]; g <= lastIndices[0]; ++g) {

      // Get the first batch, which may be part way through this conv group
      // if this is the first conv group (otherwise it is just batch 0).
      unsigned batchBegin = g == beginIndices[0] ?
                              beginIndices[1] : 0;
      // Similarly for the last batch.
      unsigned batchLast = g == lastIndices[0] ?
                             lastIndices[1] : firstInGroup.dim(1) - 1;

      // The part of the addend that is added to this channel group.
      auto scaleWindow = scaleByGroup[g];

      // Loop through the N batch dimension.
      for (unsigned b = batchBegin; b <= batchLast; ++b) {
        // Get the first channel group index, which may be part way through
        // this batch if this is the first batch in the first group.
        unsigned begin = g == beginIndices[0] && b == beginIndices[1] ?
                           beginIndices[2] : 0;
        unsigned last = g == lastIndices[0] && b == lastIndices[1] ?
                          lastIndices[2] : firstInGroup.dim(2) - 1;

        auto actsWindow =
            acts[g][b].flatten().slice(begin * outChansPerGroup,
                                       (last + 1) * outChansPerGroup);
        auto actsOutWindow =
            actsOut[g][b].flatten().slice(begin * outChansPerGroup,
                                          (last + 1) * outChansPerGroup);
        addPiece(actsWindow, actsOutWindow, scaleWindow);
      }
    }
  }

  graph.setInitialValue(v["n"], num);
  graph.setFieldSize(v["scale"], num);
  graph.setFieldSize(v["scaleLen"], num);
  graph.setFieldSize(v["actsIn"], num);
  graph.setFieldSize(v["actsOut"], num);
  graph.setFieldSize(v["actsBlockCount"], num);

}



} // anonymous namespace

void addToChannel(Graph &graph,
                  const Tensor &actsUngrouped,
                  const Tensor &addend,
                  float scale,
                  Sequence &prog,
                  const std::string debugPrefix) {

  const auto fnPrefix = debugPrefix + "/addToChannel";
  auto cs = graph.addComputeSet(fnPrefix);

  // Convert actsUngrouped back into its internal layout, which matches
  // the in-memory layout. It is [G][C1][N]...[C2] where C2 is a nice
  // number like 8 or 16. N is the batch dimension, ... are the spatial
  // dimensions, G is the conv group dimension and C1 is the remaining channel
  // dimensions. Also, the [G] dimension is removed because it is 1, so the
  // shape is now [C1][N]...[C2]

  const auto acts =
      splitActivationChanGroups(
        actsToInternalShape(actsUngrouped, 1)
      )[0];

  const auto dType = acts.elementType();
  const auto &target = graph.getTarget();
  // The number of channels in the inner-most dimension (i.e. adjacent in
  // memory). This is C2.
  const auto outChansPerGroup = acts.dim(acts.rank() - 1);
  // Reshape addend so that addendByGroup[i] is the i'th C2-sized group. The
  // shape should be [C1][C2].
  const auto addendByGroup =
      addend.reshape({addend.numElements() / outChansPerGroup,
                      outChansPerGroup});

  assert(addendByGroup.rank() == 2);
  assert(addendByGroup.dim(0) == acts.dim(0));
  assert(addendByGroup.dim(1) == outChansPerGroup);

  // Get the first element in each C2-sized group, and flatten so that the
  // final shape is [C1][N][...].
  const auto firstInGroup = acts.slice(0, 1, acts.rank() - 1)
                                .flatten(2, acts.rank());
  const auto firstInGroupMapping = graph.getTileMapping(firstInGroup);
  const unsigned numTiles = firstInGroupMapping.size();

  const std::string vertexName = scale == 1.0f ? "poplin::AddToChannel"
                                               : "poplin::ScaledAddToChannel";
  const std::string vertexName2D = vertexName + "2D";
  const std::string templateVertexName2D = templateVertex(vertexName2D, dType);

  // Limits for the 2D vertex.
  const auto maxBlockCount = std::min<unsigned>(
      graph.getMaxVertexFieldValue(templateVertexName2D, "actsBlockCount"),
      target.getRptCountMax()
    );
  const auto maxAddendLen =
      graph.getMaxVertexFieldValue(templateVertexName2D, "addendLen");

  for (unsigned tile = 0; tile != numTiles; ++tile) {

    const auto singleGroup = getAssignedGroupForTile(firstInGroupMapping[tile],
                                                     firstInGroup.shape());
    // singleGroup.second is true if all elements mapped to this tile are in
    // the same channel group. I.e. they only require one addendByGroup[x].
    //
    // If this is the case, then singleGroup.first is the value of `x`, i.e.
    // the group that they are all in.
    //
    // In this case we use AddToChannel otherwise we use AddToChannel2D.
    if (singleGroup.second) {
      std::vector<Interval> actsSlices;
      for (const auto &t : firstInGroupMapping[tile]) {
          actsSlices.emplace_back(t.begin() * outChansPerGroup,
                                  t.end() * outChansPerGroup);
      }
      auto vActs = concat(acts.flatten().slices(actsSlices));
      auto vAddend = addendByGroup[singleGroup.first];

      addAddToChannelSupervisorVertex(graph, cs, vertexName, dType,
                            vActs, vAddend, scale, tile);

    } else {
      // We have elements from multiple groups on this tile. Split the groups
      // between workers. The size corresponds to actsBlockSize which
      // we limit to maxBlockCount.
      const auto perWorkerGroups =
          splitRegionsBetweenWorkers(target, firstInGroupMapping[tile],
                                     1, 0,
                                     maxBlockCount);
      for (const auto &groupsForWorker : perWorkerGroups) {
        // Add a vertex for this worker.
        addAddToChannel2DVertex(graph, cs, vertexName2D, dType,
                                acts,
                                firstInGroup,
                                addendByGroup,
                                groupsForWorker,
                                scale, tile,
                                maxBlockCount, maxAddendLen);
      }
    }
  }
  prog.add(Execute(cs));
}


Tensor channelMul(Graph &graph,
                  const Tensor &actsUngrouped,
                  const Tensor &scale,
                  Sequence &prog,
                  const std::string &debugPrefix) {

  const auto fnPrefix = debugPrefix + "/channelMul";
  auto cs = graph.addComputeSet(fnPrefix);

  auto actsOutUngrouped = graph.clone(actsUngrouped, fnPrefix + "/actsIn");
  const auto acts =
      splitActivationChanGroups(
        actsToInternalShape(actsUngrouped, 1)
      )[0];
  const auto actsOut =
      splitActivationChanGroups(
        actsToInternalShape(actsOutUngrouped, 1)
      )[0];
  const auto dType = acts.elementType();
  const auto &target = graph.getTarget();
  const auto outChansPerGroup = acts.dim(acts.rank() - 1);
  const auto scaleByGroup =
      scale.reshape({scale.numElements() / outChansPerGroup,
                      outChansPerGroup});

  assert(scaleByGroup.rank() == 2);
  assert(scaleByGroup.dim(0) == acts.dim(0));
  assert(scaleByGroup.dim(1) == outChansPerGroup);

  const auto firstInGroup = acts.slice(0, 1, acts.rank() - 1)
                                .flatten(2, acts.rank());
  const auto firstInGroupMapping = graph.getTileMapping(firstInGroup);
  const unsigned numTiles = firstInGroupMapping.size();

  const std::string vertexName = "poplin::ChannelMul";
  const std::string vertexName2D = vertexName + "2D";
  const std::string templateVertexName2D = templateVertex(vertexName2D, dType);

  // Limits for the 2D vertex.
  const auto maxBlockCount = std::min<unsigned>(
      graph.getMaxVertexFieldValue(templateVertexName2D, "actsBlockCount"),
      target.getRptCountMax()
    );
  const auto maxScaleLen =
      graph.getMaxVertexFieldValue(templateVertexName2D, "scaleLen");

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto singleGroup = getAssignedGroupForTile(firstInGroupMapping[tile],
                                                     firstInGroup.shape());
    // singleGroup.second is true if all elements mapped to this tile are in
    // the same channel group. I.e. they only require one scaleByGroup[x].
    //
    // If this is the case, then singleGroup.first is the value of `x`, i.e.
    // the group that they are all in.
    //
    // In this case we use ChannelMul otherwise we use ChannelMul2D.
    if (singleGroup.second) {
      std::vector<Interval> actsSlices;
      for (const auto &t : firstInGroupMapping[tile]) {
          actsSlices.emplace_back(t.begin() * outChansPerGroup,
                                  t.end() * outChansPerGroup);
      }
      auto vActs = concat(acts.flatten().slices(actsSlices));
      auto vActsOut = concat(actsOut.flatten().slices(actsSlices));
      auto vScale = scaleByGroup[singleGroup.first];

      addChannelMulSupervisorVertex(graph, cs, vertexName, dType,
                                    vActs, vActsOut, vScale, tile);
    } else {
      // We have elements from multiple groups on this tile. Split the groups
      // between workers. The size corresponds to actsBlockSize which
      // we limit to maxBlockCount.
      const auto perWorkerGroups =
          splitRegionsBetweenWorkers(target, firstInGroupMapping[tile],
                                     1, 0,
                                     maxBlockCount);
      for (const auto &groupsForWorker : perWorkerGroups) {
        // Add a vertex for this worker.
        addChannelMul2DVertex(graph, cs, vertexName2D, dType,
                              acts, actsOut,
                              firstInGroup,
                              scaleByGroup,
                              groupsForWorker,
                              tile,
                              maxBlockCount, maxScaleLen);
      }
    }
  }
  prog.add(Execute(cs));
  return actsOutUngrouped;
}

} // namespace poplin
