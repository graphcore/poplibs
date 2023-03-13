// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include "poputil/VertexTemplates.hpp"
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/TargetType.hpp>
#include <poplin/experimental/LuFactorization.hpp>
#include <popops/Cast.hpp>
#include <poputil/TileMapping.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <list>
#include <vector>

#include <poplin/codelets.hpp>

using namespace std;
using namespace poplar;
using namespace poplin;
using namespace poplar::program;

namespace poplin {

namespace experimental {

namespace {

inline unsigned alignUp(const float value, const unsigned aligment) {
  return ceil(value / aligment) * aligment;
}

struct LuDataLayout {
  unsigned m;
  unsigned n;
  unsigned blocksInRow;
  unsigned blocksInCol;
  unsigned blockSize;
  unsigned numTiles;
  unsigned minEdge;
  unsigned maxEdge;
  std::vector<unsigned> dataToTile;
};

struct LuCompute {
  unsigned x;
  unsigned y;
  unsigned width;
  unsigned height;
};

struct PrevStage {
  unsigned start;
  unsigned end;
  unsigned depth;
};

// Get Id of IPU tile that is assigned to given data block
unsigned getLUTileId(const LuDataLayout &layout, unsigned beginY,
                     unsigned beginX) {

  const unsigned dataTileIdx =
      ((beginY / layout.blockSize) * layout.blocksInRow +
       (beginX / layout.blockSize));
  assert(dataTileIdx < layout.dataToTile.size());
  return layout.dataToTile[dataTileIdx];
}

// X X R R R R
// X X R R R R
// C C 0 0 0 0
// C C 0 0 0 0
// C C 0 0 0 0
// C C 0 0 0 0
// Update rows (R) at level of (X) and cols (C) bellow (X).
// X coordinates are: {{start,start}, {start + depth, start + depth}}
void updateRowCol(const LuDataLayout &layout, Graph &graph, ComputeSet &cs,
                  Tensor &tensor, unsigned start, unsigned depth) {
  auto sliceCore = tensor.slice({start, start}, {start + depth, start + depth});
  unsigned blockOffset = start + depth;
  while (blockOffset < layout.n) {
    unsigned blockSize =
        layout.blockSize -
        (blockOffset - (blockOffset / layout.blockSize * layout.blockSize));
    blockSize = blockOffset + blockSize < layout.n ? blockSize
                                                   : (layout.n - blockOffset);
    auto sliceLR = tensor.slice({start, blockOffset},
                                {start + depth, blockOffset + blockSize});
    const unsigned alignment = (depth % 2 == 0 && blockSize % 2 == 0 ? 8 : 4);
    auto vRow = graph.addVertex(
        cs, poputil::templateVertex("poplin::experimental::LUDRowVertex",
                                    alignment));
    graph.connect(vRow["sliceLU"], sliceLR);
    graph.connect(vRow["sliceLUCore"], sliceCore);
    graph.setInitialValue(vRow["depth"], depth);
    graph.setInitialValue(vRow["width"], blockSize);
    graph.setTileMapping(vRow, getLUTileId(layout, start, blockOffset));
    blockOffset += blockSize;
  }

  blockOffset = start + depth;
  while (blockOffset < layout.m) {
    unsigned blockSize =
        layout.blockSize -
        (blockOffset - (blockOffset / layout.blockSize * layout.blockSize));
    blockSize = blockOffset + blockSize < layout.m ? blockSize
                                                   : (layout.m - blockOffset);
    auto sliceLC = tensor.slice({blockOffset, start},
                                {blockOffset + blockSize, start + depth});
    const unsigned alignment = (depth % 2 == 0 && blockSize % 2 == 0 ? 8 : 4);
    auto vCol = graph.addVertex(
        cs, poputil::templateVertex("poplin::experimental::LUDColVertex",
                                    alignment));
    graph.connect(vCol["sliceLU"], sliceLC);
    graph.connect(vCol["sliceLUCore"], sliceCore);
    graph.setInitialValue(vCol["depth"], depth);
    graph.setInitialValue(vCol["height"], blockSize);
    graph.setTileMapping(vCol, getLUTileId(layout, blockOffset, start));
    blockOffset += blockSize;
  }
}

// Assign pending computations to IPU tiles and store this information in
// blockMM
void assignBlock(const LuDataLayout &layout, unsigned start, unsigned size,
                 std::vector<std::list<LuCompute>> &blockMM) {

  const unsigned stop = std::min(start + size, layout.n);
  unsigned y = start;
  while (y < layout.m) {
    unsigned height = layout.blockSize - (alignUp(y, layout.blockSize) - y);
    height = y + height < layout.m ? height : (layout.m - y);
    assert(height > 0);
    unsigned x = start;
    unsigned rowWidth = y < stop ? layout.n : stop;
    while (x < rowWidth) {
      unsigned width = layout.blockSize - (alignUp(x, layout.blockSize) - x);
      width = (x + width) < layout.n ? width : (layout.n - x);
      assert(width > 0);
      unsigned tileId = getLUTileId(layout, y, x);
      blockMM[tileId].push_back({x, y, width, height});
      x += width;
    }
    y += height;
  }
}

// Add computations planed in blockMM to graph
void executeBlock(Graph &graph, ComputeSet &cs, Tensor &tensor, unsigned base,
                  unsigned depth, std::vector<std::list<LuCompute>> &blockMM) {

  unsigned tiles = 1;
  while (tiles > 0) {
    tiles = 0;
    for (unsigned tileId = 0; tileId < blockMM.size(); tileId++) {
      if (blockMM[tileId].size() == 0)
        continue;
      tiles++;
      LuCompute &block = blockMM[tileId].front();
      auto sliceLCol =
          tensor.slice({block.y, base}, {block.y + block.height, base + depth});
      auto sliceLRow =
          tensor.slice({base, block.x}, {base + depth, block.x + block.width})
              .transpose();
      auto sliceU = tensor.slice(
          {block.y, block.x}, {block.y + block.height, block.x + block.width});
      const unsigned alignment =
          (block.width % 2 == 0 && block.height % 2 == 0 && depth % 2 == 0 ? 8
                                                                           : 4);
      auto vBlock = graph.addVertex(
          cs, poputil::templateVertex("poplin::experimental::LUDBlockVertex",
                                      alignment));
      graph.connect(vBlock["sliceLU"], sliceU);
      graph.connect(vBlock["sliceLURow"], sliceLRow);
      graph.connect(vBlock["sliceLUCol"], sliceLCol);
      graph.setInitialValue(vBlock["depth"], depth);
      graph.setInitialValue(vBlock["width"], block.width);
      graph.setInitialValue(vBlock["height"], block.height);
      graph.setTileMapping(vBlock, tileId);
      blockMM[tileId].pop_front();
    }
  }
#ifndef NDEBUG
  for (auto &tilePlan : blockMM) {
    assert(tilePlan.size() == 0);
  }
#endif
}

// All computations are executed on tU tensor. After factorization is done,
// data in bottom left needs to pe copied to tL tensor, and 0 filled in tU
void tensorSplit(const LuDataLayout &layout, Graph &graph, ComputeSet &cs,
                 Tensor &tL, Tensor &tU) {
  for (unsigned y = 0; y < layout.m; y += layout.blockSize) {
    unsigned height = layout.blockSize;
    if (y + height > layout.m) {
      height = layout.m - y;
    }
    const unsigned limit = std::min(y, layout.minEdge);
    for (unsigned x = 0; x <= limit; x += layout.blockSize) {
      unsigned width = layout.blockSize;
      if (x + width > layout.minEdge) {
        width = layout.minEdge - x;
      }
      if (!width) {
        continue;
      }
      std::string vertexName;
      if (x != y) {
        vertexName = "poplin::experimental::LUDBlockSplitVertex";
      } else {
        vertexName = "poplin::experimental::LUDCoreSplitVertex";
      }
      auto sliceLU = tU.slice({y, x}, {y + height, x + width});
      auto sliceL = tL.slice({y, x}, {y + height, x + width});
      auto vSplit = graph.addVertex(cs, vertexName);
      graph.connect(vSplit["sliceLU"], sliceLU);
      graph.connect(vSplit["sliceL"], sliceL);
      graph.setInitialValue(vSplit["height"], height);
      graph.setInitialValue(vSplit["width"], width);
      graph.setTileMapping(vSplit, getLUTileId(layout, y, x));
    }
  }
}

// Initial assign of tensor data blocks to IPU tiles.
void bindDataToTile(LuDataLayout &layout) {
  unsigned tileId = 0;
  unsigned blockMaxEdge = std::max(layout.blocksInCol, layout.blocksInRow);
  unsigned blockMinEdge = std::min(layout.blocksInCol, layout.blocksInRow);
  for (unsigned i = 0; i < blockMinEdge; i++) {
    // Top left corner
    assert(i * layout.blocksInRow + i < layout.dataToTile.size());
    layout.dataToTile[i * layout.blocksInRow + i] =
        (tileId++ % layout.numTiles);
    for (unsigned block = i + 1; block < blockMaxEdge; block++) {
      // Row
      if (i * layout.blocksInRow + block < layout.dataToTile.size()) {
        layout.dataToTile[i * layout.blocksInRow + block] =
            (tileId++ % layout.numTiles);
      }
      // Col
      if (block * layout.blocksInRow + i < layout.dataToTile.size()) {
        layout.dataToTile[block * layout.blocksInRow + i] =
            (tileId++ % layout.numTiles);
      }
    }
  }
}

} // namespace

std::pair<poplar::Tensor, poplar::Tensor>
LUFactorization(poplar::Graph &graph, poplar::Tensor &input,
                poplar::program::Sequence &seq,
                const poplar::DebugContext &debugContext) {
  // Height of input tensor
  const unsigned m = input.shape()[0];
  // Width of input tensor
  const unsigned n = input.shape()[1];
  const unsigned numTiles = graph.getTarget().getNumTiles();

  Tensor tL = graph.addVariable(FLOAT, {m, m}, "L");
  Tensor tU = graph.addVariable(FLOAT, {m, n}, "U");

  poputil::mapTensorLinearly(graph, tL);
  poputil::mapTensorLinearly(graph, tU);

  // Try to select optimal block size (arbitrary thresholds)
  LuDataLayout layout;
  layout.minEdge = std::min(m, n);
  layout.maxEdge = std::max(m, n);
  unsigned blockSize = 48;
  if (layout.minEdge < 768) {
    blockSize = 6;
  } else if (layout.minEdge < 1048) {
    blockSize = 12;
  } else if (layout.minEdge < 1200) {
    blockSize = 24;
  }

  layout.blockSize = blockSize;
  layout.blocksInCol = ceil((float)m / blockSize);
  layout.blocksInRow = ceil((float)n / blockSize);
  layout.numTiles = numTiles;
  layout.m = m;
  layout.n = n;
  layout.dataToTile.resize(layout.blocksInRow * layout.blocksInCol);

  bindDataToTile(layout);

  std::vector<std::list<LuCompute>> blockMM(numTiles);

  // All calculations will be done on tensor U. L will be populated only at very
  // last stage.
  if (input.elementType() == FLOAT) {
    seq.add(Copy(input, tU));
  } else {
    // LU requires high precision and should not be executed on half tensors
    seq.add(popops::cast(graph, input, tU, {debugContext}));
  }

  // Update poplar graph data to tile mapping
  for (unsigned y = 0; y < layout.blocksInCol; y++) {
    unsigned blockYbegin = y * blockSize;
    unsigned blockYend = y * blockSize + blockSize;
    if (blockYend > layout.m) {
      blockYend = layout.m;
    }
    for (unsigned x = 0; x < layout.blocksInRow; x++) {
      unsigned blockXbegin = x * blockSize;
      unsigned blockXend = x * blockSize + blockSize;
      if (blockXend > layout.n) {
        blockXend = layout.n;
      }
      auto sliceU =
          tU.slice({blockYbegin, blockXbegin}, {blockYend, blockXend});
      unsigned tileIdx = getLUTileId(layout, blockYbegin, blockXbegin);
      graph.setTileMapping(sliceU, tileIdx);
    }
  }

  // Depth represents number of factorized rows and columns in single step
  const unsigned defaultDepth = blockSize < 6 ? blockSize : 6;
  const unsigned nRound = ceil((float)layout.minEdge / defaultDepth);
  unsigned start = 0;

  // Algorithm may leave some computations form previous iteration for the next
  // one. This allows to improve Tile utilization in IPU. PrevStage struct holds
  // information about previous computations
  PrevStage prev{0, layout.maxEdge, 0};

  // Main seq
  for (unsigned i = 0; i < nRound; i++) {
    unsigned depth = defaultDepth;
    if (start + defaultDepth > layout.minEdge) {
      depth = layout.minEdge - start;
    }

    // X X 0 0 0 0
    // X X 0 0 0 0
    // 0 0 0 0 0 0
    // 0 0 0 0 0 0
    // 0 0 0 0 0 0
    // 0 0 0 0 0 0
    // Example for 6x6 matrix and depth = 2
    // STAGE 1
    // Factorize tile in top left corner (X).
    {
      auto sliceCore = tU.slice({start, start}, {start + depth, start + depth});
      ComputeSet csCore = graph.addComputeSet("csCore");
      const unsigned alignment = (depth % 2 == 0 ? 8 : 4);
      auto vCore = graph.addVertex(
          csCore, poputil::templateVertex("poplin::experimental::LUDCoreVertex",
                                          alignment));
      graph.connect(vCore["sliceLU"], sliceCore);
      graph.setInitialValue(vCore["depth"], depth);
      unsigned tileIdx = getLUTileId(layout, start, start);
      // Update pending data (P) from previous stage 3
      if (prev.end < layout.maxEdge) {
        assignBlock(layout, prev.end, layout.maxEdge - prev.end, blockMM);
        // Reassign core computations to least occupied tile.
        unsigned minWorkTile = tileIdx;
        unsigned minWork = blockMM[tileIdx].size();
        for (size_t idx = 0; idx < blockMM.size(); idx++) {
          auto &tileWork = blockMM[idx];
          if (tileWork.size() < minWork) {
            minWork = tileWork.size();
            minWorkTile = idx;
          }
        }
        tileIdx = minWorkTile;
        executeBlock(graph, csCore, tU, prev.start, prev.depth, blockMM);
        prev.end = layout.maxEdge;
      }
      graph.setTileMapping(vCore, tileIdx);
      seq.add(Execute(csCore));
    }

    // If this is the last seq pass we are done

    // X X R R R R
    // X X R R R R
    // C C 0 0 0 0
    // C C 0 0 0 0
    // C C 0 0 0 0
    // C C 0 0 0 0
    // STAGE 2
    // Update rows (R) at level of sliceCore and cols (C) bellow sliceCore.
    {
      ComputeSet csRowsCols = graph.addComputeSet("cs_rows_cols");
      updateRowCol(layout, graph, csRowsCols, tU, start, depth);
      seq.add(Execute(csRowsCols));
    }

    // X X R R R R
    // X X R R R R
    // C C B B B B
    // C C B B B B
    // C C B B P P
    // C C B B P P
    // STAGE 3
    // Update rest of the matrix (B).
    // This data will participate in next computations.
    // Not all data has to be updated in this compute set, only parts required
    // to compute stages 1 and 2 in next seq pass. Rest of workload (P) might be
    // deferred and computed in parallel to STAGE 1
    {
      ComputeSet csBlock = graph.addComputeSet("cs_block");
      const unsigned begin = start + depth;
      const unsigned end =
          std::min(alignUp(begin + (layout.blockSize * 16), layout.blockSize),
                   layout.maxEdge);
      unsigned size = end - begin;
      assignBlock(layout, begin, size, blockMM);
      executeBlock(graph, csBlock, tU, start, depth, blockMM);
      seq.add(Execute(csBlock));
      if (end < layout.maxEdge) {
        prev.end = end;
        prev.start = start;
        prev.depth = depth;
      }
    }
    start += depth;
  }

  // LU factorization is done, now move data from U tensor to L tensor and zero
  // lower part of U
  {
    ComputeSet csLUTensorSplit = graph.addComputeSet("cs_lu_tensor_split");
    tensorSplit(layout, graph, csLUTensorSplit, tL, tU);
    seq.add(Execute(csLUTensorSplit));
  }

  return {tL, tU};
}

} // namespace experimental
} // namespace poplin
