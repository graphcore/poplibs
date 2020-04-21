// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "BSUtils.hpp"
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include <unordered_map>
#include <vector>

using namespace poplar;
using namespace poputil;

namespace popsparse {
namespace experimental {

void bsCreateMaskTensor(poplar::Graph &graph, unsigned blockRow,
                        unsigned blockCol, unsigned blockRows,
                        unsigned blockCols, const unsigned char *sparsity,
                        popsparse::experimental::SubBlockMask subBlockMaskType,
                        float maskedValue, float unMaskedValue,
                        const Type &dataType, std::vector<Tensor> &maskBlocks,
                        std::vector<unsigned> &diagBlockIdxs,
                        std::vector<bool> &emptyRowsMask,
                        const std::string &debugStr) {
  if (subBlockMaskType == SubBlockMask::None) {
    throw poplibs_error("No valid masking rule was specified");
  }
  unsigned rows = blockRow * blockRows;

  Tensor maskTensor;
  maskBlocks.clear();
  diagBlockIdxs.clear();
  emptyRowsMask.resize(rows, false);
  std::unordered_map<int, Tensor> maskBlocksPool;

  unsigned idxBlockSparse = 0;
  unsigned idxBlockDense = 0;
  for (unsigned int br = 0; br < blockRows; ++br) {
    for (unsigned int bc = 0; bc < blockCols; ++bc, ++idxBlockDense) {
      if (sparsity[idxBlockDense]) {
        unsigned top = br * blockRow;
        unsigned bottom = top + blockRow;
        unsigned left = bc * blockCol;
        unsigned right = left + blockCol;
        if (left < bottom && right > top) {
          // Hit diagonal
          int rowDiagOffset = top - left;
          Tensor maskBlock;
          auto iter = maskBlocksPool.find(rowDiagOffset);
          if (iter == maskBlocksPool.end()) {
            // Create a new mask
            std::vector<float> valuesInMask(blockRow * blockCol);
            for (int br = 0, idx = 0; br < static_cast<int>(blockRow); ++br) {
              for (int bc = 0; bc < static_cast<int>(blockCol); ++bc, ++idx) {
                if (subBlockMaskType == SubBlockMask::ZeroUpperTriangle) {
                  valuesInMask[idx] =
                      bc - rowDiagOffset > br ? maskedValue : unMaskedValue;
                } else {
                  valuesInMask[idx] =
                      bc - rowDiagOffset < br ? maskedValue : unMaskedValue;
                }
              }
            }
            maskBlock =
                graph.addConstant(dataType, {1, blockRow * blockCol},
                                  valuesInMask.data(), debugStr + "/maskBlock");
            mapTensorLinearly(graph, maskBlock);
            maskBlocksPool[rowDiagOffset] = maskBlock;
          } else {
            maskBlock = iter->second;
          }

          maskBlocks.push_back(maskBlock);
          diagBlockIdxs.push_back(idxBlockSparse);

          // Flag individual rows as empty
          for (unsigned r = br * blockRow; r < br * blockRow + blockRow; ++r) {
            if (left > r &&
                subBlockMaskType == SubBlockMask::ZeroUpperTriangle) {
              emptyRowsMask[r] = true;
            } else if (right <= r &&
                       subBlockMaskType == SubBlockMask::ZeroLowerTriangle) {
              emptyRowsMask[r] = true;
            }
          }
        } else if (left >= bottom &&
                   subBlockMaskType == SubBlockMask::ZeroUpperTriangle) {
          // The whole block must be masked out - we treat this as user input
          // error.
          throw poplibs_error(
              std::string(
                  "Incorrect sparsity mask is provided. The whole block # ") +
              std::to_string(idxBlockSparse) + " is above the diagonal");
        } else if (right <= top &&
                   subBlockMaskType == SubBlockMask::ZeroLowerTriangle) {
          // The whole block must be masked out - we treat this as user input
          // error.
          throw poplibs_error(
              std::string(
                  "Incorrect sparsity mask is provided. The whole block # ") +
              std::to_string(idxBlockSparse) + " is below the diagonal");
        }
        ++idxBlockSparse;
      }
    }
  }
}

} // namespace experimental
} // namespace popsparse