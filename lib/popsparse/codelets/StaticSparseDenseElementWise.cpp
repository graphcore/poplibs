// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "poplibs_support/ExternalCodelet.hpp"
#include <SparseMetaInfo.hpp>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

namespace popsparse {

template <class FPType, class AccumType>
class [[poplar::constraint(
    "elem(*workList) != elem(*in)")]] StaticSparseDenseElementWise
    : public MultiVertex {
  poplar::Output<poplar::Vector<AccumType, poplar::VectorLayout::ONE_PTR, 8>>
      out;
  poplar::Input<poplar::Vector<FPType, poplar::VectorLayout::ONE_PTR, 8>> in;
  poplar::Input<poplar::Vector<FPType, poplar::VectorLayout::ONE_PTR>> nz;
  poplar::Input<poplar::Vector<unsigned short, poplar::VectorLayout::ONE_PTR>>
      workList;
  unsigned numZ;

  using MetaInfoType = unsigned short;
  using MetaInfo = popsparse::StaticMetaInfo<MetaInfoType>;

public:
  StaticSparseDenseElementWise();
  IS_EXTERNAL_CODELET(true);

  bool compute(unsigned int wid) {
    const auto *wl =
        reinterpret_cast<const MetaInfo::WorkListEntry *>(&workList[0]);

    const auto numRows = wl[wid].numRows;
    const auto wlOffset = wl[wid].metaInfoOffsetOutputEntry;
    const auto nzOffset = wl[wid].sparseOffset;
    const auto numZWorker = wl[wid].numZ;
    const auto zOffset = wl[wid].offsetZ;
    const unsigned baseRowOffset = wl[wid].rowOffset;
    const auto *outputEntry =
        reinterpret_cast<const MetaInfo::OutputEntry *>(&workList[wlOffset]);

    const FPType *pNZ = &nz[nzOffset];
    constexpr unsigned typeSize = std::is_same_v<FPType, float> ? 4 : 2;

    for (unsigned r = 0; r != numRows; ++r) {
      // extract Z offset
      unsigned int numCols = outputEntry->numYm1 + std::is_same_v<FPType, half>;
      const auto *inputEntryRow =
          reinterpret_cast<const MetaInfo::InputEntry *>(outputEntry + 1);
      const FPType *pNZRow = pNZ;

      // scale to row offset
      auto rowOffset = ((baseRowOffset + r) * numZ) + zOffset;
      for (unsigned z = 0; z != numZWorker; ++z) {
        float acc = 0.0f;
        const auto *inputEntry = inputEntryRow;
        for (unsigned c = 0; c != numCols; ++c) {
          unsigned col = (inputEntry->offsetYInS) / typeSize + zOffset;
          acc += (float)in[col + z] * (float)pNZRow[c];
          ++inputEntry;
        }
        out[rowOffset + z] = acc;
      }
      outputEntry = reinterpret_cast<const MetaInfo::OutputEntry *>(
          inputEntryRow + numCols);

      // move pointer forward
      pNZ += numCols;
    }
    return true;
  }
};

template class StaticSparseDenseElementWise<float, float>;
template class StaticSparseDenseElementWise<half, half>;
} // namespace popsparse
