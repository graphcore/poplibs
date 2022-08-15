// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = VectorLayout::ONE_PTR;

template <typename VertexClass>
static constexpr inline bool hasAssemblyVersion() {
  return false;
}

namespace popsparse {

template <typename Type>
class BufferIndexUpdate
    : public SupervisorVertexIf<hasAssemblyVersion<BufferIndexUpdate<Type>>() &&
                                ASM_CODELETS_ENABLED> {
public:
  BufferIndexUpdate();

  // Pointers to buckets of sparse non-zero input values in r.
  InOut<Type> index;

  IS_EXTERNAL_CODELET((hasAssemblyVersion<BufferIndexUpdate<Type>>()));

  void compute() { *index = 1 - *index; }
};

template class BufferIndexUpdate<unsigned>;

template <typename StorageType, typename IndexType>
class BitIsSet : public SupervisorVertexIf<
                     hasAssemblyVersion<BitIsSet<StorageType, IndexType>>() &&
                     ASM_CODELETS_ENABLED> {
  static_assert(std::is_unsigned<StorageType>::value,
                "Storage type must be an unsigned integer");
  static_assert((sizeof(StorageType) & (sizeof(StorageType) - 1)) == 0,
                "Storage type must have power of 2 size");
  using OutputType = unsigned;
  static_assert(sizeof(OutputType) >= sizeof(StorageType),
                "Output type size must be greater or equal storage type size");

public:
  BitIsSet();

  Input<Vector<StorageType, ONE_PTR>> bits;
  Input<IndexType> index;
  // unsigned avoids sub-word writes even though we only need bool really.
  Output<OutputType> out;

  IS_EXTERNAL_CODELET((hasAssemblyVersion<BitIsSet<StorageType, IndexType>>()));
  void compute() {
    const auto storageMask = 1u << (index & (sizeof(StorageType) - 1));
    *out = bits[index / sizeof(StorageType)] & storageMask;
  }
};

template class BitIsSet<unsigned short, unsigned>;

} // end namespace popsparse
