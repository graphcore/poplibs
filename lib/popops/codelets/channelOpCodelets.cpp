// This file contains all the "BroacastVectorInnerXXXX" codelets, historically
// called 'Channel Ops' codelets (AddToChannel, ChannelMul etc.).
// They deal with broadcasting a vector 'B' in the innermost (i.e. rightmost,
// contiguous in memory) dimension of a tensor ('data') and storing the result
// either in the same 'data' tensor ('inplace' operation) or in a differnt
// tensor ('out')

#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cassert>
#include <cmath>
#include <type_traits>

#include "popops/ExprOp.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto DELTAN = poplar::VectorListLayout::DELTAN;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;
static constexpr auto SCALED_PTR64 = poplar::VectorLayout::SCALED_PTR64;


#if defined(__IPU__) && !defined(POPLIBS_DISABLE_ASM_CODELETS)
#define EXTERNAL_CODELET true
#else
#define EXTERNAL_CODELET false
#endif

namespace popops {

// Define function templates to do add or a multiply (based on a
// 'expr::BroadcastOpType' parameter) with float and half
template<expr::BroadcastOpType op, typename T> struct ElementOp {};

template<typename T> struct ElementOp<expr::BroadcastOpType::ADD, T>
{ static T fn(T a, T b) {return a + b;} };

template<typename T> struct ElementOp<expr::BroadcastOpType::MULTIPLY, T>
{ static T fn(T a, T b) {return a * b;} };



// A macro to instantiate each template for all supported operations and
// the two data types (float, half)
#define INSTANTIATE(name) \
template class name<expr::BroadcastOpType::ADD, float>;\
template class name<expr::BroadcastOpType::ADD, half>;\
template class name<expr::BroadcastOpType::MULTIPLY, float>;\
template class name<expr::BroadcastOpType::MULTIPLY, half>;\
template class name<expr::BroadcastOpType::SCALED_ADD, float>;\
template class name<expr::BroadcastOpType::SCALED_ADD, half>





// -----------------  Supervisor version (non-inplace) ----------------

template <expr::BroadcastOpType op, class FPType>
class
[[poplar::constraint("elem(*data) != elem(*out)")]]
BroadcastVectorInnerByColumnSupervisor : public SupervisorVertex {
public:
  BroadcastVectorInnerByColumnSupervisor();

  Input<Vector<FPType, SPAN, 8>> B;
  Input<Vector<FPType, ONE_PTR, 8>> data;
  // dataBlockCount = data.size() / B.size();
  // dataBlockCountPacked = (dataBlockCount/6 << 3) | (dataBlockCount % 6)
  const uint16_t dataBlockCountPacked;
  Output<Vector<FPType, ONE_PTR, 8>> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    unsigned chansPerGroup = B.size();
    unsigned dataBlockCount = (dataBlockCountPacked >> 3) * 6
                              + (dataBlockCountPacked & 0x07);
    for (unsigned j = 0; j != dataBlockCount; ++j) {
      for (unsigned k = 0; k != chansPerGroup; ++k) {
        out[j*chansPerGroup + k] =
                     ElementOp<op, FPType>::fn(data[j*chansPerGroup + k],B[k]);
      }
    }
    return true;
  }
};

// partial specialization for SCALED_ADD
template <class FPType>
class
[[poplar::constraint("elem(*data) != elem(*out)")]]
BroadcastVectorInnerByColumnSupervisor<expr::BroadcastOpType::SCALED_ADD,
                                            FPType> : public SupervisorVertex {
public:
  BroadcastVectorInnerByColumnSupervisor();

  Input<Vector<FPType, SPAN, 8>> B;
  Input<Vector<FPType, ONE_PTR, 8>> data;
  // dataBlockCount = data.size() / B.size();
  // dataBlockCountPacked = (dataBlockCount/6 << 3) | (dataBlockCount % 6)
  const uint16_t dataBlockCountPacked;
  const FPType scale;
  Output<Vector<FPType, ONE_PTR, 8>> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    unsigned chansPerGroup = B.size();
    unsigned dataBlockCount = (dataBlockCountPacked >> 3) * 6
                              + (dataBlockCountPacked & 0x07);
    for (unsigned j = 0; j != dataBlockCount; ++j) {
      for (unsigned k = 0; k != chansPerGroup; ++k) {
        out[j * chansPerGroup + k] = data[j * chansPerGroup + k] +
                                     B[k] * scale;
      }
    }
    return true;
  }
};

INSTANTIATE(BroadcastVectorInnerByColumnSupervisor);



// ----------------- Supervisor version in-place -------------
template <expr::BroadcastOpType op, class FPType>
class
BroadcastVectorInnerByColumnInPlaceSupervisor : public SupervisorVertex {
public:
  BroadcastVectorInnerByColumnInPlaceSupervisor();

  // The half float code for the ADD inplace operation requires the 'acts'
  // tensor to be in an interleaved region, to be able to use the ldst64pace
  // instruction. This is really needed only if addend.size() is a multiple of
  // of four (fast optimized code).
  static const bool needsInterleave = std::is_same<FPType, half>::value &&
                                      op == expr::BroadcastOpType::ADD;

  Input<Vector<FPType, SPAN, 8>> B;
  InOut<Vector<FPType, ONE_PTR, 8, needsInterleave>> data;
  // dataBlockCount = data.size() / B.size();
  // dataBlockCountPacked = (actsBlockCount/6 << 3) | (actsBlockCount % 6)
  const uint16_t dataBlockCountPacked;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    unsigned chansPerGroup = B.size();
    unsigned dataBlockCount = (dataBlockCountPacked >> 3) * 6
                              + (dataBlockCountPacked & 0x07);
    for (unsigned j = 0; j != dataBlockCount; ++j) {
      for (unsigned k = 0; k != chansPerGroup; ++k) {
        data[j*chansPerGroup + k] =
                    ElementOp<op, FPType>::fn(data[j*chansPerGroup + k],B[k]);
      }
    }
    return true;
  }
};

// Partial specialization for SCALED_ADD
template <class FPType>
class
BroadcastVectorInnerByColumnInPlaceSupervisor<expr::BroadcastOpType::SCALED_ADD,
                                             FPType> : public SupervisorVertex {
public:
  BroadcastVectorInnerByColumnInPlaceSupervisor();

  // The half float code for this inplace operation requires the 'acts' tensor
  // to be in an interleaved region, to be able to use the ldst64pace
  // instruction. This is really needed only if addend.size() is a multiple of
  // of four (fast optimized code).
  static const bool needsInterleave = std::is_same<FPType, half>::value;

  Input<Vector<FPType, SPAN, 8>> B;
  InOut<Vector<FPType, ONE_PTR, 8, needsInterleave>> data;
  // dataBlockCount = data.size() / B.size();
  // dataBlockCountPacked = (dataBlockCount/6 << 3) | (dataBlockCount % 6)
  const uint16_t dataBlockCountPacked;
  const FPType scale;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    unsigned chansPerGroup = B.size();
    unsigned dataBlockCount = (dataBlockCountPacked >> 3) * 6
                              + (dataBlockCountPacked & 0x07);
    for (unsigned j = 0; j != dataBlockCount; ++j) {
      for (unsigned k = 0; k != chansPerGroup; ++k) {
        data[j * chansPerGroup + k] += B[k] * scale;
      }
    }
    return true;
  }
};

INSTANTIATE(BroadcastVectorInnerByColumnInPlaceSupervisor);



// ----------------- Worker 2D version (non-inplace) ----------------
template <expr::BroadcastOpType op, class FPType>
class
[[poplar::constraint("elem(**data) != elem(**out)")]]
BroadcastVectorInnerByColumn2D : public Vertex {
public:
  BroadcastVectorInnerByColumn2D();

  // n is equal to B.size(), BLen.size(), data.size()
  // and dataBlockCount.size()
  const uint32_t n;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> B;
  Vector<uint16_t, ONE_PTR> BLen;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> data;
  Vector<uint16_t, ONE_PTR> dataBlockCount;
  Vector<Output<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i != n; ++i) {
        unsigned blockCount = dataBlockCount[i];
        unsigned len = BLen[i];

        for (unsigned b = 0; b != blockCount; ++b) {
            for (unsigned a = 0; a != len; ++a) {
                out[i][b * len + a] =
                      ElementOp<op, FPType>::fn(data[i][b * len + a], B[i][a]);
            }
        }
    }

    return true;
  }
};

// Partial specialization for SCALED_ADD
template <class FPType>
class
[[poplar::constraint("elem(**data) != elem(**out)")]]
BroadcastVectorInnerByColumn2D<expr::BroadcastOpType::SCALED_ADD, FPType> :
                                                                public Vertex {
public:
  BroadcastVectorInnerByColumn2D();

  // n is equal to B.size(), BLen.size(), data.size()
  // and dataBlockCount.size()
  const uint32_t n;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> B;
  Vector<uint16_t, ONE_PTR> BLen;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> data;
  Vector<uint16_t, ONE_PTR> dataBlockCount;
  const FPType scale;
  Vector<Output<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i != n; ++i) {
      unsigned blockCount = dataBlockCount[i];
      unsigned len = BLen[i];

      for (unsigned b = 0; b != blockCount; ++b) {
        for (unsigned a = 0; a != len; ++a) {
            out[i][b * len + a] = data[i][b * len + a] + B[i][a] * scale;
        }
      }
    }

    return true;
  }
};

INSTANTIATE(BroadcastVectorInnerByColumn2D);



// ----------------- Worker 2D version inplace ----------------
template <expr::BroadcastOpType op, class FPType>
class
BroadcastVectorInnerByColumn2DInPlace : public Vertex {
public:
  BroadcastVectorInnerByColumn2DInPlace();

  // The half float code for this inplace operation requires the 'acts' tensor
  // to be in an interleaved region, to be able to use the ldst64pace
  // instruction. This is really needed only if addend.size() is a multiple of
  // of four (fast optimized code)
  static const bool needsInterleave = std::is_same<FPType, half>::value;

  // n is equal to B.size(), B.size(), data.size()
  // and dataBlockCount.size()
  const uint32_t n;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> B;
  Vector<uint16_t, ONE_PTR> BLen;
  Vector<InOut<Vector<FPType, ONE_PTR, 8, needsInterleave>>, ONE_PTR> data;
  Vector<uint16_t, ONE_PTR> dataBlockCount;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i != n; ++i) {
        unsigned blockCount = dataBlockCount[i];
        unsigned len = BLen[i];

        for (unsigned b = 0; b != blockCount; ++b) {
            for (unsigned a = 0; a != len; ++a) {
                data[i][b * len + a] =
                      ElementOp<op, FPType>::fn(data[i][b * len + a], B[i][a]);
            }
        }
    }

    return true;
  }
};

// Partial specialization for SCALED_ADD
template <class FPType>
class
BroadcastVectorInnerByColumn2DInPlace<expr::BroadcastOpType::SCALED_ADD,
                                                      FPType> : public Vertex {
public:
  BroadcastVectorInnerByColumn2DInPlace();

  static const bool needsInterleave = std::is_same<FPType, half>::value;

  // n is equal to B.size(), BLen.size(), data.size()
  // and dataBlockCount.size()
  const uint32_t n;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> B;
  Vector<uint16_t, ONE_PTR> BLen;
  Vector<InOut<Vector<FPType, ONE_PTR, 8, needsInterleave>>, ONE_PTR> data;
  Vector<uint16_t, ONE_PTR> dataBlockCount;
  const FPType scale;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i != n; ++i) {
        unsigned blockCount = dataBlockCount[i];
        unsigned len = BLen[i];

        for (unsigned b = 0; b != blockCount; ++b) {
            for (unsigned a = 0; a != len; ++a) {
                data[i][b * len + a] += B[i][a] * scale;
            }
        }
    }

    return true;
  }
};

INSTANTIATE(BroadcastVectorInnerByColumn2DInPlace);


} // end namespace popops
