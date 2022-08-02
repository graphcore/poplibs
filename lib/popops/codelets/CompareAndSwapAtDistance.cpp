// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/Alignment.hpp>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto DELTAN = poplar::VectorListLayout::DELTANELEMENTS;

namespace popops {

namespace {

using WorklistType = unsigned short;

} // end anonymous namespace

template <typename T> constexpr static inline T min(const T &a, const T &b) {
  return a < b ? a : b;
}

// The format of the worklists for each worker is as follows in order as they
// appear in memory.
//
// [numEntriesM1]         A count for the number of entries that will follow,
//                        minus 1.
//                        the always-present info in the worklist.
// [initialOffset]        An offset in groups into the indices/values that
//                        this worker will start at.
// [initialOrder/initialCount] 2 entries to form a 32-bit packed field.
//   [0-1: initialDirection] The initial direction for this worker
//   [1-32:  initialCount]   The initial count for the counter that tracks
//                        changes of direction.
// [firstInnerElemCount]  Because the first entry for a worker may start
//                        at a weird offset, the element count for the first
//                        inner loop is given in the worklist.
// numEntries * {
//   [distance]             The distance at which to do compare and swap.
//   [elemCount]            The total number of elements to compare and swap
//                          for this entry.
// }
template <typename Impl>
static void workerCompute(unsigned wid, Impl impl, const WorklistType *worklist,
                          unsigned distanceToChangeOrder) {
  const auto numEntriesM1 = *worklist++;
  auto numEntries = numEntriesM1 + 1;
  const auto initialOffset = *worklist++;
  impl.increment(initialOffset);
  auto *worklistUnsigned = reinterpret_cast<const unsigned *>(worklist);
  const auto packedOrderAndCount = *worklistUnsigned++;
  worklist = reinterpret_cast<const WorklistType *>(worklistUnsigned);
  bool order = packedOrderAndCount & 1u;
  const auto initialCount = packedOrderAndCount >> 1u;
  auto changeOrderCounter = distanceToChangeOrder - initialCount;

  unsigned innerElemCount = *worklist++;
  // numEntries should never be 0
  unsigned distance = *worklist++;
  unsigned numElems = *worklist++;

  while (numEntries-- > 0) {
    while (numElems != 0) {
      numElems -= innerElemCount;
      changeOrderCounter -= innerElemCount;

      // This will be a rpt loop.
      while (innerElemCount-- > 0) {
        impl.compareAndSwap(distance, order);
        impl.increment(1);
      }
      impl.increment(distance);
      innerElemCount = min(distance, numElems);
      if (changeOrderCounter == 0) {
        order = !order;
        changeOrderCounter = distanceToChangeOrder;
      }
    }
    // We will overread here but only by 2 worklist elements i.e.
    // 4 bytes.
    distance = *worklist++;
    numElems = *worklist++;
    innerElemCount = min(distance, numElems);
  }
}

template <typename KeyType> constexpr inline bool hasAssemblyVersion() {
  return false;
}

template <typename KeyType> struct KeyImpl {
  KeyType *keys;
  void compareAndSwap(unsigned distance, bool order) {
    if (order == (keys[0] > keys[distance])) {
      std::swap(keys[0], keys[distance]);
    }
  }
  void increment(unsigned n) { keys += n; }
};

template <typename KeyType>
class CompareAndSwapAtDistance : public MultiVertex {
public:
  InOut<Vector<KeyType, ONE_PTR>> keys;

  // Outer dimension of array indexed by worker context id
  Vector<Input<Vector<WorklistType, ONE_PTR, 4>>> worklists;

  // Used with the logical offset to determine the direction of comparison for
  // a given comparison.
  unsigned distanceToChangeOrder;

  IS_EXTERNAL_CODELET((hasAssemblyVersion<KeyType>()));

  void compute(unsigned wid) {
    const auto numWorkers = worklists.size();
    KeyImpl<KeyType> impl = {&keys[0]};
    if (wid < numWorkers) {
      const WorklistType *worklist = &worklists[wid][0];
      workerCompute(wid, impl, worklist, distanceToChangeOrder);
    }
  }
};

template class CompareAndSwapAtDistance<float>;
template class CompareAndSwapAtDistance<unsigned>;
template class CompareAndSwapAtDistance<int>;

template <typename KeyType, typename ValueType, bool valuesAreSecondaryKey>
constexpr inline bool hasAssemblyVersionKeyVal() {
  return !valuesAreSecondaryKey && std::is_same_v<KeyType, float> &&
         std::is_same_v<ValueType, float>;
}

template <typename KeyType, typename ValueType, bool valuesAreSecondaryKey,
          bool sortValuesInReverseOrder>
struct KeyValImpl {
  KeyType *keys;
  ValueType *values;
  void compareAndSwap(unsigned distance, bool order) {
    if constexpr (valuesAreSecondaryKey) {
      const bool keysAreEqual = keys[0] == keys[distance];
      const bool valuesCmpRes = sortValuesInReverseOrder
                                    ? values[0] < values[distance]
                                    : values[0] > values[distance];
      const bool swapByKeys =
          (!keysAreEqual) && (order == (keys[0] > keys[distance]));
      const bool swapByValues = keysAreEqual && (order == valuesCmpRes);
      if (swapByKeys || swapByValues) {
        std::swap(keys[0], keys[distance]);
        std::swap(values[0], values[distance]);
      }
    } else {
      if (order == (keys[0] > keys[distance])) {
        std::swap(keys[0], keys[distance]);
        std::swap(values[0], values[distance]);
      }
    }
  }
  void increment(unsigned n) {
    keys += n;
    values += n;
  }
};

template <typename KeyType, typename ValueType, bool valuesAreSecondaryKey,
          bool sortValuesInReverseOrder>
class CompareAndSwapAtDistanceKeyVal : public MultiVertex {
public:
  InOut<Vector<KeyType, ONE_PTR>> keys;
  InOut<Vector<ValueType, ONE_PTR>> values;

  // Outer dimension of array indexed by worker context id
  Vector<Input<Vector<WorklistType, ONE_PTR, 4>>> worklists;

  // Used with the logical offset to determine the direction of comparison for
  // a given comparison.
  unsigned distanceToChangeOrder;

  IS_EXTERNAL_CODELET(
      (hasAssemblyVersionKeyVal<KeyType, ValueType, valuesAreSecondaryKey>()));

  void compute(unsigned wid) {
    const auto numWorkers = worklists.size();
    KeyValImpl<KeyType, ValueType, valuesAreSecondaryKey,
               sortValuesInReverseOrder>
        impl = {&keys[0], &values[0]};
    if (wid < numWorkers) {
      const WorklistType *worklist = &worklists[wid][0];
      workerCompute(wid, impl, worklist, distanceToChangeOrder);
    }
  }
};

template class CompareAndSwapAtDistanceKeyVal<float, float, false, false>;
template class CompareAndSwapAtDistanceKeyVal<float, unsigned, false, false>;
template class CompareAndSwapAtDistanceKeyVal<float, int, false, false>;
template class CompareAndSwapAtDistanceKeyVal<unsigned, float, false, false>;
template class CompareAndSwapAtDistanceKeyVal<int, float, false, false>;

template class CompareAndSwapAtDistanceKeyVal<float, float, true, false>;
template class CompareAndSwapAtDistanceKeyVal<float, unsigned, true, false>;
template class CompareAndSwapAtDistanceKeyVal<float, int, true, false>;
template class CompareAndSwapAtDistanceKeyVal<unsigned, float, true, false>;
template class CompareAndSwapAtDistanceKeyVal<unsigned, unsigned, true, false>;
template class CompareAndSwapAtDistanceKeyVal<unsigned, int, true, false>;
template class CompareAndSwapAtDistanceKeyVal<int, float, true, false>;
template class CompareAndSwapAtDistanceKeyVal<int, unsigned, true, false>;
template class CompareAndSwapAtDistanceKeyVal<int, int, true, false>;

template class CompareAndSwapAtDistanceKeyVal<float, float, true, true>;
template class CompareAndSwapAtDistanceKeyVal<float, unsigned, true, true>;
template class CompareAndSwapAtDistanceKeyVal<float, int, true, true>;
template class CompareAndSwapAtDistanceKeyVal<unsigned, float, true, true>;
template class CompareAndSwapAtDistanceKeyVal<unsigned, unsigned, true, true>;
template class CompareAndSwapAtDistanceKeyVal<unsigned, int, true, true>;
template class CompareAndSwapAtDistanceKeyVal<int, float, true, true>;
template class CompareAndSwapAtDistanceKeyVal<int, unsigned, true, true>;
template class CompareAndSwapAtDistanceKeyVal<int, int, true, true>;

} // end namespace popops
