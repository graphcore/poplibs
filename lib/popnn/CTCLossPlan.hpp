// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popnn_CTCLossPlan_hpp
#define popnn_CTCLossPlan_hpp

#include <poplar/Interval.hpp>
#include <poplibs_support/Algorithm.hpp>
#include <popnn/CTCLoss.hpp>

namespace popnn {
namespace ctc {

// An input consists of:
// prob[maxT, B, A] of type half or float
// labels[B, maxL] of type unsigned short
// probLengths[B] (unsigned, <= maxT)
// labelLengths[B] (unsigned, <= maxL)
//
// Where T = Timesteps    B = batch size  A = labels in the alphabet
// Temporary memory resides in a Tensor:
// temp[maxT, B, El] (half or float containing alpha or beta)
// Where El = ExtendedLabels = 2 * maxL + 1
//
// `prob` is compared to `labels` and the gradient returned:
// Output: grad[maxT, B, A]
//
// Each input in a batch has a different length (in both T and L) hence the
// `probLengths` and `labelLengths` inputs.
// The input shape is designed/planned as maxT and maxL which cannot then be
// exceeded

// Work is divided: `SerialPartition` and `ParallelPartition`. Ideally a
// `ParallelPartition` will allocate the whole
// dataset (prob[], labels[], tempory data etc) over the whole IPU.
// If this is not possible, `SerialPartition` will allocate a part of the
// data at once, resulted in repeated processing of `ParallelPartition`s.
// The nature of the data dependency when calculating gradient will result in
// serial processing in many cases even when using a single `ParallelPartition`.
//
// This means that it is unlikey in many cases that truly efficient parallel
// processing will be possible - ie - using 4 tiles may result in taking 3/4
// of the time, not 1/4 of the time.  However the maximum data per tile
// (permanent or temporary) can be spread evenly.
//
// However when splitting work based on a `ParallelPartition` any
// temporary memory will be present throughout.  A `SerialPartition` attempts
// to reduce that temporary memory.

template <typename T, typename B> struct ParallelPartition {
  // The number of splits across the batch dimension.  Each is completely
  // independent so there is no exchange / temporary memory duplication cost
  // to splitting in this way.  Each can be processed in parallel
  T batch;

  // When splitting work by time and label, dependencies operate in a grid
  // (1a means in timestep 1 this tile can compute alpha
  //  2ab means that in timestep 2 this tile can compute alpha and beta)
  // Scheduling could result in 4 timesteps if only 1 of alpha or beta is
  // calculated by a tile at once
  //             T
  //   +------------------------+
  //   | 1a 2-     | 2ab        |
  // El| 3b        |            |
  //   +------------------------+
  //   | 2ab       | 1b 2-      |
  //   |           | 3a         |
  //   +------------------------+
  //
  // Below we describe splitting each dimension on its own, but the two can be
  // combined.

  // The number of partitions the time dimension is split into
  //
  // Suppose the amount of data and therefore work is the same in each case and
  // has a time `t` to calculate all of alpha, `t` to calculate all of beta
  // Splitting by 2 allows for parallel calculation of alpha, beta (with
  // exchange of alpha[El] and beta[El] elements).
  // Continuing this logic, and describing one partition per tile
  // 1 tile : T0: alpha  (t)
  //          T0: beta   (t) Total: 2t
  // 2 tiles: T0: alpha  T1: beta (t/2)
  //          T0: grad   T1: grad (t/2) Total: t
  // 3 tiles: T0: alpha  T1: idle  T2: beta (t/3)
  //          T0: idle   T1: alpha T2: idle (t/3)
  //          T0: idle   T1: grad  T2: idle (t/3)
  //          T0: grad   T1: idle  T2: grad (t/3) Total: 4t/3 = 1.3t
  // 4 tiles: T0: alpha  T1: idle  T2: idle T3: beta (t/4)
  //          T0: idle   T1: alpha T2: beta T3: idle (t/4)
  //          T0: idle   T1: grad  T2: grad T3: alpha (t/4)
  //          T0: grad   T1: idle  T2: idle T3: grad (t/4) Total: 4t/t = t
  // As the number of tiles increases (and is even) the pattern will follow that
  // of 4 tiles, but with more idle tiles.  Likewise odd and 3 tiles - where
  // one has to process alpha then beta while all others sit idle.
  //
  // As per the example above we have idle tiles, so the chosen partition
  // could be mapped onto fewer tiles with no speed penalty.  Memory use will
  // increase however.  The 4 partition 4 tile example becomes 2 tiles, but
  // still 4 partitions.
  // 2 tiles: P0: alpha  P1: idle  P2: idle P3: beta (t/4)
  //          P0: idle   P1: alpha P2: beta P3: idle (t/4)
  //          P0: idle   P1: grad  P2: grad P3: alpha (t/4)
  //          P0: grad   P1: idle  P2: idle P3: grad (t/4)
  //         ^___________________^ ^______________________^
  //               Tile 0                  Tile 1
  // As partition P0,P1 are never active together and similarly P2,P3 there is
  // no speed cost to mapping them to the same time.  This method extends
  // to cases where partitions are made in both time and label
  T time;
  T timePartitionsPerTile;

  // The number of tiles the labels (equivalent to El) dimension is split over.
  // Each tile produces the results for a number of the labels results
  // over multiple timesteps.  This will require an overlap of data as
  // (consider alpha) there is a dependency on alpha[t-1,B,El],alpha[t-1,B,El-1]
  // and alpha[t-1,B,El-2], where it is possible that the last 2 values are on a
  // different tile.
  //         t-1     t  ->Time....
  // El-3    a0
  // El-2    a1
  // El-1    a2
  // El      a3      a4 <- Computing this, which is fn(a1,a2,a3). a1,a2 could be
  //                       on the previous tile.
  //
  // If a tile computes 1/2 its alpha workload (in T - eg a0-7 below), exchanges
  // its last 2 elements at all timesteps to the next tile (eg a0-7), that tile
  // can calculate 1/2 its alpha workload (in T - eg A0-7).  So each tile can
  // compute but lag behind.  This seems a simple method of allowing
  // parallel processing (Of course 1/2 is just an example)
  //
  // Tile 0:    t0 .. t3 STOP,EXCHANGE  t4 .. t7
  //      El=0  a  .. a                 a  .. a
  //      El=1  a0 .. a6                a8 .. a14
  //      El=2  a1 .. a7                a9 .. a15
  // Tile 1:
  //      El=3  A0 .. A6                A8 .. A14
  //      El=4  A1 .. A7                A9 .. A15
  //      El=5  A  .. A                 A  .. A
  //
  // When split by label each tile will produce gradients for a subset of the
  // labels that then need to be combined. (Eg blank appears on every tile,
  // `a` may appear on multiple tiles, and multiple times on a tile depending on
  // the input sequence).  We can choose to calculate gradient in a [T,B,El]
  // array or combine on tile into a [T,B,A] array (doing part of the reduction
  // while generating gradient).  Choice using the sliceIntoOutput flag
  T label;
  B sliceIntoOutput;

  // We can choose to split the storage of the prob[] alphabet dimension over
  // tiles, however as each input requires a dynamicSlice based on the content
  // of labels[] this will add that dynamic slice processing step (see
  // sliceFromInput)
  T alphabet;

  // We can choose to:
  // a) Read each prob[] element, using labels[] to lookup
  // b) `sliceFromInput` to produce a sliceProb[] array plus a blank symbols
  // If splitting work by label a) implies that the whole alphapet of prob[] is
  // on each tile.  If the alphabet is large, yet the number of labels processed
  // per tile is small method b) will reduce temporary memory.
  //
  // When slicing we don't expand fully and repeat the blank slice:
  // Given an expanded labels sequence - a - b - c - a - a - d.....
  // With - a - b - c - a - a - d being processed by a tile; that tile needs
  // 7 slices.
  // 1 for "-", which is static and we know we will always need it.
  // And 6 more slices for a,b,c,a,a,d (There are 3 a's as the presence of a in
  // the labels sequence is dynamic and we could have seen a,b,c,d,e,f)
  B sliceFromInput;
};

// Processing requires temporary data:
// Exchange/duplication of the alphabet over tiles
//   OR slices of alphabet to make the input sequence
// Compute alpha(or beta) in a [T,B,El] shaped tensor
// Reduce/slice that into a [T,B,A] gradient result
//
// So serialisation may be necessary to reduce temporary memory.

template <typename T> struct SerialPartition {
  // To reduce temporary memory we can calculate a subset of the batch items
  // at once.  These are independent so there are no complications
  T batch;

  // Serialising in T will involve some recomputation. Suppose we serialise
  // splitting T in 2:
  // 1. Compute alpha 0..T/2 and discard
  // 2. Compute and store alpha T/2..T
  // 3. Compute beta and so gradient T/2..T and reduce to gradient result
  // 4. Keep a beta[T/2] checkpoint
  // 5. Recompute alpha 0..T/2
  // 6. Compute beta 0..T/2 and reduce to gradient result.
  // When splitting more, alpha[T/splits] checkpoints will also be useful
  T time;

  // Serialising in El will involve some recomputation. Suppose we serialise
  // splitting El in 2:
  // 1. Compute alpha 0..El/2 and discard
  // 2. Compute and store alpha El/2..El
  // 3. Compute beta and therefore gradient also for El/2..El and reduce into
  //    gradient result
  // 4. Keep 2 beta checkpoints - at El/2 +1 and El/2 +2 as that is what
  //    remaining betas are dependent on
  // 5. Recompute alpha 0..El/2
  // 6. Compute beta 0..El/2 and therefore gradient and reduce into the
  //    gradient result
  T label;
};

// Internal Plan implementation
class Plan::Impl {
  poplar::Interval partition(unsigned fullSize, unsigned partitions,
                             unsigned index) const {
    const auto partitionSize = poplibs_support::ceildiv(fullSize, partitions);
    const auto begin = std::min(partitionSize * index, fullSize);
    const auto end = std::min(partitionSize * (index + 1), fullSize);
    return {begin, end};
  }

public:
  SerialPartition<unsigned> serial;
  ParallelPartition<unsigned, bool> parallel;

  // Given a batch size and partition index, return range of batch elements
  // represented in this partition
  poplar::Interval partitionBatch(unsigned batchSize, unsigned index) const {
    return partition(batchSize, parallel.batch, index);
  }

  // Given a time size and partition index, return range of time elements
  // represented in this partition
  poplar::Interval partitionTime(unsigned timeSize, unsigned index) const {
    return partition(timeSize, parallel.time, index);
  }

  // Given a label size and partition index, return range of label elements
  // represented in this partition
  poplar::Interval partitionLabel(unsigned labelSize, unsigned index) const {
    return partition(labelSize, parallel.label, index);
  }

  // Note passed labelSize NOT extendedLabelSize - result made to match
  // partitionLabel result when partitioning the label
  poplar::Interval partitionExtendedLabel(const std::size_t labelSize,
                                          const std::size_t partition) const {
    const auto labelPartition = partitionLabel(labelSize, partition);
    const auto begin = 2 * labelPartition.begin();
    const auto end =
        2 * labelPartition.end() + (partition == parallel.label - 1);
    return {begin, end};
  }

  unsigned getTile(unsigned batch, unsigned time, unsigned label) const {
    return batch * (parallel.time * parallel.label) // Batch
           + time * parallel.label                  // Time
           + label;                                 // Label
  }

  unsigned numTiles() const {
    return parallel.batch * parallel.time * parallel.label;
  }
  std::unique_ptr<Plan::Impl> clone() const {
    return std::make_unique<Plan::Impl>(*this);
  };
};

bool operator<(const Plan::Impl &a, const Plan::Impl &b) noexcept;
bool operator==(const Plan::Impl &a, const Plan::Impl &b) noexcept;

struct CycleEstimate {
  uint64_t alphaBetaComputeCycles;
  uint64_t alphaBetaExchangeCost;
  uint64_t gradComputeCycles;
  uint64_t gradExchangeCost;
  unsigned steps;
  unsigned serialVertexExecutions;

  uint64_t total() const {
    return alphaBetaComputeCycles + alphaBetaExchangeCost + gradComputeCycles +
           gradExchangeCost;
  }
};

// Per tile memory estimate in bytes
struct MemoryEstimate {
  uint64_t data;
  uint64_t labels;

  // Gradient for each class in partition (not yet reduced). It is the same
  // shape as the gradient tensor, but multiplied by the size of the El
  // partition for the given tile. e.g. [sizeOfElOnTile, maxT, B, A]
  uint64_t gradient;

  // Temporary storage of alpha and beta required to calculate gradient
  uint64_t alphaBetaTemp;
  // Used to pass previous state across the partition boundary
  uint64_t tempDependancies;

  uint64_t total() const {
    return data + labels + gradient + alphaBetaTemp + tempDependancies;
  }
};

} // namespace ctc
} // namespace popnn

#endif // #ifndef popnn_CTCLossPlan_hpp
