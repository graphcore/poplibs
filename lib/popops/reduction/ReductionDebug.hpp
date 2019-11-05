#ifndef ReductionDebug_hpp
#define ReductionDebug_hpp

#include <string>
#include <vector>

#include <poplar/Interval.hpp>

#include "ReductionPlan.hpp"

namespace popops {

/// This class records data about a reduction. This can be used to generate
/// a visualisation of how the reduction is going to proceed.
///
/// The reductions that are done on a tile can be done in one stage ("single
/// stage") or they can be split into two stages ("first stage" and "second
/// stage"). A tile can contain both single-stage and two-stage reductions.
struct ReductionDebug {
  /// Debugging information about a partial input.
  struct Partial {
    /// The tile that the partial's variable is mapped to. In all cases it
    /// should be only one tile.
    unsigned sourceTile{0};

    /// The rectangular region in the the source tensor that this partial
    /// is from.
    ///
    /// For the very first vertices the source tensor is 2D and we may take
    /// a rectangular slice of it as the input if it is contiguous.
    ///
    /// In all other stages, the source tensor is 1D and we take a 1D slice,
    /// and sourceRows has the default value {0, 1}.
    poplar::Interval sourceCols;
    poplar::Interval sourceRows{0, 1};
  };

  /// Debug information about an output.
  struct Output {
    /// The corresponding region in the final output for the entire reduction.
    poplar::Interval outputRegion;

    /// The region in the output tensor's `data` variable that this tensor
    /// corresponds to. May not be set if this is not one contiguous region.
    ///
    /// For `firstStageRegions` this is the region of the on-tile intermediate
    /// partials variable.
    poplar::Interval dataRegion;
  };

  /// A single reduction of a set of partials to one output region. A vertex
  /// can do multiple of these.
  struct RegionReduction {
    /// The vertex that is doing this reduction.
    unsigned vertex{0};

    /// The inputs to this reduction.
    std::vector<Partial> partials;

    /// The output of this reduction.
    Output output;
  };

  /// The reductions that are done in the first compute set. This includes
  /// the single-stage reductions, and the first stage of the two-stage
  /// reductions
  struct FirstStageReduction {
    /// The single-stage reductions (may be empty).
    std::vector<RegionReduction> singleStageRegions;
    /// The first stage of the two-stage reductions (may be empty).
    std::vector<RegionReduction> firstStageRegions;

    /// The cycle estimates per vertex (TODO: T12968)
    // std::vector<std::uint64_t> vertexCycleEstimates;
  };

  /// The reductions that are done in the second compute set (if it exists).
  struct SecondStageReduction {
    /// The second stage of the two-stage reductions (may be empty).
    std::vector<RegionReduction> secondStageRegions;

    /// The cycle estimates per vertex (TODO: T12968)
    // std::vector<std::uint64_t> vertexCycleEstimates;
  };

  /// The reductions done on a tile for a particular reduction step.
  struct TileReduction {
    /// The tile number.
    unsigned tileIndex{0};

    /// The single- and first-stage reductions.
    FirstStageReduction firstStage;

    /// The second-stage reductions.
    SecondStageReduction secondStage;
  };

  /// A reduction stage (this doesn't refer to the two-stage reductions that
  /// can happen on a tile).
  struct ReductionStage {
    /// An optional label for the stage, e.g. "Intermediate to Output".
    std::string label;

    /// The set of tiles that have reductions. These may not be in order.
    std::vector<TileReduction> tiles;
  };

  /// Every stage of the reduction. There will be at least one stage.
  std::vector<ReductionStage> stages;

  /// The number of elements in the output tensor (number of columns in the
  /// 2D input).
  std::size_t outputSize;

  /// The overall reduction ratio (number of rows in the 2D input).
  std::size_t reductionRatio;
};

} // namespace popops

#endif // ReductionDebug_hpp
