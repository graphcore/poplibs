// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
#ifndef IntermediatePartials_hpp
#define IntermediatePartials_hpp

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <boost/container/flat_set.hpp>
#include <boost/icl/interval_map.hpp>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

namespace popops {

/// IntermediatePartials stores the result of any stage of the reduction where
/// the data has been reduced as much as possible on each tile.
///
/// Each tile stores a 1D `data` tensor that contains the reduced values.
/// Each element in the `data` tensor corresponds to an element in the
/// final reduction output.
///
/// An interval map from the index into the `data` tensor to the index into
/// the final reduction result is stored. This map is stored in various
/// redundant ways to enable efficient lookup.
///
class IntermediatePartials {
public:
  /// Get the `data` tensor for a given tile, i.e. the output of the reduction
  /// for that tile. Throws an exception if the tile has no data tensor.
  const poplar::Tensor &data(unsigned tile) const;

  /// Get the index of an output element for a tile, given the
  /// index into the `data` tensor of that tile.
  std::size_t outputElement(unsigned tile, std::size_t dataIdx) const;

  /// Get the index of a `data` element for a tile, given the
  /// output index.
  std::size_t dataElement(unsigned tile, std::size_t outputIdx) const;

  /// Get the output regions stored on a tile. This returns the value set by
  /// setOutputRegions().
  const boost::icl::interval_set<std::size_t> &
  outputRegions(unsigned tile) const;

  /// Get the set of tiles that are used. I.e. every tile that setTensor()
  /// has been called on.
  const std::set<unsigned> &tiles() const;

  /// Set an output tensor for a tile that stores the intermediate partials.
  /// This must be a 1D tensor otherwise an exception is thrown.
  ///
  /// This also records which regions in the final output tensor this 1D
  /// tensor corresponds to. outputIdxs must be a set of regions whose size
  /// is the same as the number of elements in t.
  void setTensor(unsigned tile, poplar::Tensor &t,
                 const boost::icl::interval_set<std::size_t> &outputIdxs);

  /// Get the number of output elements of the entire reduction. This should
  /// be equal to the size of the union of every outputRegions() result.
  /// However it is actually set manually with setOutputSize(). Purely
  /// stored here for convenience.
  std::size_t outputSize() const;

  /// For convenience, set the output size of the entire reduction.
  void setOutputSize(std::size_t s);

  /// The type of data stored in the `data` tensors. This is actually
  /// set manually with setDataType().
  const poplar::Type &dataType() const;

  /// For convenience set the data type of the data tensors stored here.
  void setDataType(const poplar::Type &type);

  /// Return the set of tiles that has partials for each output index. A
  /// flat_set is used to reduce the memory overhead of std::set since these
  /// usually only contain a few elements and the only thing that is done is
  /// to iterate over the members. This is memoized.
  const boost::icl::interval_map<std::size_t,
                                 boost::container::flat_set<unsigned>> &
  getTilesForOutput() const;

private:
  struct TileData {
    // 1D tensor containing the reduced partials on the tile.
    poplar::Tensor data;

    // The following members all store the same information in different ways.

    // outputIndices is the map from the data index to the corresponding index
    // in the final output tensor. This could be stored as a vector<size_t>
    // but to make it a bit more efficient we do this region-wise. That is,
    // assume we have the following vector:
    //
    //    `data` index:  0 1 2 3   4  5  6   7  8  9 10 11 12 13 14
    //  `output` index:  4 5 6 7, 10 11 12, 20 30 31 32 33 34 35 36
    //
    // We can store it as an ICL interval_map, like this:
    //
    //    `data` index:  0 1 2 3   4 5 6    7    8 9 10 11 12 13 14
    // `outputIndices`: [4 . . .] [6 . .] [13] [22 .  .  .  .  .  .]
    //
    // We actually store the difference between the output index and the data
    // data index, rather than storing the data index explicitly. This is to
    // take advantage of the fact that ICL automatically merges adjacent regions
    // with the same value. It also makes looking up values easier since
    // we don't need to explicitly find the start of each region.
    //
    // partial_enricher is necessary because by default the map doesn't
    // store identity elements (0).
    boost::icl::interval_map<std::size_t, std::size_t,
                             boost::icl::partial_enricher>
        outputIndices;

    // We'll also store the reverse map, from `output` index to `data` index.
    // Similarly to the above we'll use an interval_map and store the
    // difference between the indices, so for the above example we'd have:
    //
    // `output` index:  4 5 6 7    10 11 12    20    30 31 32 33 34 35 36
    //   `data` index: [-4 . . .] [-6  .  .] [-13] [-22  .  .  .  .  .  .]
    //
    boost::icl::interval_map<std::size_t, std::size_t,
                             boost::icl::partial_enricher>
        dataIndices;

    // Finally we'll store it as a set of output regions. In this example it is
    //
    //    [4 8)  [10 13)  [20 21)  [30 37)
    boost::icl::interval_set<std::size_t> outputRegions;
  };

  // This is a map from tile id to the output tensor data for that tile.
  std::map<unsigned, TileData> tileData;

  // If tileData.keys() existed, this would be it.
  std::set<unsigned> tileDataKeys;

  // A map from output index to a list of tiles that contain partials for it.
  mutable boost::icl::interval_map<std::size_t,
                                   boost::container::flat_set<unsigned>>
      tilesForOutput;

  // For convenience, the total number of reduction outputs.
  std::size_t outputSize_ = 0;

  // The type of the data in tiles[.].result;
  poplar::Type dataType_;
};

} // namespace popops

#endif // IntermediatePartials_hpp
