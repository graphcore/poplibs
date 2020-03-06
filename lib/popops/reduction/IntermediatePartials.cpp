// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
#include "IntermediatePartials.hpp"

#include <cassert>
#include <poputil/exceptions.hpp>

namespace popops {

const poplar::Tensor &IntermediatePartials::data(unsigned tile) const {
  return tileData.at(tile).data;
}

std::size_t IntermediatePartials::outputElement(unsigned tile,
                                                std::size_t dataIdx) const {
  const auto &indices = tileData.at(tile).outputIndices;

  auto it = indices.find(dataIdx);
  assert(it != indices.end());

  // The map stores the difference from dataIdx to outputIdx.
  return dataIdx + it->second;
}

std::size_t IntermediatePartials::dataElement(unsigned tile,
                                              std::size_t outputIdx) const {
  const auto &indices = tileData.at(tile).dataIndices;

  auto it = indices.find(outputIdx);
  assert(it != indices.end());

  // The map stores the difference from outputIdx to dataIdx.
  return outputIdx + it->second;
}

const boost::icl::interval_set<std::size_t> &
IntermediatePartials::outputRegions(unsigned tile) const {
  return tileData.at(tile).outputRegions;
}

std::size_t IntermediatePartials::outputSize() const { return outputSize_; }

const poplar::Type &IntermediatePartials::dataType() const { return dataType_; }

const std::set<unsigned> &IntermediatePartials::tiles() const {
  return tileDataKeys;
}

void IntermediatePartials::setTensor(
    unsigned tile, poplar::Tensor &t,
    const boost::icl::interval_set<std::size_t> &outputIdxs) {

  if (t.rank() != 1)
    throw poputil::poplibs_error(
        "IntermediatePartials::setTensor() called with a tensor of rank " +
        std::to_string(t.rank()) + " (should be 1)");

  if (t.numElements() != outputIdxs.size())
    throw poputil::poplibs_error(
        "IntermediatePartials::setTensor() called with mismatched sizes. "
        "Tensor has " +
        std::to_string(t.numElements()) +
        " elements but "
        "the indices have " +
        std::to_string(outputIdxs.size()));

  auto &td = tileData[tile];
  td.data = t;
  td.outputRegions = outputIdxs;

  td.outputIndices.clear();
  td.dataIndices.clear();

  std::size_t pos = 0;
  for (const auto &ival : outputIdxs) {
    auto size = boost::icl::size(ival);

    // Add to the dataIdx->outputIdx map.
    td.outputIndices.set(std::make_pair(
        boost::icl::interval<std::size_t>::right_open(pos, pos + size),
        ival.lower() - pos));

    // Add to the outputIdx->dataIdx map.
    td.dataIndices.set(std::make_pair(ival, pos - ival.lower()));

    pos += size;
  }

  // Note that this tile is used.
  tileDataKeys.insert(tile);
}

void IntermediatePartials::setOutputSize(std::size_t s) { outputSize_ = s; }

void IntermediatePartials::setDataType(const poplar::Type &type) {
  dataType_ = type;
}

const boost::icl::interval_map<std::size_t,
                               boost::container::flat_set<unsigned>> &
IntermediatePartials::getTilesForOutput() const {

  // tilesForOutput is memo-ised via slight abuse of "if it's empty it
  // hasn't been calculated yet", which should be fine.
  if (tilesForOutput.empty()) {
    for (const auto &tit : tileData) {
      for (const auto &outputRegion : tit.second.outputRegions) {
        boost::container::flat_set<unsigned> thisTile;
        thisTile.insert(tit.first);
        tilesForOutput.add(std::make_pair(outputRegion, thisTile));
      }
    }
  }

  return tilesForOutput;
}

} // namespace popops
