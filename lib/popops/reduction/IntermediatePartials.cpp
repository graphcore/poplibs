#include "IntermediatePartials.hpp"

#include <poputil/exceptions.hpp>

namespace popops {

const poplar::Tensor &IntermediatePartials::data(unsigned tile) const {
  return tileData.at(tile).data;
}

std::size_t IntermediatePartials::outputElement(unsigned tile,
                                                 std::size_t dataIdx) const {
  const auto &indices = tileData.at(tile).outputIndices;
  // Get the interval that dataIdx is in. E.g. if dataIdx is 10
  // this might return [5, 15)
  auto it = indices.find(dataIdx);

  assert(it != indices.end());

  // Get the output element for the first data index in this interval
  // and add on the offset from it to dataIdx.
  return it->second + (dataIdx - it->first.lower());
}

std::size_t IntermediatePartials::dataElement(unsigned tile,
                                               std::size_t outputIdx) const {
  const auto &indices = tileData.at(tile).dataIndices;
  // Get the interval that dataIdx is in. E.g. if dataIdx is 10
  // this might return [5, 15)
  auto it = indices.find(outputIdx);

  assert(it != indices.end());

  // Get the output element for the first data index in this interval
  // and add on the offset from it to dataIdx.
  return it->second + (outputIdx - it->first.lower());
}

const boost::icl::interval_set<std::size_t> &
IntermediatePartials::outputRegions(unsigned tile) const {
  return tileData.at(tile).outputRegions;
}

std::size_t IntermediatePartials::outputSize() const {
  return outputSize_;
}

const poplar::Type &IntermediatePartials::dataType() const {
  return dataType_;
}

const std::set<unsigned> &IntermediatePartials::tiles() const {
  return tileDataKeys;
}

void IntermediatePartials::setTensor(
    unsigned tile,
    poplar::Tensor &t,
    const boost::icl::interval_set<std::size_t> &outputIdxs) {

  if (t.rank() != 1)
    throw poputil::poplib_error(
        "IntermediatePartials::setTensor() called with a tensor of rank "
        + std::to_string(t.rank()) + " (should be 1)");

  if (t.numElements() != outputIdxs.size())
    throw poputil::poplib_error(
        "IntermediatePartials::setTensor() called with mismatched sizes. "
        "Tensor has " + std::to_string(t.numElements()) + " elements but "
        "the indices have " + std::to_string(outputIdxs.size()));

  auto &td = tileData[tile];
  td.data = t;
  td.outputRegions = outputIdxs;

  td.outputIndices.clear();
  td.dataIndices.clear();

  // Add to the dataIdx->outputIdx map.
  std::size_t pos = 0;
  for (const auto &ival : outputIdxs) {
    auto size = ival.upper() - ival.lower();

    td.outputIndices.set(std::make_pair(
        boost::icl::interval<std::size_t>::right_open(pos, pos + size),
        ival.lower()));

    pos += size;
  }

  // Add to the outputIdx->dataIdx map.
  pos = 0;
  for (const auto &ival : outputIdxs) {
    auto size = ival.upper() - ival.lower();

    td.dataIndices.set(std::make_pair(ival, pos));

    pos += size;
  }

  // Note that this tile is used.
  tileDataKeys.insert(tile);
}

void IntermediatePartials::setOutputSize(std::size_t s) {
  outputSize_ = s;
}

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

}
