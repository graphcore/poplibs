#include "ComputeSetList.hpp"

#include <cassert>

using namespace poplar;

ComputeSetList::ComputeSetList(std::vector<ComputeSet> &css) : css(css) {}

ComputeSet ComputeSetList::add(Graph &graph, StringRef name) {
  if (pos_ > css.size()) {
    throw std::logic_error("ComputeSetList::add() with pos " +
                           std::to_string(pos_) + " and size " +
                           std::to_string(css.size()));
  } else if (pos_ == css.size()) {
    // Add a new compute set.
    css.emplace_back(graph.addComputeSet(name));
  }
  return css[pos_++];
}

std::size_t ComputeSetList::pos() const { return pos_; }

void ComputeSetList::setPos(std::size_t newPos) {
  if (newPos > css.size())
    throw std::logic_error("ComputeSetList::setPos(" + std::to_string(newPos) +
                           ")" + " which is > " + std::to_string(css.size()));

  pos_ = newPos;
}
