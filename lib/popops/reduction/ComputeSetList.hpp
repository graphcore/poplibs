// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef ComputeSetList_hpp
#define ComputeSetList_hpp

#include <cstdint>
#include <poplar/Graph.hpp>
#include <poplar/StringRef.hpp>
#include <vector>

// This is a convenience class for using a vector<ComputeSet>/
class ComputeSetList {
public:
  // Create a wrapper around a vector<ComputeSet> that records the
  // latest compute set we have used and adds more as needed. The vector
  // must outlive this wrapper.
  explicit ComputeSetList(std::vector<poplar::ComputeSet> &css);

  // Return the compute set for the current pos() and increment pos(). If
  // there isn't one, create one with the given name.
  poplar::ComputeSet add(poplar::Graph &graph, poplar::StringRef name);

  // Return the number of times add() has been called for this list. Note
  // that the underlying vector<ComputeSet> may be larger.
  std::size_t pos() const;

  // Set pos(). An exception is thrown if newPos is greater than the
  // underlying vector's size.
  void setPos(std::size_t newPos);

  // Functions for accessing compute sets used in 2-stage reductions
  poplar::ComputeSet &getCs1(const unsigned computeSets);
  poplar::ComputeSet &getCs2(const unsigned computeSets);

private:
  std::vector<poplar::ComputeSet> &css;
  std::size_t pos_ = 0;
};

#endif // ComputeSetList_hpp
