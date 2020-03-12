// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <limits>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <utility>

namespace popops {

template <typename BIter> static void reverse(BIter begin, BIter end) {
  while (begin < end) {
    end--;
    std::swap(*begin, *end);
    begin++;
  }
}

template <typename BIter>
static void rotate(BIter begin, BIter new_begin, BIter end) {
  reverse(begin, new_begin);
  reverse(new_begin, end);
  reverse(begin, end);
}

static std::uint32_t root() { return 0; }

static std::uint32_t parent(std::uint32_t index) { return (index - 1) / 2; }

static std::uint32_t left(std::uint32_t index) { return (index * 2) + 1; }

static std::uint32_t right(std::uint32_t index) { return (index * 2) + 2; }

} // namespace popops
