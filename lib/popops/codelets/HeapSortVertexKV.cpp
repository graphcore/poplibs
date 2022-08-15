// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "HeapSort.hpp"

namespace popops {

template <typename KeyType, typename ValueType>
class HeapSortVertexKV : public poplar::Vertex {
public:
  poplar::InOut<poplar::Vector<KeyType>> key;
  poplar::InOut<poplar::Vector<ValueType>> value;

  void compute() {
    // The index one past the end of the max-heap in `key` and `value`.
    std::uint32_t tail = 0;

    // If we are in an intermediate step of the sort, we know the range
    // [begin() + 1, begin() + size() - 2] is in order. We can use this to
    // reestablish the max-heap-invariant in linear time, leaving only the two
    // new elements to push.
    //
    // For example, suppose `key` = [6, 1, 2, 3, 4, 5, 9], where the 6 and 9 are
    // new elements exchanged with our neighbours. The rotation will partition
    // `key` so that it's first n-2 elements are in order.
    // key := rotate(key, 1) = [1, 2, 3, 4, 5, 9, 6]
    //
    // The reverse reestablishes the max-heap in the sorted region
    // key := reverse(key.begin(), key.end-2) = [5, 4, 3, 2, 1, 9, 6]
    if (key.size() > 2) {
      rotate(key.begin(), key.begin() + 1, key.begin() + key.size());
      rotate(value.begin(), value.begin() + 1, value.begin() + value.size());

      reverse(key.begin(), key.begin() + key.size() - 3);
      reverse(value.begin(), value.begin() + value.size() - 3);
    }

    // Then we can push the two new elements into the max-heap
    // key := push(key, 9) = [9, 4, 5, 2, 1, 3, 6]
    // key := push(key, 6) = [9, 4, 6, 2, 1, 3, 5]
    //
    // We push all the elements just in case our assumption about `key` have a
    // sorted sub-range is false. This is true on the initial sort. Fortunately,
    // if the element is already in the right place, this is a very cheap push.
    for (std::size_t i = 0; i < key.size(); ++i) {
      tail = push(tail, key[i], value[i]);
    }

    // We can then repeatedly pop the elements until `key` is in order.
    // key := popall(key) = [1, 2, 3, 4, 5, 6, 9]
    while (tail > 0) {
      tail = pop(tail, key[tail - 1], value[tail - 1]);
    }

    // `key` and `value` are now sorted
  }

private:
  // Swap two elements by index
  void swap_elem(std::uint32_t tail, std::uint32_t a, std::uint32_t b) {
    if (a < tail && b < tail) {
      std::swap(key[a], key[b]);
      std::swap(value[a], value[b]);
    }
  }

  // Get the key of an element by index
  KeyType elem_key(std::uint32_t tail, std::uint32_t index) {
    if (index < tail) {
      return key[index];
    } else {
      return std::numeric_limits<KeyType>::lowest();
    }
  }

  // Get the value of an element by index
  ValueType elem_value(std::uint32_t tail, std::uint32_t index) {
    if (index < tail) {
      return value[index];
    } else {
      return std::numeric_limits<ValueType>::lowest();
    }
  }

  // Find the index of the larger child at a given index.
  // Defaults to left when they are equal.
  std::uint32_t max_child(std::uint32_t tail, std::uint32_t index) {
    if (elem_key(tail, left(index)) < elem_key(tail, right(index))) {
      return right(index);
    } else {
      return left(index);
    }
  }

  // Push a key-value pair into the max-heap
  std::uint32_t push(std::uint32_t tail, KeyType k, ValueType v) {
    std::uint32_t index = tail++;
    key[index] = k;
    value[index] = v;

    for (std::uint32_t i = index;
         i != root() && elem_key(tail, i) > elem_key(tail, parent(i));
         i = parent(i)) {
      swap_elem(tail, parent(i), i);
    }

    return tail;
  }

  // Pop a key-value pair from the max-heap
  std::uint32_t pop(std::uint32_t tail, KeyType &k, ValueType &v) {
    swap_elem(tail, root(), tail - 1);
    tail--;

    std::uint32_t i = root();
    while (elem_key(tail, i) < elem_key(tail, max_child(tail, i))) {
      const auto dst = max_child(tail, i);
      swap_elem(tail, i, dst);
      i = dst;
    }

    k = key[tail];
    v = value[tail];
    return tail;
  }
};

template class HeapSortVertexKV<float, float>;
template class HeapSortVertexKV<float, int>;
template class HeapSortVertexKV<float, unsigned>;
template class HeapSortVertexKV<float, half>;
template class HeapSortVertexKV<int, float>;
template class HeapSortVertexKV<int, int>;
template class HeapSortVertexKV<int, half>;
template class HeapSortVertexKV<unsigned, float>;
template class HeapSortVertexKV<unsigned, int>;
template class HeapSortVertexKV<unsigned, half>;
template class HeapSortVertexKV<half, float>;
template class HeapSortVertexKV<half, int>;
template class HeapSortVertexKV<half, unsigned>;
template class HeapSortVertexKV<half, half>;

} // namespace popops
