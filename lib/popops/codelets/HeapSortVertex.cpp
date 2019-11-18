#include "HeapSort.hpp"

namespace popops {

template <typename ValueType> class HeapSortVertex : public poplar::Vertex {
public:
  poplar::InOut<poplar::Vector<ValueType>> out;

  bool compute() {
    // The index one past the end of the max-heap in `out`.
    std::uint32_t tail = 0;

    // If we are in an intermediate step of the sort, we know the range
    // [begin() + 1, begin() + size() - 2] is in order. We can use this to
    // reestablish the max-heap-invariant in linear time, leaving only the two
    // new elements to push.
    //
    // For example, suppose `out` = [6, 1, 2, 3, 4, 5, 9], where the 6 and 9 are
    // new elements exchanged with our neighbours. The rotation will partition
    // `out` so that it's first n-2 elements are in order.
    // out := rotate(out, 1) = [1, 2, 3, 4, 5, 9, 6]
    //
    // The reverse reestablishes the max-heap in the sorted region
    // out := reverse(out.begin(), out.end-2) = [5, 4, 3, 2, 1, 9, 6]
    if (out.size() > 2) {
      rotate(out.begin(), out.begin() + 1, out.begin() + out.size());
      reverse(out.begin(), out.begin() + out.size() - 3);
    }

    // Then we can push the two new elements into the max-heap
    // out := push(out, 9) = [9, 4, 5, 2, 1, 3, 6]
    // out := push(out, 6) = [9, 4, 6, 2, 1, 3, 5]
    //
    // We push all the elements just in case our assumption about `out` have a
    // sorted sub-range is false. This is true on the initial sort. Fortunately,
    // if the element is already in the right place, this is a very cheap push.
    for (std::size_t i = 0; i < out.size(); ++i) {
      tail = push(tail, out[i]);
    }

    // We can then repeatedly pop the elements until `out` is in order.
    // out := popall(out) = [1, 2, 3, 4, 5, 6, 9]
    while (tail > 0) {
      tail = pop(tail, out[tail - 1]);
    }

    // `out` is now sorted
    return true;
  }

private:
  // Swap two elements by index
  void swap_elem(std::uint32_t tail, std::uint32_t a, std::uint32_t b) {
    if (a < tail && b < tail) {
      std::swap(out[a], out[b]);
    }
  }

  // Get the value of an element by index
  ValueType elem(std::uint32_t tail, std::uint32_t index) {
    if (index < tail) {
      return out[index];
    } else {
      return std::numeric_limits<ValueType>::lowest();
    }
  }

  // Find the index of the larger child at a given index.
  // Defaults to left when they are equal.
  std::uint32_t max_child(std::uint32_t tail, std::uint32_t index) {
    if (elem(tail, left(index)) < elem(tail, right(index))) {
      return right(index);
    } else {
      return left(index);
    }
  }

  // Push an element into the max-heap
  std::uint32_t push(std::uint32_t tail, ValueType value) {
    std::uint32_t index = tail++;
    out[index] = value;

    for (std::uint32_t i = index;
         i != root() && elem(tail, i) > elem(tail, parent(i)); i = parent(i)) {
      swap_elem(tail, parent(i), i);
    }

    return tail;
  }

  // Pop an element from the max-heap
  std::uint32_t pop(std::uint32_t tail, ValueType &value) {
    swap_elem(tail, root(), tail - 1);
    tail--;

    std::uint32_t i = root();
    while (elem(tail, i) < elem(tail, max_child(tail, i))) {
      const auto dst = max_child(tail, i);
      swap_elem(tail, i, dst);
      i = dst;
    }

    value = out[tail];
    return tail;
  }
};

template class HeapSortVertex<float>;
template class HeapSortVertex<int>;
template class HeapSortVertex<half>;

} // namespace popops
