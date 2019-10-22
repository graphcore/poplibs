// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_support_vv_iterator_hpp
#define poplibs_support_vv_iterator_hpp

#include <iterator>
#include <vector>

namespace poplibs {

// Private utility functions for vv_iterator and vv_const_iterator to reduce
// code duplication.
class vv_iterator_util {

  template <typename T> friend class vv_iterator;
  template <typename T> friend class vv_const_iterator;

  template <typename T>
  static void incrementIndices(const std::vector<std::vector<T>> &vv,
                               std::size_t &idxInner, std::size_t &idxOuter) {
    // If we haven't reached the end of this sub-vector.
    if (idxInner + 1 < vv[idxOuter].size()) {
      // Go to the next element.
      ++idxInner;
    } else {
      // Otherwise skip to the next sub-vector, and keep skipping over empty
      // ones until we reach a non-empty one or the end.
      do {
        ++idxOuter;
      } while (idxOuter < vv.size() && vv[idxOuter].empty());

      // Go to the start of this vector.
      idxInner = 0;
    }
  }

  template <typename T>
  static void decrementIndices(const std::vector<std::vector<T>> &vv,
                               std::size_t &idxInner, std::size_t &idxOuter) {
    // If we haven't reached the start of this sub-vector.
    if (idxInner > 0) {
      // Go to the previous element.
      --idxInner;
    } else {
      // Otherwise skip to the previous sub-vector, and keep skipping over empty
      // ones until we reach a non-empty one.
      do {
        --idxOuter;
      } while ((*vv)[idxOuter].empty());

      // Go to the end of this vector.
      idxInner = (*vv)[idxOuter].size() - 1;
    }
  }
};

/// This is an iterator over a vector<vector<T>>. It is not a random access
/// iterator so you can't use it for sort() but it is useful just for generic
/// for-loops.
///
/// Example use:
///
///     vector<vector<int>> foo = {{1, 2, 3}, {}, {4}, {5, 6}};
///     auto begin = vv_iterator<int>::begin(foo);
///     auto end = vv_iterator<int>::end(foo);
///     std::reverse(begin, end);
///
/// foo is now {{6, 5, 4}, {}, {3}, {2, 1}}
///
template <typename T>
class vv_iterator : public std::iterator<std::bidirectional_iterator_tag, T> {
public:
  static vv_iterator<T> begin(std::vector<std::vector<T>> &vv) {
    return vv_iterator(&vv, 0, 0);
  }
  static vv_iterator<T> end(std::vector<std::vector<T>> &vv) {
    return vv_iterator(&vv, vv.size(), 0);
  }

  vv_iterator() = default;
  // ++prefix operator
  vv_iterator &operator++() {
    vv_iterator_util::incrementIndices(*vv, idxInner, idxOuter);
    return *this;
  }
  // --prefix operator
  vv_iterator &operator--() {
    vv_iterator_util::decrementIndices(*vv, idxInner, idxOuter);
    return *this;
  }
  // postfix++ operator
  vv_iterator operator++(int) {
    T retval = *this;
    ++(*this);
    return retval;
  }
  // postfix-- operator
  vv_iterator operator--(int) {
    T retval = *this;
    --(*this);
    return retval;
  }
  bool operator==(const vv_iterator &other) const {
    return other.vv == vv && other.idxOuter == idxOuter &&
           other.idxInner == idxInner;
  }
  bool operator!=(const vv_iterator &other) const { return !(*this == other); }
  T &operator*() const { return (*vv)[idxOuter][idxInner]; }
  T *operator->() const { return &(*vv)[idxOuter][idxInner]; }

private:
  vv_iterator(std::vector<std::vector<T>> *vv, std::size_t idxOuter,
              std::size_t idxInner)
      : vv(vv), idxOuter(idxOuter), idxInner(idxInner) {}

  std::vector<std::vector<T>> *vv = nullptr;
  std::size_t idxOuter = 0;
  std::size_t idxInner = 0;
};

/// A const version of vv_iterator.
template <typename T>
class vv_const_iterator
    : public std::iterator<std::bidirectional_iterator_tag, T> {
public:
  static vv_const_iterator<T> begin(const std::vector<std::vector<T>> &vv) {
    return vv_const_iterator(&vv, 0, 0);
  }
  static vv_const_iterator<T> end(const std::vector<std::vector<T>> &vv) {
    return vv_const_iterator(&vv, vv.size(), 0);
  }

  // Create a const iterator from a non-const one.
  vv_const_iterator(const vv_iterator<T> &other)
      : vv(other.vv), idxOuter(other.idxOuter), idxInner(other.idxInner) {}

  vv_const_iterator() = default;
  // ++prefix operator
  vv_const_iterator &operator++() {
    vv_iterator_util::incrementIndices(*vv, idxInner, idxOuter);
    return *this;
  }
  // --prefix operator
  vv_const_iterator &operator--() {
    // If we haven't reached the start of this sub-vector.
    vv_iterator_util::decrementIndices(*vv, idxInner, idxOuter);
    return *this;
  }
  // postfix++ operator
  vv_const_iterator operator++(int) {
    T retval = *this;
    ++(*this);
    return retval;
  }
  // postfix-- operator
  vv_const_iterator operator--(int) {
    T retval = *this;
    --(*this);
    return retval;
  }
  bool operator==(const vv_const_iterator &other) const {
    return other.vv == vv && other.idxOuter == idxOuter &&
           other.idxInner == idxInner;
  }
  bool operator!=(const vv_const_iterator &other) const {
    return !(*this == other);
  }
  const T &operator*() const { return (*vv)[idxOuter][idxInner]; }
  const T *operator->() const { return &(*vv)[idxOuter][idxInner]; }

private:
  vv_const_iterator(const std::vector<std::vector<T>> *vv, std::size_t idxOuter,
                    std::size_t idxInner)
      : vv(vv), idxOuter(idxOuter), idxInner(idxInner) {}

  const std::vector<std::vector<T>> *vv = nullptr;
  std::size_t idxOuter = 0;
  std::size_t idxInner = 0;
};

} // namespace poplibs

#endif // poplibs_support_vv_iterator_hpp
