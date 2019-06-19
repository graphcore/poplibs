#ifndef POPLIBS_SUPPORT_MULTI_ARRAY_HPP
#define POPLIBS_SUPPORT_MULTI_ARRAY_HPP

#include <memory>
#include <sstream>

#include <boost/container/small_vector.hpp>
#include <boost/range/numeric.hpp>
#include <boost/range/iterator_range.hpp>

namespace poplibs_support {

using MultiArrayShape = boost::container::small_vector<std::size_t, 4>;
using MultiArrayShapeRange =
  boost::iterator_range<MultiArrayShape::const_iterator>;

// Similar to Boost.MultiArray except the dimensions are specified at runtime.
// Internally the data is stored in row-major order aka boost::c_storage_order.
template <typename T>
class MultiArray {
  using Shape = MultiArrayShape;
  using ShapeRange = MultiArrayShapeRange;

public:
  class Reference {
  public:
    Reference(const ShapeRange shape, T *data) : shape(shape), data(data) {}

    // moveable, non-copyable, but support assignment from inner-most dim.
    Reference(Reference &&) = default;
    Reference(const Reference &) = delete;

    T &operator=(const Reference &o) {
      return *this = static_cast<const T &>(o);
    }

    T &operator=(const T &value) {
      T &d(*this);
      d = value;
      return d;
    }

    Reference operator[](const std::size_t idx) {
      return slice<Reference>(shape, data, idx);
    }

    operator const T &() const {
      return cast<const T &>(shape, data);
    }

    operator T &() {
      return cast<T &>(shape, data);
    }

  private:
    ShapeRange shape;
    T *data;
  };

  class ConstReference {
  public:
    ConstReference(const ShapeRange shape, const T *const data)
    : shape(shape), data(data) {}

    // moveable, but non-copyable. can enforce this at compile time as this is
    // a const reference and so cannot write to it's underlying data.
    ConstReference(ConstReference &&) = default;
    ConstReference &operator=(ConstReference &&) = default;
    ConstReference(const ConstReference &) = delete;
    ConstReference &operator=(const ConstReference &) = delete;

    ConstReference operator[](const std::size_t idx) const {
      return slice<ConstReference>(shape, data, idx);
    }

    operator const T &() const {
      return cast<const T &>(shape, data);
    }

  private:
    ShapeRange shape;
    const T *data;
  };

  MultiArray(std::initializer_list<std::size_t> shape)
  : shape_{shape}, data_{nullptr} {
    init();
  }

  MultiArray(const ShapeRange shape)
  : shape_{std::begin(shape), std::end(shape)}, data_{nullptr} {
    init();
  }

  T *data() const {
    return data_.get();
  }

  ShapeRange shape() const {
    return shape_;
  }

  std::size_t size() const {
    return shape_[0];
  }

  std::size_t numElements() const {
    return product(shape_);
  }

  std::size_t numDimensions() const {
    return shape_.size();
  }

  Reference operator[](const std::size_t idx) {
    return slice<Reference>(shape_, data_.get(), idx);
  }

  Reference operator[](const ShapeRange indices) {
    return slice<Reference>(shape_, data_.get(), indices);
  }

  ConstReference operator[](const std::size_t idx) const {
    return slice<ConstReference>(shape_, data_.get(), idx);
  }

  ConstReference operator[](const ShapeRange indices) const {
    return slice<ConstReference>(shape_, data_.get(), indices);
  }

private:
  void init() {
    if (shape_.empty()) {
      throw std::runtime_error("Cannot create a MultiArray with a rank of 0.");
    }

    const auto size = numElements();
    data_.reset(new T[size]);
    std::fill_n(data_.get(), size, T{});
  }

  // get a view of the data at the next dimension in, offsetted by index.
  template <typename R, typename D>
  static R slice(const ShapeRange shape, D data, const std::size_t idx) {
    if (shape.empty()) {
      throw std::runtime_error(
        "Cannot slice MultiArray, already at inner-most dimension.");
    }

    if (idx >= shape[0]) {
      std::stringstream ss;
      ss << "MultiArray out of bounds error: " << idx << " >= " << shape[0]
         << ".";
      throw std::out_of_range(ss.str());
    }

    // remove the outer-most dimension from the shape.
    auto newShape = shape;
    newShape.advance_begin(1);

    // get the pointer to where this slice begins.
    auto offset = std::next(data, idx * product(newShape));

    return R{newShape, offset};
  }

  // get a view of the data at the next N-dimensions in, as described by the
  // range of indices.
  template <typename R, typename D>
  static R slice(const ShapeRange shape, D data, const ShapeRange indices) {
    if (shape.empty()) {
      throw std::runtime_error(
        "Cannot slice MultiArray, already at inner-most dimension.");
    }

    T *offset = data;
    auto newShape = shape;
    for (unsigned i = 0; i < shape.size(); ++i) {
      if (indices[i] >= shape[i]) {
        std::stringstream ss;
        ss << "MultiArray out of bounds error: " << indices[i] << " >= "
           << shape[i] << ".";
        throw std::out_of_range(ss.str());
      }

      // remove the outer-most dimension from the shape.
      newShape.advance_begin(1);

      // get the pointer to where this slice begins.
      offset = std::next(offset, indices[i] * product(newShape));
    }

    return R{newShape, offset};
  }

  // if this is a reference to a single element inside the multi-array (ie,
  // the shape is empty), then we can implicitly cast the reference to the
  // underlying value. same goes for assignment.
  template <typename R, typename D>
  static R cast(const ShapeRange shape, D data) {
    if (!shape.empty()) {
      throw std::runtime_error(
        "Cannot cast MultiArray slice to underlying data as this view is not "
        "of the inner-most dimension.");
    }

    return *data;
  }

  static std::size_t product(const ShapeRange shape) {
    return boost::accumulate(shape, 1ul, std::multiplies<std::size_t>{});
  }

  Shape shape_;
  std::unique_ptr<T[]> data_;
};

// given a MultiArray shape, call the functor with each value from the
// cartesian product of all the indices. this is basically a way of looping
// through as if you were to handwrite nested loops except the depth is decided
// at runtime. eg.
//  forEachIndex({3, 2, 1}, [](ShapeRange indices) { printf(indices); });
// will print:
//  0, 0, 0
//  0, 1, 0
//  1, 0, 0
//  1, 1, 0
//  2, 0, 0
//  2, 1, 0
// these shapes can be passed to a MultiArray via the subscript operator to get
// a slice/value at that point.
class ForEachIndex {
public:
  template <typename Fn>
  void operator()(const MultiArrayShapeRange shape, const Fn &fn) const {
    MultiArrayShape indices(shape.size(), 0ul);
    step(shape, fn, indices, 0);
  }

private:
  template <typename Fn>
  static void step(const MultiArrayShapeRange shape,
                   const Fn &fn,
                   MultiArrayShape &indices,
                   const std::size_t depth) {
    if (depth == shape.size()) {
      fn(MultiArrayShapeRange{indices});
    } else {
      for (unsigned i = 0; i < shape[depth]; ++i) {
        indices[depth] = i;
        step(shape, fn, indices, depth + 1);
      }
    }
  }
};

constexpr ForEachIndex forEachIndex{};

} // namespace poplibs_support

#endif // POPLIB_SUPPORT_MULTI_ARRAY_HPP
