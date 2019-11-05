// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
#ifndef popops_Padder_hpp
#define popops_Padder_hpp
#include "poputil/Broadcast.hpp"
#include "poputil/exceptions.hpp"
#include <poplar/Graph.hpp>
#include <popops/Pad.hpp>
#include <vector>

namespace popops {
namespace padding {

class Padder {
public:
  Padder() {}
  virtual ~Padder() = default;

  poplar::Tensor getPaddedTensor(const poplar::Tensor &t,
                                 const std::vector<ptrdiff_t> &pLows,
                                 const std::vector<ptrdiff_t> &pUpps);

  /// \param d The dimension of the tensor to apply padding to
  /// \param pLow The amount of padding to add at the beginning of dimension d
  /// (may be negative)
  /// \param pUpp The amount of padding to add at the end of dimension d (may
  /// be negative)
  /// \return The Tensor after padding in dimension d.
  poplar::Tensor getPartPaddedTensor(const poplar::Tensor &, unsigned d,
                                     ptrdiff_t pLow, ptrdiff_t pUpp);

private:
  /// Return the Tensor to append to Tensor t during padding.
  /// \param padSize The amount of padding to apply (MUST be positive)
  /// \param padIsLow Whether the padding is for the beggining (true) of t
  /// or the end (false).
  virtual poplar::Tensor getPaddingTensor(const poplar::Tensor &t, unsigned d,
                                          ptrdiff_t padSize, bool padIsLow) = 0;

  /// Confirm that pLow and pUpp are sufficiently large:
  /// (i) pLow + t.dim(s) >= 0
  /// (ii) pUpp + t.dim(s) >= 0
  /// (iii) pLow + pUpp + t.dim(s) >= 0.
  // TODO(jamesn): T12951 we may consider removing conditions (i) and (ii)
  // for example, pLow = -100, pUpp = 100, dsize = 10 might be considered valid
  // (even though the new tensor is made of pure padding)
  void validatePadArgs(const poplar::Tensor &t, unsigned d, ptrdiff_t pLow,
                       ptrdiff_t pUpp);
};

// Shared method for ValuePadder template specialisations to map padding.
void mapPadding(poplar::Graph &graph, MappingMethod mappingMethod,
                const poplar::Tensor &tPrepad, const poplar::Tensor &padding,
                unsigned dim, bool padIsLow);

/// Padder which pads Tensors with a constant value.
template <class T1> class ValuePadder : public Padder {
public:
  ValuePadder(poplar::Graph &g, T1 v, MappingMethod mappingMethod)
      : Padder(), graph(g), val(v), mappingMethod(mappingMethod) {}
  virtual ~ValuePadder() = default;

private:
  poplar::Graph &graph;
  T1 val;
  MappingMethod mappingMethod;

  template <class T2>
  poplar::Tensor getPaddingTensorImpl(const poplar::Tensor &t, const T2 &val,
                                      unsigned dim, ptrdiff_t padSize,
                                      bool padIsLow) {
    const auto type = t.elementType();
    auto paddingShape = t.shape();
    paddingShape[dim] = static_cast<std::size_t>(padSize);
    auto c = graph.addConstant(type, paddingShape, val, "ValuePadder/padding");
    mapPadding(graph, mappingMethod, t, c, dim, padIsLow);
    return c;
  }

  poplar::Tensor getPaddingTensorImpl(const poplar::Tensor &t,
                                      const poplar::Tensor &val, unsigned dim,
                                      ptrdiff_t padSize, bool padIsLow) {
    (void)padIsLow;
    if (val.numElements() != 1) {
      throw poputil::poplibs_error("Padding tensor is not a scalar.");
    }
    // TODO: T12953 Take account of mapping method by duplicating value and
    // mapping the result?
    auto paddingShape = t.shape();
    paddingShape[dim] = static_cast<std::size_t>(padSize);
    poplar::Tensor out = val;
    poputil::broadcastToMatch(out, paddingShape);
    return out;
  }

  virtual poplar::Tensor getPaddingTensor(const poplar::Tensor &t, unsigned d,
                                          ptrdiff_t padSize,
                                          bool padIsLow) override final {
    return getPaddingTensorImpl(t, val, d, padSize, padIsLow);
  }
};

/// Padder which pads Tensors according to numpy "edge" spec.
class EdgePadder : public Padder {
public:
  EdgePadder() : Padder() {}
  virtual ~EdgePadder() = default;

private:
  virtual poplar::Tensor getPaddingTensor(const poplar::Tensor &, unsigned d,
                                          ptrdiff_t padSize,
                                          bool padIsLow) override final;
};

class ReflectPadder : public Padder {

public:
  ReflectPadder() : Padder() {}
  virtual ~ReflectPadder() = default;

private:
  virtual poplar::Tensor getPaddingTensor(const poplar::Tensor &, unsigned d,
                                          ptrdiff_t padSize,
                                          bool padIsLow) override final;
};

std::unique_ptr<Padder> getPtrPadder(Type type);
} // namespace padding
} // namespace popops

#endif
