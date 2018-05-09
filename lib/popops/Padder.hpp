// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
#ifndef popops_Padder_hpp
#define popops_Padder_hpp
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
  poplar::Tensor getPartPaddedTensor(const poplar::Tensor &,
                                     unsigned d,
                                     ptrdiff_t pLow,
                                     ptrdiff_t pUpp);

private:
  /// Return the Tensor to append to Tensor t during padding.
  /// \param padSize The amount of padding to apply (MUST be positive)
  /// \param padIsLow Whether the padding is for the beggining (true) of t
  /// or the end (false).
  virtual poplar::Tensor getPaddingTensor(const poplar::Tensor &t,
                                          unsigned d,
                                          ptrdiff_t padSize,
                                          bool padIsLow) = 0;

  /// Confirm that pLow and pUpp are sufficiently large:
  /// (i) pLow + t.dim(s) >= 0
  /// (ii) pUpp + t.dim(s) >= 0
  /// (iii) pLow + pUpp + t.dim(s) >= 0.
  // TODO(jamesn) we may consider removing conditions (i) and (ii)
  // for example, pLow = -100, pUpp = 100, dsize = 10 might be considered valid
  // (even though the new tensor is made of pure padding)
  void validatePadArgs(const poplar::Tensor &t,
                       unsigned d,
                       ptrdiff_t pLow,
                       ptrdiff_t pUpp);
};

/// Padder which pads Tensors with a constant value.
class ConstPadder : public Padder {
public:
  ConstPadder(poplar::Graph &g, float v) : Padder(), graph(g), val(v) {}
  virtual ~ConstPadder() = default;

private:
  poplar::Graph &graph;
  /// The constant to perform padding with.
  // TODO(jamesn) consider the option of
  // this the padding value to be integral
  float val;
  virtual poplar::Tensor getPaddingTensor(const poplar::Tensor &t,
                                          unsigned d,
                                          ptrdiff_t padSize,
                                          bool padIsLow) override final;
};

/// Padder which pads Tensors according to numpy "edge" spec.
class EdgePadder : public Padder {
public:
  EdgePadder() : Padder() {}
  virtual ~EdgePadder() = default;

private:
  virtual poplar::Tensor getPaddingTensor(const poplar::Tensor &,
                                          unsigned d,
                                          ptrdiff_t padSize,
                                          bool padIsLow) override final;
};

class ReflectPadder : public Padder {

public:
  ReflectPadder() : Padder() {}
  virtual ~ReflectPadder() = default;

private:
  virtual poplar::Tensor getPaddingTensor(const poplar::Tensor &,
                                          unsigned d,
                                          ptrdiff_t padSize,
                                          bool padIsLow) override final;
};

std::unique_ptr<Padder> getPtrPadder(Type type);
} // namespace padding
} // namespace popops

#endif
