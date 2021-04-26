// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef ReductionConnection_hpp
#define ReductionConnection_hpp

#include "ComputeSetList.hpp"
#include "Reduction.hpp"
#include "ReductionVertexDefs.hpp"

#include "popops/Reduce.hpp"

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

#include <boost/range.hpp>
#include <boost/variant.hpp>

#include <iosfwd>
#include <vector>

namespace popops {

// Partials for reduction can be stored in two ways -
// 1. If the partials are all in the same region, each of the same length
//    and spaced regularly they can be represented with a single tensor, offset
//    into that tensor and stride.  This provides memory layout information.
// 2. If any of the criteria for RegularPartials is not met we store a vector
//    or tensors instead.  We have no information about the memory layout.
struct RegularPartials {
  // By example, if partial's data belongs to the numbered columns:
  // 0 1 2 3
  // 0 1 2 3
  // 4 5 6 7
  // 4 5 6 7
  // 0 1 2 3
  // 0 1 2 3
  // 4 5 6 7
  // 4 5 6 7
  // 0 1 2 3
  // 0 1 2 3
  // When reducing 0,1 their locations in the data region are described by:
  // an inner pattern: offset=0 stride=4 (innerFactor=1, outerFactor = 2)
  // An outer pattern: outerStride = 16 numOuterStrides = 3
  // When reducing 2,3 offset = 2, other parameters are the same as 0,1
  //
  // When reducing 4,5 their locations in the data region are described by:
  // an inner pattern: offset=8 stride=4 (innerFactor=1, outerFactor = 2)
  // An outer pattern: outerStride = 16 numOuterStrides = 2
  // When reducing 6,7 offset = 10, other parameters are the same as 4,5
  std::vector<poplar::Tensor> data;
  unsigned offset;
  unsigned stride;
  unsigned outerStride = 0;
  unsigned numOuterStrides = 1;
};
struct IrregularPartials {
  std::vector<poplar::Tensor> data;
};

/// This structure represents the reduction of a set of 1D input regions
/// to a single 1D output region. One reduction vertex can reduce a set
/// of these.
///
/// The regions are represented by as poplar::Tensors. The size of the partial
/// regions must be equal to a multiple of the size of the output region.
///
/// The shape of the partial and output tensors is ignored - they are treated
/// as if they are 1D. partials that are longer than the output tensor are
/// wrapped.
///
struct RegionReduction {
  // The output region.
  poplar::Tensor output;
  // The input regions - optionally either regular or irregular
  boost::variant<RegularPartials, IrregularPartials> partials;
  // innerFactor indicates that each partial contains innerFactor elements
  // to be reduced into the 1st output element, followed by innerFactor elements
  // to be reduced into the second etc...  A two stage approach is used to
  // implement this.
  unsigned innerFactor = 1;
  unsigned outerFactor = 1;

  // Functions to access the partials variants.
  bool regularPartials() const {
    return partials.type() == typeid(RegularPartials);
  }

  // Access partials
  const std::vector<poplar::Tensor> &getPartials() const {
    if (regularPartials()) {
      return boost::get<RegularPartials>(partials).data;
    } else {
      return boost::get<IrregularPartials>(partials).data;
    }
  }

  std::vector<poplar::Tensor> &getPartials() {
    if (regularPartials()) {
      return boost::get<RegularPartials>(partials).data;
    } else {
      return boost::get<IrregularPartials>(partials).data;
    }
  }

  // Get the number of separate contiguous regions that the partials occupy
  std::size_t getNumPartials() const {
    if (regularPartials()) {
      bool partialsGroupContiguous = getStride() == output.numElements();
      auto factor = partialsGroupContiguous ? 1 : outerFactor;
      return factor * boost::get<RegularPartials>(partials).numOuterStrides;
    } else {
      return boost::get<IrregularPartials>(partials).data.size();
    }
  }

  // Get the number of actual partials elements that will be reduced
  // (Excluding those in the regular partials Tensor that won't be reduced)
  unsigned getNumPartialsElements() const {
    if (regularPartials()) {
      return innerFactor * outerFactor * output.numElements() *
             boost::get<RegularPartials>(partials).numOuterStrides;
    } else {
      return concat(boost::get<IrregularPartials>(partials).data).numElements();
    }
  }

  unsigned getNumPartialsElementsPerOuterStride() const {
    if (regularPartials()) {
      return innerFactor * outerFactor * output.numElements();
    } else {
      return concat(boost::get<IrregularPartials>(partials).data).numElements();
    }
  }

  // Offset
  unsigned getOffset() const {
    if (regularPartials()) {
      return boost::get<RegularPartials>(partials).offset;
    } else {
      throw poputil::poplibs_error(
          "Irregular reduction partials have no offset");
    }
  }
  unsigned &getOffset() {
    if (regularPartials()) {
      return boost::get<RegularPartials>(partials).offset;
    } else {
      throw poputil::poplibs_error(
          "Irregular reduction partials have no offset");
    }
  }

  // Stride
  unsigned getStride() const {
    if (regularPartials()) {
      return boost::get<RegularPartials>(partials).stride;
    } else {
      throw poputil::poplibs_error(
          "Irregular reduction partials have no stride");
    }
  }

  unsigned &getStride() {
    if (regularPartials()) {
      return boost::get<RegularPartials>(partials).stride;
    } else {
      throw poputil::poplibs_error(
          "Irregular reduction partials have no stride");
    }
  }

  // Outer Stride
  unsigned getOuterStride() const {
    if (regularPartials()) {
      return boost::get<RegularPartials>(partials).outerStride;
    } else {
      throw poputil::poplibs_error(
          "Irregular reduction partials have no outer stride");
    }
  }

  unsigned &getOuterStride() {
    if (regularPartials()) {
      return boost::get<RegularPartials>(partials).outerStride;
    } else {
      throw poputil::poplibs_error(
          "Irregular reduction partials have no outer stride");
    }
  }

  // Num Outer Strides
  unsigned getNumOuterStrides() const {
    if (regularPartials()) {
      return boost::get<RegularPartials>(partials).numOuterStrides;
    } else {
      throw poputil::poplibs_error(
          "Irregular reduction partials have no num outer stride");
    }
  }

  unsigned &getNumOuterStrides() {
    if (regularPartials()) {
      return boost::get<RegularPartials>(partials).numOuterStrides;
    } else {
      throw poputil::poplibs_error(
          "Irregular reduction partials have no num outer strides");
    }
  }
};

inline std::ostream &operator<<(std::ostream &os, const RegionReduction &r) {
  if (r.regularPartials()) {
    const auto &partials = boost::get<RegularPartials>(r.partials);
    os << "{ inner = " << r.innerFactor << ", outer = " << r.outerFactor
       << ", numPartials = " << r.getNumPartials()
       << ", numPartialsElements = " << r.getNumPartialsElements()
       << "; regular partials: offset = " << partials.offset
       << ", stride = " << partials.stride
       << ", outerStride = " << partials.outerStride
       << ", numOuterStrides = " << partials.numOuterStrides << " }";
  } else {
    os << "{ inner = " << r.innerFactor
       << ", numPartials = " << r.getNumPartials()
       << ", numPartialsElements = " << r.getNumPartialsElements()
       << "; irregular partials }";
  }
  return os;
}

/// Add vertices to the graph to perform the given reductions on the specified
/// tile and connect the vertex inputs and outputs.
///
/// If every partial region is exactly the same size as its output a more
/// optimal code path is automatically used.
///
/// The case where the output region is small an on-tile two-stage reduction
/// may be performed. This is why a vector of compute sets is passed
/// instead of a single one. The vector will be enlarged to the number
/// of compute sets required if necessary, which will always be 1 or 2.
///
/// If two compute sets are used, there will never be any exchange between
/// them so a local tile sync will be performed between them rather than a
/// full IPU sync.
///
/// The reductions are *distributed* between vertices, but not split (except
/// as described above). Before calling this function you should ensure that
/// the reductions are split appropriately so there are enough to distribute.
/// It makes a basic attempt to keep the split roughly even based on an estimate
/// of the number of cycles each reduction will take.
///
/// \param graph  The compute graph.
/// \param css    The compute sets to add the vertices to. This may use one
///               or two compute sets.
/// \param params The reduce operation to perform. Note that in multi-stage
///               operations you only want to do the scale or update in the
///               last stage.
/// \param inputType     The type of the input to this stage of the reduction.
/// \param partialType   The type of any partials created at this stage of the
///                      reduction.
/// \param outputType    The type of the outputs from this stage of the
///                      reduction.
/// \param tile          The tile to map the vertices to.
/// \param reductions    The set of reductions to distribute between vertices.
/// \param dnai          Debug information
///
void connectReductions(poplar::Graph &graph, ComputeSetList &css,
                       ReduceParams params, poplar::Type inputType,
                       poplar::Type partialType, poplar::Type outputType,
                       unsigned tile,
                       const std::vector<RegionReduction> &reductions,
                       bool reductionUsesInput,
                       const poplar::DebugNameAndId &dnai);

/// Find the appropriate vertex specialisation to use
/// \param graph   The compute graph
/// \param params  The reduce operation to solve
/// \param regions The set of reductions to perform
/// \param reductionUsesInput Flag - reduction is input stage or intermediate
using RegionReductionRange =
    boost::iterator_range<std::vector<RegionReduction>::const_iterator>;

ReductionSpecialisation getReductionVertexSpecialisation(
    const poplar::Graph &graph, const ReduceParams &params,
    const RegionReductionRange regions, poplar::Type partialType,
    bool reductionUsesInput);

bool inline reductionSupportsScaling(ReductionSpecialisation specialisation) {
  return specialisation == ReductionSpecialisation::DEFAULT ||
         specialisation == ReductionSpecialisation::SCALAR_OUTPUT_REGIONS ||
         specialisation == ReductionSpecialisation::ALL_REGIONS_CONTINUOUS ||
         specialisation == ReductionSpecialisation::STRIDED_REDUCE;
}
} // namespace popops
#endif // ReductionConnection_hpp
