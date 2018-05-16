#include "Reduction.hpp"

#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

#include <boost/icl/separate_interval_set.hpp>

#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Add.hpp>
#include <popops/Cast.hpp>
#include <popops/Zero.hpp>
#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/print.hpp>

#include "IntermediatePartials.hpp"
#include "IntermediatePartialsUtil.hpp"
#include "ReductionPlan.hpp"
#include "ReductionStages.hpp"

using namespace poplar;

namespace popops {

namespace {

// Reduce a 2D tensor in the first dimension. No other tensor shape
// is supported. The tensor must be at least 1x2.
//
// `out` must be a 1D tensor with the right number of elements. Its tile mapping
// need not necessarily be set.
Tensor reduceFirstDim2D(Graph &graph,
                        const Tensor &A,
                        const Tensor &out,
                        ReduceParams params,
                        program::Sequence &prog,
                        const std::string &debugPrefix,
                        ReductionDebug *debug) {

  // We only accept reductions over 2D tensors.
  if (A.rank() != 2) {
    throw poputil::poplib_error("expected rank 2 but got rank "
                                + std::to_string(A.rank()));
  }
  // Output should be 1D.
  if (out.rank() != 1) {
    throw poputil::poplib_error("expected rank 1 but got rank "
                                + std::to_string(out.rank()));
  }
  // And correct size.
  if (A.dim(1) != out.dim(0)) {
    throw poputil::poplib_error("expected output size "
                                + std::to_string(A.dim(1)) + " but got "
                                + std::to_string(out.dim(0)));
  }

  // Get the tile mapping once at the start of this function because it can be
  // slow and we really don't want to call it more than once.
  auto mapping = graph.getTileMapping(A);

  // Find the output value whose inputs are spread over the most tiles. In other
  // words find the column of A that is mapped to the most tiles.
  auto maxTileSpread = getMaxTileSpread(mapping, A.dim(1));

  // Possible if there are no elements but we shouldn't get to this point
  // if that is true.
  if (maxTileSpread < 1)
    throw poputil::poplib_error("internal error calculating tile spread for "
                                "reduction plan");

  // Visualisation stuff.
  if (debug != nullptr) {
    debug->reductionRatio = A.dim(0);
    debug->outputSize = A.dim(1);
  }

  if (maxTileSpread == 1) {
    // Do the entire reduction on each tile with no exchange at all.
    return inputToOutputNoExchange(graph,
                                   A,
                                   mapping,
                                   out,
                                   params,
                                   prog,
                                   debugPrefix + "/full_on_tile",
                                   debug);

  } else {

    IntermediatePartials ip;

    // Check if we can just convert it without doing anything.
    if (!mappingHasMultipleValuesFromOneColumnOnTheSameTile(mapping,
                                                            A.dim(1))) {
      ip = tensorToIntermediatePartials(A, mapping, debug);
    } else {
      // Reduce as much as possible on each tile and return the intermediate
      // partials. We don't scale or update here.
      ip = inputToIntermediateNoExchange(graph,
                                         A,
                                         mapping,
                                         params.op,
                                         prog,
                                         debugPrefix + "/on_tile",
                                         debug);
      // If it was a SQUARE_ADD, then at this point we have now done the
      // SQUARE - change it to an ADD.
      if (params.op == Operation::SQUARE_ADD)
        params.op = Operation::ADD;
    }

    for (unsigned i = 0;; ++i) {
      // At each point, see if it is worth doing another reduction stage or if
      // we should just do the final reduction, and if so should we do
      // it spread over the IPU or at the destination?
      switch (calculateNextStep(ip)) {
      case INTERMEDIATE_TO_INTERMEDIATE:
        // When splitting up the input we should split it into separate
        // reductions (i.e. split the columns up) as much as possible down to
        // the grain size) and then if necessary split it vertically (chunks
        // of rows) so that we can spread it over the IPU.

        // Don't do the scale or update.
        ip = intermediateToIntermediate(graph,
                                        ip,
                                        params.op,
                                        prog,
                                        debugPrefix + "/stage_"
                                          + std::to_string(i),
                                        debug);
        // If it was a SQUARE_ADD, then at this point we have now done the
        // SQUARE - change it to an ADD.
        if (params.op == Operation::SQUARE_ADD)
          params.op = Operation::ADD;
        break;
      case INTERMEDIATE_TO_OUTPUT:
        return intermediateToOutput(graph,
                                    ip,
                                    out,
                                    params,
                                    prog,
                                    debugPrefix + "/final_stage",
                                    debug);
      }
    }
  }

  POPLIB_UNREACHABLE();
}

} // anonymous namespace

Tensor reduce(Graph &graph,
              const Tensor &A,
              const std::vector<std::size_t> &dims,
              ReduceParams params,
              program::Sequence &prog,
              const std::string &debugPrefix,
              ReductionDebug *debug) {
  return reduce(graph, A, A.elementType(), dims, params, prog,
                debugPrefix, debug);
}

Tensor reduce(Graph &graph,
              const Tensor &A,
              const poplar::Type &outType,
              const std::vector<std::size_t> &dims,
              ReduceParams params,
              program::Sequence &prog,
              const std::string &debugPrefix,
              ReductionDebug *debug) {

  if (params.update)
    throw poputil::poplib_error("Cannot do an update using reduce(); "
                                "call reduceWithOutput() instead.");

  std::set<std::size_t> reducedDims(dims.begin(), dims.end());

  auto shape = A.shape();
  std::vector<std::size_t> reducedShape;
  for (std::size_t d = 0; d < shape.size(); ++d)
    if (reducedDims.count(d) == 0)
      reducedShape.push_back(shape[d]);

  auto out =
      graph.addVariable(outType, reducedShape, debugPrefix + "/output");

  // Deliberately don't set the tile mapping - this will be detected
  // in reduceLastDim2D() and set appropriately.

  return reduceWithOutput(
      graph, A, out, dims, params, prog, debugPrefix, debug);
}

// This wangles the tensors into a 2D matrix so that the reduction only
// has to be done on the first dimension. Then it calls reduceFirstDim2D
// to do the reduction.
Tensor reduceWithOutput(Graph &graph,
                        const Tensor &A,
                        const Tensor &out,
                        const std::vector<std::size_t> &dims,
                        ReduceParams params,
                        program::Sequence &prog,
                        const std::string &debugPrefix,
                        ReductionDebug *debug) {

  if (params.scale != 1.0f &&
      !(params.op == popops::Operation::ADD ||
        params.op == popops::Operation::SQUARE_ADD)) {
    throw poputil::poplib_error("Scale can only be used with ADD or "
                                "SQUARE_ADD");
  }
  if (params.update &&
      !(params.op == popops::Operation::ADD ||
        params.op == popops::Operation::SQUARE_ADD)) {
    throw poputil::poplib_error("Update can only be used with ADD or "
                                "SQUARE_ADD");
  }

  // Convert the dimensions into a unique set.
  std::set<unsigned> reducedDims(dims.begin(), dims.end());

  // Check that all the dimensions are actual dimensions of the input.
  for (auto dim : reducedDims) {
    if (dim >= A.rank())
      throw poputil::poplib_error("Invalid dimension " + std::to_string(dim)
                                  + " for tensor rank "
                                  + std::to_string(A.rank()));
  }

  // Check that the output tensor has the right shape.
  std::vector<std::size_t> reducedShape;
  for (std::size_t d = 0; d < A.rank(); ++d)
    if (reducedDims.count(d) == 0)
      reducedShape.push_back(A.dim(d));

  if (out.shape() != reducedShape) {
    std::stringstream s;
    s << "Dimension mismatch in output. Input shape: ";
    printContainer(A.shape(), s);
    s << " Output shape: ";
    printContainer(out.shape(), s);
    s << " Reduced dimensions: ";
    printContainer(dims, s);

    throw poputil::poplib_error(s.str());
  }

  // If there are no output elements... this is easy!
  if (out.numElements() == 0)
    return out;

  // If the number of input elements is zero we definitely don't need
  // to do a reduction. In this case there are 0 elements in the input,
  // but some elements in the output. This is possible for example when
  // reducing a 10x10x0 tensor in the third dimension to 10x10. It's a bit
  // weird but it makes sense. This is how Tensorflow works.
  if (A.numElements() == 0) {
    // If the output mapping isn't complete, just linearly map it.
    try {
      graph.getTileMapping(out);
    } catch (poplar::invalid_tile_mapping &) {
      poputil::mapTensorLinearly(graph, out);
    }

    // Initialise it to the default value which depends on the operation.

    if (params.op == popops::Operation::ADD ||
        params.op == popops::Operation::SQUARE_ADD ||
        params.op == popops::Operation::LOGICAL_OR) {
      popops::zero(graph, out, prog, debugPrefix + "/add_init");
    } else {
      double initVal = 0.0;
      switch (params.op) {
      case popops::Operation::MUL:
        initVal = params.scale;
        break;
      case popops::Operation::MIN:
        initVal = std::numeric_limits<double>::infinity();
        break;
      case popops::Operation::MAX:
        initVal = -std::numeric_limits<double>::infinity();
        break;
      case popops::Operation::LOGICAL_AND:
        initVal = 1.0;
        break;
      default:
        throw poputil::poplib_error("Internal error, unhandled reduction type: "
                                    + std::to_string(
                                      static_cast<int>(params.op)));
      }

      Tensor initialiser = graph.addConstant(out.elementType(), out.shape(),
                                             initVal);
      prog.add(program::Copy(initialiser, out));
    }

    return out;
  }

  // If the input only reduces dimensions of size 1, we don't need to
  // do a reduction - just copy the input tensor to the output. A copy is
  // returned (using cast) rather than just returning the original because
  // then the user can be sure that the returned tensor refers to a distinct
  // variable and changing the tile mapping won't affect anything else.
  bool reductionRequired = false;

  for (auto dim : reducedDims) {
    if (A.dim(dim) >= 2) {
      reductionRequired = true;
      break;
    }
  }

  if (!reductionRequired) {
    // If the graph mapping isn't complete, set it to the same as the input.
    try {
      graph.getTileMapping(out);
    } catch (poplar::invalid_tile_mapping &) {
      graph.setTileMapping(out, graph.getTileMapping(A));
    }

    // If it is a scale or update we still need to do that.
    if (params.update || params.scale != 1.0f) {
      // TODO: This is a bit silly but I can't see a good way to do it without
      // a popops::mul(Tensor A, float k).
      if (!params.update) {
        popops::zero(graph, out, prog, debugPrefix + "/zero_hack");
      }

      // If A isn't the same type as out, cast it.
      poplar::Tensor Acast = A;
      if (A.elementType() != out.elementType()) {
        Acast = popops::cast(
            graph, A, out.elementType(), prog, debugPrefix + "/cast");
      }

      popops::addTo(
          graph, out, Acast, params.scale, prog, debugPrefix + "/update");

    } else {
      // Cast is used here rather than copy because the type could be different
      // if AND or OR are used. If the type is the same cast() will
      // automatically switch to copy.
      auto castProg = popops::cast(graph, A, out, debugPrefix + "/cast");
      prog.add(castProg);
    }

    return out;
  }

  // At this point there is definitely some reducing to do. The tensor
  // dimensions are partitioned into 'reduced' and 'other' sets. The tensor
  // is dimshuffled so that the reduced dimensions are at the front, and then
  // it is flattened to 2D - one dimension for reducedDims, and one for
  // the otherDims.

  // The set of dimensions that aren't reduced.
  std::set<unsigned> otherDims;
  for (unsigned i = 0; i < A.rank(); ++i) {
    if (reducedDims.count(i) == 0) {
      otherDims.insert(i);
    }
  }

  // Flatten it to 2D.
  auto mangledTensor = mangleTo2D(A, reducedDims);

  // Do the 2D->1D reduction.
  Tensor R = reduceFirstDim2D(graph,
                              mangledTensor.tensor,
                              out.flatten(),
                              params,
                              prog,
                              debugPrefix,
                              debug);

  // Now reshape it back to its proper shape.
  return R.reshape(mangledTensor.inflatedShape);
}

MangledTensor mangleTo2D(const Tensor &A, std::set<unsigned> &reducedDims) {

  // The set of dimensions that aren't reduced.
  std::set<unsigned> otherDims;
  for (unsigned i = 0; i < A.rank(); ++i) {
    if (reducedDims.count(i) == 0) {
      otherDims.insert(i);
    }
  }

  // How to permute the dimensions to get the reduced dimensions to the back.
  std::vector<unsigned> permutation;

  // Add the reduced dimensions, then the non-reduced ones.
  permutation.insert(permutation.end(), reducedDims.begin(), reducedDims.end());
  permutation.insert(permutation.end(), otherDims.begin(), otherDims.end());

  // Calculate the number of input elements to each output element.
  std::size_t reductionFactor = 1;
  for (auto i : reducedDims)
    reductionFactor *= A.dim(i);

  // And the number of parallel reductions.
  std::size_t numReductions = A.numElements() / reductionFactor;

  // The dimension of the flattened tensor.
  std::vector<std::size_t> flattenedShape = {reductionFactor, numReductions};

  // Now permute and flatten the tensor to 2D.
  MangledTensor m;
  m.tensor = A.dimShuffle(permutation).reshape(flattenedShape);
  std::vector<std::size_t> outputShape;
  for (auto dim : otherDims)
    m.inflatedShape.push_back(A.dim(dim));

  return m;
}

}
