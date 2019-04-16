#include "Reduction.hpp"

#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

#include <boost/optional.hpp>

#include <boost/icl/separate_interval_set.hpp>
#include <boost/variant.hpp>

#include <poputil/Broadcast.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Cast.hpp>
#include <popops/Zero.hpp>
#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/print.hpp>
#include "poplibs_support/OptionParsing.hpp"

#include "IntermediatePartials.hpp"
#include "IntermediatePartialsUtil.hpp"
#include "ReductionPlan.hpp"
#include "ReductionStages.hpp"

using namespace poplar;

namespace popops {

namespace {


// Structure to describe the accumulation types at different levels of the
// reduction.
struct ReductionTypes {
  // The type for temporary stack values used for accumulation within a vertex.
  //
  // If we do multi-stage reductions on a tile, this type is also used for the
  // intermediate values.
  //
  // If this type is not the output type of the reduction then the vertices
  // in the final stage of the reduction may output the accumulator type and
  // then a separate cast operation will convert the output to the final
  // output type.
  Type inVertex;
  // The type for values sent between tiles on the exchange.
  Type interTile;
};



// Reduce a 2D tensor in the first dimension. No other tensor shape
// is supported. The tensor must be at least 1x2.
//
// `out` must be a 1D tensor with the right number of elements. Its tile mapping
// need not necessarily be set.
void reduceFirstDim2D(Graph &graph,
                      const Tensor &in,
                      const Tensor &out,
                      ReduceParams params,
                      const ReductionTypes &reductionTypes,
                      std::vector<ComputeSet> &css,
                      const std::string &debugPrefix,
                      ReductionDebug *debug) {

  // We only accept reductions over 2D tensors.
  if (in.rank() != 2) {
    throw poputil::poplibs_error("expected rank 2 but got rank "
                                + std::to_string(in.rank()));
  }
  // Output should be 1D.
  if (out.rank() != 1) {
    throw poputil::poplibs_error("expected rank 1 but got rank "
                                + std::to_string(out.rank()));
  }
  // And correct size.
  if (in.dim(1) != out.dim(0)) {
    throw poputil::poplibs_error("expected output size "
                                + std::to_string(in.dim(1)) + " but got "
                                + std::to_string(out.dim(0)));
  }

  // Get the tile mapping once at the start of this function because it can be
  // slow and we really don't want to call it more than once.
  auto mapping = graph.getTileMapping(in);

  // Find the output value whose inputs are spread over the most tiles. In other
  // words find the column of A that is mapped to the most tiles.
  auto maxTileSpread = getMaxTileSpread(mapping, in.dim(1));

  // Possible if there are no elements but we shouldn't get to this point
  // if that is true.
  if (maxTileSpread < 1)
    throw poputil::poplibs_error("internal error calculating tile spread for "
                                "reduction plan");

  // Visualisation stuff.
  if (debug != nullptr) {
    debug->reductionRatio = in.dim(0);
    debug->outputSize = in.dim(1);
  }
  ComputeSetList csList(css);

  if (maxTileSpread == 1) {
    // Do the entire reduction on each tile with no exchange at all.
    inputToOutputNoExchange(graph,
                            in,
                            mapping,
                            out,
                            reductionTypes.inVertex,
                            params,
                            csList,
                            debugPrefix + "/ReduceOnTile",
                            debug);

  } else {

    IntermediatePartials ip;

    // Check if we can just convert it without doing anything.
    if (!mappingHasMultipleValuesFromOneColumnOnTheSameTile(mapping,
                                                            in.dim(1))) {
      ip = tensorToIntermediatePartials(in, mapping, debug);
    } else {
      // Reduce as much as possible on each tile and return the intermediate
      // partials. We don't scale or update here.
      ip = inputToIntermediateNoExchange(graph,
                                         in,
                                         mapping,
                                         params.op,
                                         reductionTypes.inVertex,
                                         reductionTypes.interTile,
                                         csList,
                                         debugPrefix + "/ReduceOnTile",
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
                                        reductionTypes.inVertex,
                                        reductionTypes.interTile,
                                        csList,
                                        debugPrefix + "/ReduceStage"
                                          + std::to_string(i),
                                        debug);
        // If it was a SQUARE_ADD, then at this point we have now done the
        // SQUARE - change it to an ADD.
        if (params.op == Operation::SQUARE_ADD)
          params.op = Operation::ADD;
        break;
      case INTERMEDIATE_TO_OUTPUT:
        intermediateToOutput(graph,
                             ip,
                             out,
                             params,
                             reductionTypes.inVertex,
                             csList,
                             debugPrefix + "/ReduceFinalStage",
                             debug);
        return;
      }
    }
  }
}

Tensor makeOutputTensor(Graph &graph,
                        const Tensor &in,
                        const poplar::Type &outType,
                        const std::vector<std::size_t> &dims,
                        const std::string &debugPrefix) {
  std::set<std::size_t> reducedDims(dims.begin(), dims.end());

  auto shape = in.shape();
  std::vector<std::size_t> reducedShape;
  for (std::size_t d = 0; d < shape.size(); ++d)
    if (reducedDims.count(d) == 0)
      reducedShape.push_back(shape[d]);

  return graph.addVariable(outType, reducedShape,
                           debugPrefix + "/ReduceOutput");

  // Deliberately don't set the tile mapping - this will be detected
  // in reduceLastDim2D() and set appropriately.
}

bool opBenefitsFromHigherIntermediatePrecision(const popops::Operation &op) {
  switch (op) {
  case popops::Operation::ADD:
  case popops::Operation::SQUARE_ADD:
  case popops::Operation::MUL:
    return true;
  default:
    return false;
  }
  POPLIB_UNREACHABLE();
}

std::map<std::string, poplar::Type> accumTypeMap {
  { "half", poplar::HALF },
  { "float", poplar::FLOAT }
};

// This wangles the tensors into a 2D matrix so that the reduction only
// has to be done on the first dimension. Then it calls reduceFirstDim2D
// to do the reduction. It accepts either a vector<ComputeSet>& or a Sequence&
// because it can be a bit faster in the latter case for reductions that
// don't actually do any reducing.
void reduceWithOutputProgOrCss(Graph &graph,
                               const Tensor &in,
                               const Tensor &out,
                               const std::vector<std::size_t> &dims,
                               ReduceParams params,
                               boost::variant<std::vector<ComputeSet>&,
                                              program::Sequence&> progOrCss,
                               const std::string &debugPrefix,
                               const poplar::OptionFlags &options,
                               ReductionDebug *debug) {
  bool isProg = progOrCss.which() == 1;

  // Decide the reduction types for each stage.
  ReductionTypes reductionTypes;
  auto useFloatAccum = (out.elementType() == poplar::HALF &&
                        opBenefitsFromHigherIntermediatePrecision(params.op));
  auto accumType = useFloatAccum ? poplar::FLOAT : out.elementType();
  reductionTypes.interTile = accumType;
  reductionTypes.inVertex = accumType;

  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec reductionSpec{
    { "accumType.interTile", OptionHandler::createWithEnum(
      reductionTypes.interTile, accumTypeMap) },
    { "accumType.inVertex", OptionHandler::createWithEnum(
      reductionTypes.inVertex, accumTypeMap) }
  };

  for (const auto &entry : options) {
    reductionSpec.parse(entry.first, entry.second);
  }

 if (params.scale &&
     !(params.op == popops::Operation::ADD ||
       params.op == popops::Operation::SQUARE_ADD)) {
   throw poputil::poplibs_error("Scale can only be used with ADD or "
                               "SQUARE_ADD");
 }
 if (params.update &&
     !(params.op == popops::Operation::ADD ||
       params.op == popops::Operation::SQUARE_ADD)) {
   throw poputil::poplibs_error("Update can only be used with ADD or "
                               "SQUARE_ADD");
 }

 // Convert the dimensions into a unique set.
 std::set<unsigned> reducedDims(dims.begin(), dims.end());

 // Check that all the dimensions are actual dimensions of the input.
 for (auto dim : reducedDims) {
   if (dim >= in.rank())
     throw poputil::poplibs_error("Invalid dimension " + std::to_string(dim)
                                 + " for tensor rank "
                                 + std::to_string(in.rank()));
 }

 // Check that the output tensor has the right shape.
 std::vector<std::size_t> reducedShape;
 for (std::size_t d = 0; d < in.rank(); ++d)
   if (reducedDims.count(d) == 0)
     reducedShape.push_back(in.dim(d));

 if (out.shape() != reducedShape) {
   std::stringstream s;
   s << "Dimension mismatch in output. Input shape: ";
   printContainer(in.shape(), s);
   s << " Output shape: ";
   printContainer(out.shape(), s);
   s << " Reduced dimensions: ";
   printContainer(dims, s);

   throw poputil::poplibs_error(s.str());
 }

 // If there are no output elements... this is easy!
 if (out.numElements() == 0)
   return;

 // If the number of input elements is zero we definitely don't need
 // to do a reduction. In this case there are 0 elements in the input,
 // but some elements in the output. This is possible for example when
 // reducing a 10x10x0 tensor in the third dimension to 10x10. It's a bit
 // weird but it makes sense. This is how Tensorflow works.
 if (in.numElements() == 0) {
   if (params.update) {
     // If it's an update and there are no inputs the output won't change.
     return;
   }

   // TODO: Need a way of initialising a tensor with a value using only a
   // compute set. This is a pretty simple codelet to add.
   if (!isProg) {
     throw poputil::poplibs_error("The popops::Reduce() vector<ComputeSet> API "
                                 "cannot reduce empty inputs yet.");
   }

   auto &prog = boost::get<program::Sequence&>(progOrCss);

   // If the output mapping isn't complete, just linearly map it.
   bool mappingComplete;
   graph.getTileMapping(out, &mappingComplete);
   if (!mappingComplete) {
       poputil::mapTensorLinearly(graph, out);
   }

   // Initialise it to the default value which depends on the operation.
   if (params.op == popops::Operation::ADD ||
       params.op == popops::Operation::SQUARE_ADD ||
       params.op == popops::Operation::LOGICAL_OR) {
     popops::zero(graph, out, prog, debugPrefix + "/ReduceAddInit");
   } else {
     double initVal = 0.0;
     switch (params.op) {
     case popops::Operation::MUL:
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
       throw poputil::poplibs_error("Internal error, unhandled reduction type: "
                                   + std::to_string(
                                     static_cast<int>(params.op)));
     }
     Tensor initialiser;
     if(params.op != popops::Operation::MUL) {
        initialiser = graph.addConstant(out.elementType(),
                                        out.shape(),
                                        initVal,
                                        debugPrefix + "/initialiser");
        graph.setTileMapping(initialiser, 0);
        prog.add(program::Copy(initialiser, out));
     }
     else {
      auto paramsBroadcast = *params.scale;
      Tensor outCopy = out;
      poputil::broadcastToMatch(outCopy, paramsBroadcast);
      prog.add(program::Copy(paramsBroadcast, out));
     }
   }

   return;
 }

 // If the input only reduces dimensions of size 1, we don't need to
 // do a reduction - just copy the input tensor to the output. A copy is
 // returned (using cast) rather than just returning the original because
 // then the user can be sure that the returned tensor refers to a distinct
 // variable and changing the tile mapping won't affect anything else.
 bool reductionRequired = false;

 for (auto dim : reducedDims) {
   if (in.dim(dim) >= 2) {
     reductionRequired = true;
     break;
   }
 }

 if (!reductionRequired && isProg) {
   auto &prog = boost::get<program::Sequence&>(progOrCss);

   // If the graph mapping isn't complete, set it to the same as the input.
   bool mappingComplete;
   graph.getTileMapping(out, &mappingComplete);
   if (!mappingComplete) {
     graph.setTileMapping(out, graph.getTileMapping(in));
   }

   // If it is a scale or update, or SQUARE_ADD we still need to do that.
   if (params.update || params.scale ||
       params.op == Operation::SQUARE_ADD) {

     // If in isn't the same type as out, cast it first.
     poplar::Tensor inCast = in;
     if (in.elementType() != out.elementType()) {
       inCast = cast(
           graph, in, out.elementType(), prog, debugPrefix + "/ReduceCast");
     }

     poplar::Tensor scaleCast = *params.scale;
     if (out.elementType() != params.scale->elementType()) {
       scaleCast = cast(
           graph, *params.scale, out.elementType(), prog,
                                            debugPrefix + "/ReduceScaleCast");

     }
     // Calculate the necessary expression. E.g. the most complex case,
     //
     //   x += f * v^2
     //
     // is
     //
     //   Add(_1, Mul(Square(_2), params.scale))
     //
     // And the simplest, x = f * v, is
     //
     //   Mul(_2, params.scale)

     using namespace popops::expr;

     // TODO: This is a bit ugly; would be nice if Expr's were copyable.
     auto expr = std::unique_ptr<Expr>(new PlaceHolder(2));
     if (params.op == Operation::SQUARE_ADD)
       expr.reset(new Square(*expr));
     if (params.scale)
       expr.reset(new Mul(*expr, _3));
     if (params.update)
       expr.reset(new Add(*expr, _1));

     mapInPlace(graph, *expr, {out.flatten(), inCast.flatten(), scaleCast},
                prog, debugPrefix + "/ReduceExpression");

   } else {
     // Cast is used here rather than copy because the type could be different
     // if ADD or MUL are used. If the type is the same cast() will
     // automatically switch to copy.
     auto castProg = cast(graph, in, out, debugPrefix + "/ReduceCast");
     prog.add(castProg);
   }

   return;
 }

 // At this point there is definitely some reducing to do. The tensor
 // dimensions are partitioned into 'reduced' and 'other' sets. The tensor
 // is dimshuffled so that the reduced dimensions are at the front, and then
 // it is flattened to 2D - one dimension for reducedDims, and one for
 // the otherDims.
 auto input2D = mangleTo2D(in, reducedDims);

 // Do the 2D->1D reduction.
 if (isProg) {
   std::vector<ComputeSet> css;
   reduceFirstDim2D(graph,
                    input2D,
                    out.flatten(),
                    params,
                    reductionTypes,
                    css,
                    debugPrefix,
                    debug);
   auto &prog = boost::get<program::Sequence&>(progOrCss);
   for (const auto &cs : css) {
     prog.add(program::Execute(cs));
   }
 } else {
   reduceFirstDim2D(graph,
                    input2D,
                    out.flatten(),
                    params,
                    reductionTypes,
                    boost::get<std::vector<ComputeSet>&>(progOrCss),
                    debugPrefix,
                    debug);
 }
}

} // end anonymous namespace


void reduceWithOutput(Graph &graph,
                      const Tensor &in,
                      const Tensor &out,
                      const std::vector<std::size_t> &dims,
                      ReduceParams params,
                      std::vector<ComputeSet> &css,
                      const std::string &debugPrefix,
                      const poplar::OptionFlags &options,
                      ReductionDebug *debug) {
  reduceWithOutputProgOrCss(graph, in, out, dims, params, css,
                            debugPrefix, options, debug);
}

void reduceWithOutput(Graph &graph,
                      const Tensor &in,
                      const Tensor &out,
                      const std::vector<std::size_t> &dims,
                      ReduceParams params,
                      program::Sequence &prog,
                      const std::string &debugPrefix,
                      const poplar::OptionFlags &options,
                      ReductionDebug *debug) {
  reduceWithOutputProgOrCss(graph, in, out, dims, params, prog,
                            debugPrefix, options, debug);
}

Tensor reduce(Graph &graph,
              const Tensor &in,
              const std::vector<std::size_t> &dims,
              ReduceParams params,
              program::Sequence &prog,
              const std::string &debugPrefix,
              const poplar::OptionFlags &options,
              ReductionDebug *debug) {
  return reduce(graph, in, in.elementType(), dims, params, prog,
                debugPrefix, options, debug);
}

Tensor reduce(Graph &graph,
              const Tensor &in,
              const std::vector<std::size_t> &dims,
              ReduceParams params,
              std::vector<ComputeSet> &css,
              const std::string &debugPrefix,
              const poplar::OptionFlags &options,
              ReductionDebug *debug) {
  return reduce(graph, in, in.elementType(), dims, params, css,
                debugPrefix, options, debug);
}

Tensor reduce(Graph &graph,
              const Tensor &in,
              const poplar::Type &outType,
              const std::vector<std::size_t> &dims,
              ReduceParams params,
              std::vector<ComputeSet> &css,
              const std::string &debugPrefix,
              const poplar::OptionFlags &options,
              ReductionDebug *debug) {
  if (params.update)
    throw poputil::poplibs_error("Cannot do an update using reduce(); "
                                "call reduceWithOutput() instead.");

  auto out = makeOutputTensor(graph, in, outType, dims, debugPrefix);

  reduceWithOutput(
      graph, in, out, dims, params, css, debugPrefix, options, debug);

  return out;
}

Tensor reduce(Graph &graph,
              const Tensor &in,
              const poplar::Type &outType,
              const std::vector<std::size_t> &dims,
              ReduceParams params,
              program::Sequence &prog,
              const std::string &debugPrefix,
              const poplar::OptionFlags &options,
              ReductionDebug *debug) {
  if (params.update)
    throw poputil::poplibs_error("Cannot do an update using reduce(); "
                                "call reduceWithOutput() instead.");

  auto out = makeOutputTensor(graph, in, outType, dims, debugPrefix);

  reduceWithOutput(
      graph, in, out, dims, params, prog, debugPrefix, options, debug);

  return out;
}

Tensor mangleTo2D(const Tensor &A, std::set<unsigned> &reducedDims) {

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

  return A.dimShuffle(permutation).reshape(flattenedShape);
}

}
