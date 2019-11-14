#include "Reduction.hpp"

#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

#include <boost/icl/separate_interval_set.hpp>
#include <boost/variant.hpp>

#include "poplibs_support/Algorithms.hpp"
#include "poplibs_support/OptionParsing.hpp"
#include "poplibs_support/logging.hpp"
#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/print.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Zero.hpp>
#include <poputil/Broadcast.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include "IntermediatePartials.hpp"
#include "IntermediatePartialsUtil.hpp"
#include "ReductionPlan.hpp"
#include "ReductionStages.hpp"

using namespace poplar;
using namespace poplibs;
using namespace poplibs_support;

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
// `reductionResultTensors` is a vector into which this function will push any
// tensor that is written to with a reduction result. These can be written to
// in 2 separate compute sets. If so, they will require a WriteUndef to be
// added to the program before the compute sets in 'css' to prevent them from
// becoming always live.
void reduceFirstDim2D(Graph &graph, const Tensor &in,
                      boost::optional<Tensor> &out,
                      const std::vector<std::size_t> outputShape,
                      Type outputType, ReduceParams params,
                      const ReductionTypes &reductionTypes,
                      std::vector<ComputeSet> &css,
                      std::vector<Tensor> &reductionResultTensors,
                      const std::string &debugPrefix, ReductionDebug *debug) {
  logging::info("Reducing first dimension");
  // We only accept reductions over 2D tensors.
  if (in.rank() != 2) {
    throw poputil::poplibs_error("expected rank 2 but got rank " +
                                 std::to_string(in.rank()));
  }
  if (out) {
    // Output should be the correct size. It will be flattened when used later
    // so the shape isn't important
    if (in.dim(1) != out.get().flatten().dim(0)) {
      throw poputil::poplibs_error("expected output size " +
                                   std::to_string(in.dim(1)) + " but got " +
                                   std::to_string(out.get().flatten().dim(0)));
    }
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

  logging::info("Num elements to reduce {} -> {}", in.numElements(), in.dim(1));

  if (maxTileSpread == 1) {
    logging::debug("Reduction is completely tile local");
    // Do the entire reduction on each tile with no exchange at all.
    inputToOutputNoExchange(graph, in, mapping, out, outputShape, outputType,
                            reductionTypes.inVertex, params, csList,
                            reductionResultTensors,
                            debugPrefix + "/ReduceOnTile", debug);
    return;
  } else {

    IntermediatePartials ip;

    // Check if we can just convert it without doing anything.
    if (!mappingHasMultipleValuesFromOneColumnOnTheSameTile(mapping,
                                                            in.dim(1))) {
      ip = tensorToIntermediatePartials(in, mapping, debug);
    } else {
      // Reduce as much as possible on each tile and return the intermediate
      // partials. We don't scale or update here.
      logging::debug("Reduce locally with no exchange");
      ip = inputToIntermediateNoExchange(
          graph, in, mapping, params.op, reductionTypes.inVertex,
          reductionTypes.interTile, csList, reductionResultTensors,
          debugPrefix + "/ReduceOnTile", debug);
      // If it was a SQUARE_ADD, then at this point we have now done the
      // SQUARE - change it to an ADD.
      if (params.op == Operation::SQUARE_ADD)
        params.op = Operation::ADD;
    }

    for (unsigned i = 0;; ++i) {
      // At each point, see if it is worth doing another reduction stage or if
      // we should just do the final reduction, and if so should we do
      // it spread over the IPU or at the destination?
      switch (calculateNextStep(graph.getTarget(), ip)) {
      case INTERMEDIATE_TO_INTERMEDIATE:
        logging::debug("Introducing new intermediate to intermediate "
                       "reduction stage");
        // When splitting up the input we should split it into separate
        // reductions (i.e. split the columns up) as much as possible down to
        // the grain size) and then if necessary split it vertically (chunks
        // of rows) so that we can spread it over the IPU.

        // Don't do the scale or update.
        ip = intermediateToIntermediate(
            graph, ip, params.op, reductionTypes.interTile, csList,
            reductionResultTensors,
            debugPrefix + "/ReduceStage" + std::to_string(i), debug);
        // If it was a SQUARE_ADD, then at this point we have now done the
        // SQUARE - change it to an ADD.
        if (params.op == Operation::SQUARE_ADD)
          params.op = Operation::ADD;
        break;
      case INTERMEDIATE_TO_OUTPUT:

        logging::info("Creating final reduction stage");
        intermediateToOutput(graph, ip, out, outputShape, outputType, params,
                             reductionTypes.inVertex, csList,
                             reductionResultTensors, in,
                             debugPrefix + "/ReduceFinalStage", debug);
        return;
      }
    }
  }
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

std::map<std::string, poplar::Type> accumTypeMap{{"half", poplar::HALF},
                                                 {"float", poplar::FLOAT}};

// This wangles the tensors into a 2D matrix so that the reduction only
// has to be done on the first dimension. Then it calls reduceFirstDim2D
// to do the reduction. It accepts either a vector<ComputeSet>& or a Sequence&
// because it can be a bit faster in the latter case for reductions that
// don't actually do any reducing.
void reduceWithOutputProgOrCss(
    Graph &graph, const Tensor &in, boost::optional<Tensor> &out,
    const poplar::Type &outputType, const std::vector<std::size_t> &dims,
    ReduceParams params,
    boost::variant<std::vector<ComputeSet> &, program::Sequence &> progOrCss,
    const std::string &debugPrefix, const poplar::OptionFlags &options,
    ReductionDebug *debug) {

  const auto getShape = [](const Tensor &t) {
    std::stringstream ss;
    printContainer(t.shape(), ss);
    return ss.str();
  };
  logging::info("reduce in={}, out={}, dims={}, name={}", in.shape(),
                out.map(getShape), dims, debugPrefix);
  logging::debug("Reduce begin DebugStr: {}", debugPrefix);
  bool isProg = progOrCss.which() == 1;

  // Decide the reduction types for each stage.
  ReductionTypes reductionTypes;
  auto useFloatAccum = (outputType == poplar::HALF &&
                        opBenefitsFromHigherIntermediatePrecision(params.op));
  auto accumType = useFloatAccum ? poplar::FLOAT : outputType;
  reductionTypes.interTile = accumType;
  reductionTypes.inVertex = accumType;

  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec reductionSpec{
      {"accumType.interTile",
       OptionHandler::createWithEnum(reductionTypes.interTile, accumTypeMap)},
      {"accumType.inVertex",
       OptionHandler::createWithEnum(reductionTypes.inVertex, accumTypeMap)}};

  for (const auto &entry : options) {
    reductionSpec.parse(entry.first, entry.second);
  }

  if (params.useScale && !(params.op == popops::Operation::ADD ||
                           params.op == popops::Operation::SQUARE_ADD)) {
    throw poputil::poplibs_error("Scale can only be used with ADD or "
                                 "SQUARE_ADD");
  }
  if (params.useScale) {
    if (params.scale.elementType() != FLOAT) {
      throw poputil::poplibs_error("Scale must be of type poplar::FLOAT");
    }
  }
  if (params.update && !(params.op == popops::Operation::ADD ||
                         params.op == popops::Operation::SQUARE_ADD)) {
    throw poputil::poplibs_error("Update can only be used with ADD or "
                                 "SQUARE_ADD");
  }

  // Convert the dimensions into a unique set.
  std::set<unsigned> reducedDims(dims.begin(), dims.end());

  // Check that all the dimensions are actual dimensions of the input.
  for (auto dim : reducedDims) {
    if (dim >= in.rank())
      throw poputil::poplibs_error("Invalid dimension " + std::to_string(dim) +
                                   " for tensor rank " +
                                   std::to_string(in.rank()));
  }

  std::vector<std::size_t> outputShape;
  for (std::size_t d = 0; d < in.rank(); ++d) {
    if (reducedDims.count(d) == 0) {
      outputShape.push_back(in.dim(d));
    }
  }
  if (out) {
    // If we have one, check that the output tensor has the right shape.
    if (out.get().shape() != outputShape) {
      std::stringstream s;
      s << "Dimension mismatch in output. Input shape: ";
      printContainer(in.shape(), s);
      s << " Output shape: ";
      printContainer(out.get().shape(), s);
      s << " Reduced dimensions: ";
      printContainer(dims, s);

      throw poputil::poplibs_error(s.str());
    }
  }
  const auto numOutputElements = std::accumulate(
      outputShape.begin(), outputShape.end(), 1U, std::multiplies<>());

  // If there are no output elements... this is easy!
  // But we still need to produce an output Tensor if there isn't one.
  if (numOutputElements == 0) {
    logging::info("Empty output tensor");
    if (params.update) {
      if (!out) {
        out = graph.addVariable(outputType, {0});
      }
    }
  }

  // If the number of input elements is zero we definitely don't need
  // to do a reduction. In this case there are 0 elements in the input,
  // but some elements in the output. This is possible for example when
  // reducing a 10x10x0 tensor in the third dimension to 10x10. It's a bit
  // weird but it makes sense. This is how Tensorflow works.
  if (in.numElements() == 0) {

    logging::info("zero input elements to reduction");
    // If it's an update and there are no inputs the output won't change.

    // TODO: T12956 Add support to initialise a tensor with a value using only
    // a compute set.
    if (!isProg) {
      throw poputil::poplibs_error("The popops::Reduce() vector<ComputeSet> API"
                                   " cannot reduce empty inputs yet.");
    }

    auto &prog = boost::get<program::Sequence &>(progOrCss);
    if (out) {
      // If the output mapping isn't complete, just linearly map it.
      bool mappingComplete;
      graph.getTileMapping(out.get(), &mappingComplete);
      if (!mappingComplete) {
        poputil::mapTensorLinearly(graph, out.get());
      }
    } else {
      out = graph.addVariable(outputType, outputShape);
      poputil::mapTensorLinearly(graph, out.get());
    }
    // Initialise it to the default value which depends on the operation.
    if (params.op == popops::Operation::ADD ||
        params.op == popops::Operation::SQUARE_ADD ||
        params.op == popops::Operation::LOGICAL_OR) {
      popops::zero(graph, out.get(), prog, debugPrefix + "/ReduceAddInit");
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
        throw poputil::poplibs_error(
            "Internal error, unhandled reduction type:" +
            std::to_string(static_cast<int>(params.op)));
      }
      Tensor initialiser;
      if (params.op != popops::Operation::MUL) {
        initialiser = graph.addConstant(outputType, out.get().shape(), initVal,
                                        debugPrefix + "/initialiser");
        graph.setTileMapping(initialiser, 0);
        prog.add(program::Copy(initialiser, out.get()));
      } else {
        auto paramsBroadcast = params.scale;
        Tensor outCopy = out.get();
        poputil::broadcastToMatch(outCopy, paramsBroadcast);
        prog.add(program::Copy(paramsBroadcast, out.get()));
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
    logging::debug("No reduction required");
    auto &prog = boost::get<program::Sequence &>(progOrCss);

    if (out) {
      // If the graph mapping isn't complete, set it to the same as the input.
      bool mappingComplete;
      graph.getTileMapping(out.get(), &mappingComplete);
      if (!mappingComplete) {
        graph.setTileMapping(out.get(), graph.getTileMapping(in));
      }
    } else {
      // Create the output Tensor
      out = graph.clone(outputType, in, debugPrefix).reshape(outputShape);
    }

    // If it is a scale or update, or SQUARE_ADD we still need to do that.
    if (params.update || params.useScale ||
        params.op == Operation::SQUARE_ADD) {

      // If in isn't the same type as out, cast it first.
      poplar::Tensor inCast = in;
      if (in.elementType() != outputType) {
        inCast = cast(graph, in, outputType, prog, debugPrefix + "/ReduceCast");
      }

      poplar::Tensor scaleCast = params.scale;
      if (outputType != params.scale.elementType()) {
        scaleCast = cast(graph, params.scale, outputType, prog,
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
      Any expr = _2;
      if (params.op == Operation::SQUARE_ADD) {
        expr = Square(expr);
      }
      if (params.useScale) {
        expr = Mul(expr, _3);
      }
      if (params.update) {
        expr = Add(expr, _1);
      }

      mapInPlace(graph, expr,
                 {out.get().flatten(), inCast.flatten(), scaleCast}, prog,
                 debugPrefix + "/ReduceExpression");

    } else {
      // Cast is used here rather than copy because the type could be different
      // if ADD or MUL are used. If the type is the same cast() will
      // automatically switch to copy.
      auto castProg = cast(graph, in.flatten(), out.get().flatten(),
                           debugPrefix + "/ReduceCast");
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
  logging::debug("Get 2D view of tensor for reduction: {}, {}", input2D.dim(0),
                 input2D.dim(1));

  // Do the 2D->1D reduction.
  std::vector<Tensor> reductionResultTensors;
  if (isProg) {
    std::vector<ComputeSet> css;

    reduceFirstDim2D(graph, input2D, out, outputShape, outputType, params,
                     reductionTypes, css, reductionResultTensors, debugPrefix,
                     debug);
    auto &prog = boost::get<program::Sequence &>(progOrCss);
    // First mark with 'WriteUndef' any tensor that will be completely written
    // by this whole reduction, but may be written internally in two different
    // compute sets. This causes the tensor to unnecessarily become always live,
    // which is rectified by using WriteUndef.
    // The tensors are all concatenated together before being passed to
    // WriteUndef for efficiency.
    if (reductionResultTensors.size() > 0) {
      prog.add(program::WriteUndef(concat(reductionResultTensors)));
    }
    for (const auto &cs : css) {
      prog.add(program::Execute(cs));
    }

  } else {
    // For this variant we ignore the list of Tensors to be 'WriteUndef'd.
    reduceFirstDim2D(graph, input2D, out, outputShape, outputType, params,
                     reductionTypes,
                     boost::get<std::vector<ComputeSet> &>(progOrCss),
                     reductionResultTensors, debugPrefix, debug);
  }
}
} // end anonymous namespace

void reduceWithOutput(Graph &graph, const Tensor &in, const Tensor &out_,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      std::vector<ComputeSet> &css,
                      const std::string &debugPrefix,
                      const poplar::OptionFlags &options,
                      ReductionDebug *debug) {
  boost::optional<Tensor> out = out_;
  reduceWithOutputProgOrCss(graph, in, out, out_.elementType(), dims, params,
                            css, debugPrefix, options, debug);
}

void reduceWithOutput(Graph &graph, const Tensor &in, const Tensor &out_,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      program::Sequence &prog, const std::string &debugPrefix,
                      const poplar::OptionFlags &options,
                      ReductionDebug *debug) {
  boost::optional<Tensor> out = out_;
  reduceWithOutputProgOrCss(graph, in, out, out_.elementType(), dims, params,
                            prog, debugPrefix, options, debug);
}

Tensor reduce(Graph &graph, const Tensor &in,
              const std::vector<std::size_t> &dims, ReduceParams params,
              program::Sequence &prog, const std::string &debugPrefix,
              const poplar::OptionFlags &options, ReductionDebug *debug) {
  return reduce(graph, in, in.elementType(), dims, params, prog, debugPrefix,
                options, debug);
}

Tensor reduce(Graph &graph, const Tensor &in,
              const std::vector<std::size_t> &dims, ReduceParams params,
              std::vector<ComputeSet> &css, const std::string &debugPrefix,
              const poplar::OptionFlags &options, ReductionDebug *debug) {
  return reduce(graph, in, in.elementType(), dims, params, css, debugPrefix,
                options, debug);
}

Tensor reduce(Graph &graph, const Tensor &in, const poplar::Type &outType,
              const std::vector<std::size_t> &dims, ReduceParams params,
              std::vector<ComputeSet> &css, const std::string &debugPrefix,
              const poplar::OptionFlags &options, ReductionDebug *debug) {
  if (params.update)
    throw poputil::poplibs_error("Cannot do an update using reduce(); "
                                 "call reduceWithOutput() instead.");
  boost::optional<Tensor> out;
  reduceWithOutputProgOrCss(graph, in, out, outType, dims, params, css,
                            debugPrefix, options, debug);
  return out.get();
}

Tensor reduce(Graph &graph, const Tensor &in, const poplar::Type &outType,
              const std::vector<std::size_t> &dims, ReduceParams params,
              program::Sequence &prog, const std::string &debugPrefix,
              const poplar::OptionFlags &options, ReductionDebug *debug) {
  if (params.update)
    throw poputil::poplibs_error("Cannot do an update using reduce(); "
                                 "call reduceWithOutput() instead.");
  boost::optional<Tensor> out;
  reduceWithOutputProgOrCss(graph, in, out, outType, dims, params, prog,
                            debugPrefix, options, debug);
  return out.get();
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

} // namespace popops
