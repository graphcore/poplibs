// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "CycleEstimationFunctions.hpp"

#include "ReductionConnection.hpp"
#include "poputil/exceptions.hpp"
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/FlopEstimation.hpp>
#include <poplibs_support/cyclesTables.hpp>
#include <poplibs_support/forceInterleavedEstimates.hpp>
#include <poplibs_support/gcd.hpp>
#include <poputil/VertexTemplates.hpp>

using namespace poplibs_support;

namespace popops {

namespace {

bool isAccumOp(Operation operation) {
  return (operation == Operation::ADD || operation == Operation::SQUARE_ADD);
}

bool vectorisedOp(popops::Operation operation) {
  return (operation == Operation::MUL || operation == Operation::MAX ||
          operation == Operation::MIN || operation == Operation::ADD ||
          operation == Operation::SQUARE_ADD);
}

// Get the sizes of a Vector<Vector<>> vertex field.
std::vector<std::size_t> fieldSizes(const poplar::FieldData &field) {

  std::vector<std::size_t> sizes(field.size());
  for (std::size_t i = 0; i < sizes.size(); ++i)
    sizes[i] = field[i].size();

  return sizes;
}

unsigned flopsForReduceOp(popops::Operation operation) {
  switch (operation) {
  case popops::Operation::ADD:
    return flopsForAdd();
  case popops::Operation::MAX:
  case popops::Operation::MIN:
    return 1;
  case popops::Operation::MUL:
    return flopsForMultiply();
  case popops::Operation::LOG_ADD:
    return flopsForLogAdd();
  case popops::Operation::LOGICAL_AND:
  case popops::Operation::LOGICAL_OR:
    return 0;
  case popops::Operation::SQUARE_ADD:
    return flopsForMultiply() + flopsForAdd();
  default:
    throw poputil::poplibs_error("Unsupported operation type");
  }
}

} // anonymous namespace

poplar::VertexPerfEstimate getCyclesEstimateForReduce(
    const std::vector<std::size_t> &partialsSizes,
    const std::vector<std::size_t> &outSizes,
    const std::vector<unsigned> &numPartials,
    const std::optional<unsigned> &stride, const unsigned dataPathWidth,
    unsigned vectorWidth, const unsigned accVectorWidth,
    unsigned outTypeVectorWidth, const unsigned partialsPer64Bits,
    const poplar::Type &partialsType, const poplar::Type &outType,
    popops::Operation operation, const unsigned cyclesPerOp, bool isUpdate,
    popops::ReductionSpecialisation specialisation) {

  // Total number of reductions.
  std::size_t numReductions = outSizes.size();

  unsigned conversionCycles = outType == partialsType ? 0 : 1;

  const auto usesAccumulators = isAccumOp(operation);
  const auto opVectorWidth = usesAccumulators ? accVectorWidth : vectorWidth;
  assert(opVectorWidth % dataPathWidth == 0 ||
         dataPathWidth % opVectorWidth == 0);
  const auto interleaveFactor = getForceInterleavedEstimates() ? 2 : 1;
  // Total execution cycles.
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit
  // Trying to generalise vectorised vertex estimates...
  std::uint64_t flops = 0;
  if (vectorisedOp(operation)) {
    cycles += 3 + 23; // load state etc.

    // We do scale + store in output type so the cycles for scale, update,
    // and store are a function of the number of output type vector widths
    // that fit into the op vector width. When using accumulators there
    // is a cycle's delay on getting the first value for scale/update/
    // store.
    const auto getStoreCycles = [&](const auto width) {
      assert(width % outTypeVectorWidth == 0 ||
             outTypeVectorWidth % width == 0);
      return (2 + isUpdate) * std::min(width, outTypeVectorWidth) +
             usesAccumulators;
    };

    assert((opVectorWidth & (opVectorWidth - 1)) == 0);
    std::size_t partialsIndex = 0;
    for (unsigned r = 0; r < numReductions; ++r) {
      // Load cycles are based on how many cycles it takes
      // to call to a function that handles mis-aligned data
      // but assuming we take the fastest (most aligned) path
      // so this doesn't (or shouldn't) vary based on vector width but
      // our assembly differs between using a function call and
      // an inline load between accumulator/non-accumulator version.
      const auto loadOverhead = usesAccumulators ? 2 : 0; // call, br $lr
      // In the best case we get 2 cycles to check alignment and then
      // load. The data path width and the interleave factor determines how
      // quickly we can load each vector. Lastly we have 2 cycles to check for
      // loop exit and branch back to the start of the loop.
      const auto loadWidth =
          std::min(opVectorWidth, dataPathWidth * interleaveFactor);
      const auto cyclesPerVectorLoad =
          loadOverhead + 2 + opVectorWidth / loadWidth;

      // Will there be a vector loop to execute?
      const unsigned vectorOuterLoops = outSizes[r] / opVectorWidth;
      const auto remaining = outSizes[r] % opVectorWidth;
      // The number of checks left to do
      const auto remainingChecks = ceilLog2(opVectorWidth);

      // Cycles for _Reduce_zero_and_load excluding call
      constexpr unsigned reduceZeroAndLoadCycles = 3;
      // Cycles for _Reduce_ptr_fetch excluding call
      constexpr unsigned fetchPtrCycles = 7;

      // Common per-reduction overhead:
      // setzi, ld32, call _Reduce_outer_loop_setup
      cycles += 16;
      // Setup remainder and check for vector-width loop being needed
      cycles += 6;
      // Check if remainders loops needed: 2 cycles for each
      cycles += remainingChecks * 2;
      // Loop end condition
      cycles += 2;

      // Cycles per outer loop over vectors (pre-loop cycles, store/scale/update
      // cycles + brnzdec)
      cycles += vectorOuterLoops *
                ((usesAccumulators ? reduceZeroAndLoadCycles + 2 : 3) +
                 getStoreCycles(opVectorWidth) + 1);
      // Cycles per outer loop over vectors for numPartials[r]
      cycles += vectorOuterLoops * numPartials[r] * (fetchPtrCycles + 3);
      flops += (numPartials[r] + isUpdate - 1) * outSizes[r] *
               flopsForReduceOp(operation);

      for (unsigned i = 0; i < remainingChecks; ++i) {
        const auto remainder = (1u << i);
        if (remaining & remainder) {
          cycles += (usesAccumulators ? reduceZeroAndLoadCycles + 1 : 2) +
                    getStoreCycles(remainder);
          cycles += numPartials[r] * (fetchPtrCycles + 3);
        }
      }

      for (std::size_t i = partialsIndex; i < partialsIndex + numPartials[r];
           ++i) {
        const auto numAccumulations = partialsSizes[i] / outSizes[r];
        const auto vectorLoops = vectorOuterLoops * numAccumulations;
        // Cycles per partial vector (load, compute, loop)
        cycles += vectorLoops * (cyclesPerVectorLoad + 3);
        for (unsigned j = 0; j < remainingChecks; ++j) {
          const auto remainder = (1u << j);
          if (remaining & remainder) {
            // Make the assumption that each load is a check for alignment
            // (2 cycles) and however many loads of data path width it takes
            // to load the remainder, all inline.
            const auto numLoads =
                remainder /
                std::min(remainder, dataPathWidth * interleaveFactor);
            cycles += numAccumulations * (2 + numLoads + 3);
          }
        }
      }
      partialsIndex += numPartials[r];
    }
  } else {
    assert(operation == Operation::LOGICAL_AND ||
           operation == Operation::LOGICAL_OR ||
           operation == Operation::LOG_ADD);

    // The following numbers are approximations for C++ codelets.
    // VectorList costs ~12 cycles to load n+base+descriptorPtr for a single
    // pointer. These vertices have two VectorList::DELTANELEMENTS.
    cycles += 12 * 2;

    // Partial index.
    unsigned pi = 0;

    auto cyclesPerCppOp = operation == Operation::LOG_ADD ? 12 : cyclesPerOp;
    for (unsigned r = 0; r < numReductions; ++r) {
      cycles += 8;
      for (unsigned i = 0; i < numPartials[r]; ++i) {
        flops +=
            (partialsSizes[pi] + isUpdate - 1) * flopsForReduceOp(operation);
        cycles += 22 + // pointer, offset, bound checking etc.
                  (24 + cyclesPerCppOp + conversionCycles) * partialsSizes[pi];
        ++pi;
      }
    }
  }
  return {cycles, convertToTypeFlops(flops, outType)};
}

poplar::VertexPerfEstimate getCyclesEstimateForStridedReduce(
    const std::size_t partialsSize, const std::size_t numPartials,
    const std::size_t numOutputs, const unsigned stride,
    const unsigned numOuterStrides, const unsigned dataPathWidth,
    const unsigned vectorWidth, const unsigned accVectorWidth,
    const poplar::Type &partialsType, const poplar::Type &outType,
    popops::Operation operation, const unsigned cyclesPerOp, bool isUpdate) {
  const auto usesAccumulators = isAccumOp(operation);
  const auto opVectorWidth = usesAccumulators ? accVectorWidth : vectorWidth;
  assert(opVectorWidth % dataPathWidth == 0 ||
         dataPathWidth % opVectorWidth == 0);
  const auto interleaveFactor = getForceInterleavedEstimates() ? 2 : 1;

  // entry/exit
  std::uint64_t cycles = 2 + 2;

  // loop setup
  cycles += 6 + 5;

  unsigned cyclesLoopPost = 0;
  if (partialsType == poplar::HALF) {
    // inner loop completion
    cyclesLoopPost++;
    // result storage depending on type, operation and isUpdate
    std::map<poplar::Type, std::map<Operation, std::map<bool, unsigned>>>
        cycleMap = {{poplar::HALF,
                     {{Operation::ADD, {{false, 7}, {true, 9}}},
                      {Operation::MAX, {{false, 7}, {true, 10}}},
                      {Operation::MIN, {{false, 7}, {true, 10}}},
                      {Operation::LOG_ADD, {{false, 7}, {true, 22}}}}},
                    {poplar::FLOAT,
                     {{Operation::ADD, {{false, 6}, {true, 11}}},
                      {Operation::MAX, {{false, 7}, {true, 10}}},
                      {Operation::MIN, {{false, 7}, {true, 10}}},
                      {Operation::LOG_ADD, {{false, 7}, {true, 56}}}}}};
    cyclesLoopPost += cycleMap[outType][operation][isUpdate];
  } else {
    bool isOpAdd = (operation == Operation::ADD) ? true : false;
    bool isOpLogAdd = (operation == Operation::LOG_ADD) ? true : false;
    // inner loop completion
    cyclesLoopPost += isOpAdd ? 3 : 2;
    // result storage depending on type, operation and isUpdate
    std::map<poplar::Type, std::map<bool, std::map<bool, unsigned>>> cycleMap =
        {{poplar::HALF,
          {{false, {{false, 2}, {true, 4}}}, {true, {{false, 2}, {true, 13}}}}},
         {poplar::FLOAT,
          {{false, {{false, 2}, {true, 5}}},
           {true, {{false, 2}, {true, 27}}}}}};
    cyclesLoopPost += cycleMap[outType][isOpLogAdd][isUpdate];
  }
  // outer loop
  auto cyclesOuter = 3 + cyclesLoopPost;
  // numOuterStrides inner loops are executed
  cyclesOuter += numOuterStrides * (3 + cyclesPerOp * (numPartials - 1));
  // The elements processed per loop is limited by the data path width, the
  // interleave factor, and the op width, so with 64-bit data path width and
  // 128-bit accumulator based ops we only manage 64-bits per-cycle. With the
  // same scenario with interleave factor 2,  we can manage 64 * 2 bits
  // per-cycle.
  //
  // Additionally when performing modelling experiments, elemsPerLoop may
  // be greater than that for 64-bits but our graph construction allows
  // a stride with any multiple of 64-bits. In order to do elemsPerCycle
  // each loop we need the stride to be compatible with the number of
  // elements processed per cycle or else we won't be aligned on the
  // second (or more) iteration(s) hence we take a gcd.
  const auto elemsPerLoop =
      gcd(std::min(opVectorWidth, dataPathWidth * interleaveFactor), stride);

  // Estimate the number of outer loops. If elemsPerLoop is equivalent to
  // 64-bits of input data per cycle this is exact.
  const auto numOuterLoops = ceildiv(numOutputs, elemsPerLoop);
  cycles += cyclesOuter * numOuterLoops;

  std::uint64_t flops = static_cast<std::uint64_t>(numOutputs) *
                        (partialsSize + isUpdate - 1) * numOuterStrides *
                        flopsForReduceOp(operation);
  return {cycles, convertToTypeFlops(flops, outType)};
}

poplar::VertexPerfEstimate getCyclesEstimateForSingleInput(
    const std::size_t partialsSize, const unsigned dataPathWidth,
    const unsigned vectorWidth, const unsigned accVectorWidth,
    const poplar::Type &outType, const popops::Operation operation,
    const unsigned cyclesPerOp, bool isUpdate) {
  assert(accVectorWidth % vectorWidth == 0);
  const auto usesAccumulators = isAccumOp(operation);
  const auto opVectorWidth = usesAccumulators ? accVectorWidth : vectorWidth;
  assert(opVectorWidth % dataPathWidth == 0 ||
         dataPathWidth % opVectorWidth == 0);
  const auto interleaveFactor = getForceInterleavedEstimates() ? 2 : 1;

  // Take the minimum of the data path width * interleaved factor and
  // the op width to get the elements we can do per cycle. This is for the ASM
  // vertices. The C vertices are worse
  const auto elemsPerCycle =
      std::min(opVectorWidth, dataPathWidth * interleaveFactor);
  const auto loadsPerLoop = opVectorWidth / elemsPerCycle;
  const auto cyclesPerLoop = std::max(loadsPerLoop, cyclesPerOp);
  const auto elemsPerLoop = elemsPerCycle * loadsPerLoop;

  std::uint64_t cycles = 1 + 2; // entry, storage and exit
  // other init / checking
  cycles += 6;
  cycles += cyclesPerLoop * partialsSize / elemsPerLoop;
  // calculate remainder, load from accumulators
  cycles += 5;
  // 1 cycle per element for the remainder
  cycles += partialsSize % elemsPerLoop;
  // sum up all accumulators
  cycles += 3;

  if (outType != poplar::FLOAT)
    cycles += 5;
  // Additional operation for update
  std::uint64_t flops =
      (static_cast<std::uint64_t>(partialsSize) + isUpdate - 1) *
      flopsForReduceOp(operation);
  return {cycles, convertToTypeFlops(flops, outType)};
}

poplar::VertexPerfEstimate getCycleEstimateReduceAllRegionsContinuous(
    const unsigned numPartials, const unsigned numOutputs,
    const unsigned dataPathWidth, const unsigned accVectorWidth,
    const unsigned cyclesPerOp, bool isUpdate, const poplar::Type &type,
    popops::Operation operation) {
  assert(accVectorWidth % dataPathWidth == 0 ||
         dataPathWidth % accVectorWidth == 0);

  // elements per cycle is the minimum of the accumulator vector width
  // and the data path width times the interleave factor.
  const auto interleaveFactor = getForceInterleavedEstimates() ? 2 : 1;
  const auto elemsPerInnerLoop =
      std::min(accVectorWidth, dataPathWidth * interleaveFactor);
  // Estimate based on the code structure
  std::uint64_t cycles =
      cyclesPerOp * numOutputs * (numPartials / elemsPerInnerLoop);
  cycles += (numPartials & 1 ? 2 : 0);
  cycles += 12 * numOutputs;
  cycles += 10;
  if (isUpdate) {
    cycles = cycles + 1;
  }
  std::uint64_t flops = static_cast<std::uint64_t>(numOutputs) *
                        (numPartials + isUpdate - 1) *
                        flopsForReduceOp(operation);
  return {cycles + 7, // Call / return overhead
          convertToTypeFlops(flops, type)};
}

// For specialisations which support scaled as well as unscaled reduction, the
// assembly implementation is based on the scaled version. The unscaled version
// forms a special case with the scale factor assigned to 1.0. Hence there is
// no signficant difference between the cycle estimates between scaled and
// unscaled versions and for simplicity this aspect has been omitted from all
// the estimates.
poplar::VertexPerfEstimate getCycleEstimateForReduceVertex(
    const poplar::VertexIntrospector &vertex, const poplar::Target &target,
    const poplar::Type &partialsType, const poplar::Type &outType,
    const popops::Operation operation, bool isUpdate,
    popops::ReductionSpecialisation specialisation) {

  const auto partialsTypeSize = target.getTypeSize(partialsType);
  const auto accVectorWidth = partialsType == poplar::HALF    ? 8
                              : partialsType == poplar::FLOAT ? 4
                                                              : 1;
  const auto opIsLogAdd = operation == Operation::LOG_ADD;
  const auto logAddHasAssembler =
      specialisation == ReductionSpecialisation::STRIDED_REDUCE;

  const auto opVectorWidth = [&]() {
    if (!opIsLogAdd) {
      return accVectorWidth;
    }
    if (logAddHasAssembler) {
      return partialsType == poplar::HALF ? 4 : 2;
    }
    // C++ log-add is scalar
    return 1;
  }();

  const auto cyclesPerOp = [&]() {
    // Most operations take a single cycle and are implemented in assembler.
    if (!opIsLogAdd) {
      return 1;
    }
    // Log-add in assembler:
    // f32v2: 3 cycles (min,max,sub)
    //      + 3 * 2 (2 of f32exp)
    //      + 1 (add 1)
    //      + 6 * 2 (2 of f32log)
    //      + 1 add
    // Total: 23
    // For the f16v4 variant the exp, log instructions take 2 cycles each,
    // Total = 3 + (2*2) + 1 + (2*2) + 1 = 13
    if (logAddHasAssembler) {
      return partialsType == poplar::HALF ? 13 : 23;
    }
    // Approx result in C++ log-add by comparing to Sim execution time
    // (This is for a scalar as per opVectorWidth)
    return 20;
  }();

  const auto partialsPer64Bits = partialsTypeSize / 8;
  const auto dataPathWidth = target.getDataPathWidth() / (partialsTypeSize * 8);
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  std::optional<unsigned> stride;
  if (specialisation == ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT) {
    assert(out.size() == 1);
    // single edge case
    // paritalsPerEdge takes the number of partials for the corresponding
    // output edge
    CODELET_SCALAR_VAL(numPartials, unsigned);
    return getCyclesEstimateForSingleInput(
        numPartials, dataPathWidth, target.getVectorWidth(partialsType),
        opVectorWidth, outType, operation, cyclesPerOp, isUpdate);
  } else if (specialisation == ReductionSpecialisation::STRIDED_REDUCE) {
    CODELET_SCALAR_VAL(numPartialsM1, unsigned);
    CODELET_SCALAR_VAL(numOutputs, unsigned);
    CODELET_SCALAR_VAL(numOuterStridesM1, unsigned);
    auto numPartials = numPartialsM1 + 1;
    auto numOuterStrides = numOuterStridesM1 + 1;
    auto partialsPerEdge = numPartials * numOutputs;
    return getCyclesEstimateForStridedReduce(
        partialsPerEdge, numPartials, numOutputs, *stride, numOuterStrides,
        dataPathWidth, target.getVectorWidth(partialsType), opVectorWidth,
        partialsType, outType, operation, cyclesPerOp, isUpdate);
  } else if (specialisation ==
             ReductionSpecialisation::ALL_REGIONS_CONTINUOUS) {
    CODELET_SCALAR_VAL(numPartials, unsigned);
    CODELET_SCALAR_VAL(numOutputsM1, unsigned);
    return getCycleEstimateReduceAllRegionsContinuous(
        numPartials, numOutputsM1, dataPathWidth, opVectorWidth, cyclesPerOp,
        isUpdate, outType, operation);
  }
  assert((specialisation == ReductionSpecialisation::DEFAULT) ||
         (specialisation == ReductionSpecialisation::SCALAR_OUTPUT_REGIONS));

  // partials is a 2D edge
  CODELET_VECTOR_VALS(numPartials, unsigned);
  auto partialsPerEdge = fieldSizes(partials);
  return getCyclesEstimateForReduce(
      partialsPerEdge, fieldSizes(out), numPartials, stride, dataPathWidth,
      target.getVectorWidth(partialsType), opVectorWidth,
      target.getVectorWidth(outType), partialsPer64Bits, partialsType, outType,
      operation, cyclesPerOp, isUpdate, specialisation);
}

} // namespace popops
