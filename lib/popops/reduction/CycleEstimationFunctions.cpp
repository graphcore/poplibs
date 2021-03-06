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

bool vectorisedOp(popops::Operation operation,
                  popops::ReductionSpecialisation specialisation) {
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

  // Additional cycles required for scaling, updating and format conversion.
  unsigned scaleAndUpdateCycles = isUpdate + 1;
  unsigned conversionCyles = outType == partialsType ? 0 : 1;

  const auto usesAccumulators = isAccumOp(operation);
  const auto opVectorWidth = usesAccumulators ? accVectorWidth : vectorWidth;
  assert(opVectorWidth % dataPathWidth == 0 ||
         dataPathWidth % opVectorWidth == 0);
  const auto interleaveFactor = getForceInterleavedEstimates() ? 2 : 1;
  // Total execution cycles.
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit

  if (specialisation == ReductionSpecialisation::STRIDED_REDUCE) {
    assert(bool(stride));
    assert(numPartials.size() == 1);
    auto numOutputs = std::accumulate(outSizes.cbegin(), outSizes.cend(), 0u);
    cycles += 17; // non-loop overhead

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
        gcd(std::min(opVectorWidth, dataPathWidth * interleaveFactor), *stride);

    // Estimate the number of outer loops. If elemsPerLoop is equivalent to
    // 64-bits of input data per cycle this is exact.
    const auto numOuterLoops = ceildiv(numOutputs, elemsPerLoop);

    cycles += numOuterLoops * 6;
    cycles += cyclesPerOp * partialsSizes[0] / elemsPerLoop; // inner loop
    std::uint64_t flops = static_cast<std::uint64_t>(numOutputs) *
                          (partialsSizes[0] + isUpdate - 1) *
                          flopsForReduceOp(operation);
    return {cycles, convertToTypeFlops(flops, outType)};
  }
  if (specialisation == ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT) {
    assert(accVectorWidth % vectorWidth == 0);
    // Take the minimum of the data path width * interleaved factor and
    // the op width to get the elements we can do per cycle. This is for the ASM
    // vertices. The C vertices are worse
    const auto elemsPerCycle =
        std::min(opVectorWidth, dataPathWidth * interleaveFactor);
    cycles += 8;
    // other init / checking
    cycles += 11;
    if (partialsSizes[0] < 6) {
      cycles += 6;
    } else {
      cycles += cyclesPerOp * partialsSizes[0] / elemsPerCycle;
      // 1 cycle per element for the remainder
      cycles += partialsSizes[0] % elemsPerCycle;
    }
    if (outType != poplar::FLOAT)
      cycles += 5;
    // Additional operation for update
    std::uint64_t flops =
        (static_cast<std::uint64_t>(partialsSizes[0]) + isUpdate - 1) *
        flopsForReduceOp(operation);
    return {cycles, convertToTypeFlops(flops, outType)};
  }

  // Trying to generalise vectorised vertex estimates...
  std::uint64_t flops = 0;
  if (vectorisedOp(operation, specialisation)) {
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
    // Non-accumulator code.
    cycles += 2;
    if (specialisation == ReductionSpecialisation::DEFAULT ||
        specialisation == ReductionSpecialisation::SCALAR_OUTPUT_REGIONS) {
      // VectorList costs 7 or 9 cycles to load n+base+descriptorPtr.
      // These vertices have two VectorList::DELTAN so we'll have one of each
      // and save a cycle (basemem only created once)
      cycles += 7 + 8 - 1;
    } else {
      // Two SCALED_PTR32 to load, base only created once
      cycles += 3 + 3 + 1;
    }
    // Partial index.
    unsigned pi = 0;

    for (unsigned r = 0; r < numReductions; ++r) {
      cycles += 5; // Copied from above.

      // Calculate the maximum vector width we can actually use for this
      // reduction.
      unsigned usableVectorWidth = vectorWidth;
      while (outSizes[r] % usableVectorWidth != 0)
        usableVectorWidth /= 2;
      // This isn't quite right, but it will be much easier to estimate when
      // I've actually written the assembly.

      // I think we can calculate the above by just examining the lower bits
      // of out[r].size() and using a jump table. Conservative guess:
      cycles += 5;

      // This is all a quite wild guess.
      for (unsigned i = 0; i < numPartials[r]; ++i) {
        auto numVectorWidths =
            (partialsSizes[pi] + vectorWidth - 1) / vectorWidth;
        flops +=
            (partialsSizes[pi] + isUpdate - 1) * flopsForReduceOp(operation);
        cycles +=
            (2 * 1 + cyclesPerOp + 3 + scaleAndUpdateCycles + conversionCyles) *
            numVectorWidths;
        ++pi;
      }
    }
  }
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

// TODO: this does not take into account vertices that include scaling.
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
    // Total = 3 + (2*2) + 1 + (2*2) + 1 = 12
    if (logAddHasAssembler) {
      return partialsType == poplar::HALF ? 12 : 23;
    }
    // Approx result in C++ log-add by comparing to Sim execution time
    // (This is for a scalar as per opVectorWidth)
    return 20;
  }();

  const auto partialsPer64Bits = partialsTypeSize / 8;
  const auto dataPathWidth = target.getDataPathWidth() / (partialsTypeSize * 8);
  std::vector<unsigned> numPartialEdges;
  std::vector<size_t> partialsPerEdge;
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  std::optional<unsigned> stride;
  if (specialisation == ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT) {
    // single edge case
    // paritalsPerEdge takes the number of partials for the corresponding
    // output edge
    numPartialEdges.emplace_back(1);
    partialsPerEdge.emplace_back(partials.size());
  } else if (specialisation == ReductionSpecialisation::STRIDED_REDUCE) {
    numPartialEdges.emplace_back(1);
    CODELET_SCALAR_VAL(numPartialsM1, unsigned);
    CODELET_SCALAR_VAL(numOutputs, unsigned);
    CODELET_SCALAR_VAL(partialsWidth, unsigned);
    partialsPerEdge.emplace_back((numPartialsM1 + 1) * numOutputs);
    stride = partialsWidth;
  } else if (specialisation ==
             ReductionSpecialisation::ALL_REGIONS_CONTINUOUS) {
    CODELET_SCALAR_VAL(numPartials, unsigned);
    CODELET_SCALAR_VAL(numOutputsM1, unsigned);
    return getCycleEstimateReduceAllRegionsContinuous(
        numPartials, numOutputsM1, dataPathWidth, opVectorWidth, cyclesPerOp,
        isUpdate, outType, operation);
  } else {
    // partials is a 2D edge
    CODELET_VECTOR_VALS(numPartials, unsigned);
    numPartialEdges = numPartials;
    partialsPerEdge = fieldSizes(partials);
  }

  return getCyclesEstimateForReduce(
      partialsPerEdge, fieldSizes(out), numPartialEdges, stride, dataPathWidth,
      target.getVectorWidth(partialsType), opVectorWidth,
      target.getVectorWidth(outType), partialsPer64Bits, partialsType, outType,
      operation, cyclesPerOp, isUpdate, specialisation);
}

} // namespace popops
