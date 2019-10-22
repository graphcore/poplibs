#include "CycleEstimationFunctions.hpp"

#include "ReductionConnection.hpp"
#include <poplibs_support/cyclesTables.hpp>
#include <poputil/VertexTemplates.hpp>

namespace popops {

namespace {

bool vectorised4ReductionOp(popops::Operation operation,
                            popops::ReductionSpecialisation specialisation) {
  if (specialisation == ReductionSpecialisation::SINGLE_OUTPUT_REGION ||
      specialisation == ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT) {
    return false;
  }
  return (operation == popops::Operation::MUL ||
          operation == popops::Operation::MAX ||
          operation == popops::Operation::MIN);
}

bool vectorised8ReductionOp(popops::Operation operation,
                            popops::ReductionSpecialisation specialisation) {
  if (specialisation == ReductionSpecialisation::SINGLE_OUTPUT_REGION ||
      specialisation == ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT) {
    return false;
  }
  return (operation == popops::Operation::ADD ||
          operation == popops::Operation::SQUARE_ADD);
}

// Get the sizes of a Vector<Vector<>> vertex field.
std::vector<std::size_t> fieldSizes(const poplar::FieldData &field) {

  std::vector<std::size_t> sizes(field.size());
  for (std::size_t i = 0; i < sizes.size(); ++i)
    sizes[i] = field[i].size();

  return sizes;
}

} // anonymous namespace

std::uint64_t getCyclesEstimateForReduce(
    const std::vector<std::size_t> &partialsSizes,
    const std::vector<std::size_t> &outSizes,
    const std::vector<unsigned> &numPartials, unsigned vectorWidth,
    const poplar::Type &partialsType, const poplar::Type &outType,
    popops::Operation operation, bool isUpdate,
    popops::ReductionSpecialisation specialisation) {

  // Total number of reductions.
  std::size_t numReductions = outSizes.size();

  // Additional cycles required for scaling, updating and format conversion.
  unsigned scaleAndUpdateCycles = isUpdate * 2 + 1;
  unsigned conversionCyles = outType == partialsType ? 0 : 1;

  // Total execution cycles.
  std::uint64_t cycles = 5 + 1 + 1; // entry/exit

  if (specialisation == ReductionSpecialisation::SINGLE_OUTPUT_REGION) {
    assert(numPartials.size() == 1);
    auto numOutputs = std::accumulate(outSizes.cbegin(), outSizes.cend(), 0u);
    cycles += 17; // non-loop overhead
    unsigned opPerLoop = 1;
    if (partialsType == poplar::FLOAT)
      opPerLoop = 2;
    else if (partialsType == poplar::HALF)
      opPerLoop = 4;
    // outer loop runs once per 64bits of input
    cycles += (numOutputs + opPerLoop - 1) / opPerLoop * 6;
    // double-width accumulation is not used
    cycles += partialsSizes[0] / vectorWidth; // inner loop
    return cycles;
  }
  if (specialisation == ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT) {
    // This is for the ASM vertices. The C vertices are worse
    cycles += 8;
    // other init / checking
    cycles += 11;
    auto accVectorWidth = 2 * vectorWidth;
    if (partialsSizes[0] < 6) {
      cycles += 6;
    } else {
      // 2 cycles per vector to avoid requiring 128B loads
      cycles += (partialsSizes[0] / accVectorWidth) * 2;
      // 1 cycle per element for the remainder
      cycles += partialsSizes[0] % accVectorWidth;
    }
    if (outType != poplar::FLOAT)
      cycles += 5;
    return cycles;
  }

  if (vectorised4ReductionOp(operation, specialisation)) {
    const unsigned partialsOverhead = 11;
    cycles += 3 + 23; // load state

    for (unsigned r = 0; r < numReductions; ++r) {
      const unsigned cyclesPerInnerLoop = vectorWidth == 2 ? 5 : 8;
      // Overhead - per reduction
      cycles += 16 +    // Unpack offsets and sizes
                5 +     // Check for vectorwidth loop being needed
                2 + 2 + // Check if remainders loops needed: 2, 1
                2;      // Loop end condition
      // Will there be a vector loop to execute?
      const unsigned vectorAccumulating = outSizes[r] / vectorWidth ? 1 : 0;
      // Number of remainder loops to execute
      const unsigned remLoops[] = {0, 1, 1, 2};
      const unsigned remainderAccumulating =
          remLoops[outSizes[r] % vectorWidth];
      // Account for overhead for setting up the 2 partials loops:
      // vectorAcc loop, and remAcc loop(s)
      cycles += vectorAccumulating * 6;
      cycles += remainderAccumulating * 4;

      for (unsigned i = 0; i < numPartials[r]; i++) {
        unsigned reductionRatio = 1;
        if (outSizes[r]) {
          reductionRatio = partialsSizes[i] / outSizes[r];
        }
        const unsigned vectorAccumulatingLoops =
            (outSizes[r] / vectorWidth) * reductionRatio;
        // Overhead in setting up the loop per vectorwidth piece of
        // partial (if there is a loop)
        if (reductionRatio && vectorAccumulating) {
          cycles += partialsOverhead * partialsSizes[i] /
                    (vectorWidth * reductionRatio);
        }
        // Inner loop for vector accumulation
        cycles += cyclesPerInnerLoop * vectorAccumulatingLoops;
        cycles += remainderAccumulating * partialsOverhead;
        if (outSizes[r] != 0) {
          // Inner loop(s) for remainder accumulation
          cycles += 4 * remainderAccumulating * partialsSizes[i] / outSizes[r];
        }
      }
    }
  } else if (vectorised8ReductionOp(operation, specialisation)) {
    // Double width operations/data proessed per loop
    vectorWidth *= 2;

    const unsigned partialsOverhead = 11;
    cycles += 3 + 23; // load state etc

    for (unsigned r = 0; r < numReductions; ++r) {
      // Each inner loop reads 128 bits.  Cycles taken varies based on alignment
      // (and consistent alignement for the latter pieces of the partials)
      // but is approximately given by this:
      const unsigned cyclesPerReadHalf = (outSizes[r] / vectorWidth) ? 11 : 6;
      const unsigned cyclesPerInnerLoop =
          (vectorWidth == 4) ? 7 : cyclesPerReadHalf + 3;
      // Overhead - per reduction
      cycles += 16 +        // Unpack offsets and sizes
                5 +         // Check for vectorwidth loop being needed
                2 + 2 + 2 + // Check if remainders loops needed: 4, 2, 1
                2;          // Loop end condition
      // Will there be a vector loop to execute?
      const unsigned vectorAccumulating = outSizes[r] / vectorWidth ? 1 : 0;
      // Number of remainder loops to execute based on remainder
      const unsigned remLoops[] = {0, 1, 1, 2, 1, 2, 2, 3};
      const unsigned remainderAccumulating =
          remLoops[outSizes[r] % vectorWidth];
      // Account for overhead of vectorAcc loop, and remAcc loop(s)
      cycles += vectorAccumulating * 22;
      cycles += remainderAccumulating * 8;

      for (unsigned i = 0; i < numPartials[r]; i++) {
        unsigned reductionRatio = 1;
        if (outSizes[r]) {
          reductionRatio = partialsSizes[i] / outSizes[r];
        }
        const unsigned vectorAccumulatingLoops =
            (outSizes[r] / vectorWidth) * reductionRatio;
        // Overhead in setting up the loop per vectorwidth piece of
        // partial (if there is a loop)
        if (reductionRatio && vectorAccumulating) {
          cycles += partialsOverhead * partialsSizes[i] /
                    (vectorWidth * reductionRatio);
        }
        // Inner loop for vector accumulation
        cycles += cyclesPerInnerLoop * vectorAccumulatingLoops;
        cycles += remainderAccumulating * partialsOverhead;
        if (outSizes[r] != 0) {
          // Inner loop(s) for remainder accumulation
          cycles += 8 * remainderAccumulating * partialsSizes[i] / outSizes[r];
        }
      }
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

        cycles += (2 * 1 + 1 + 3 + scaleAndUpdateCycles + conversionCyles) *
                  numVectorWidths;
        ++pi;
      }
    }
  }
  return cycles;
}

std::uint64_t getCycleEstimateReduceAllRegionsContinuous(
    const unsigned numPartials, const unsigned numOutputs,
    const unsigned vectorWidth, bool isUpdate) {
  // Estimate based on the code structure
  std::uint64_t cycles = numOutputs * (numPartials / vectorWidth);
  cycles += (numPartials & 1 ? 2 : 0);
  cycles += 12 * numOutputs;
  cycles += 10;
  if (isUpdate) {
    cycles = cycles + 1;
  }
  return cycles + 7; // Call / return overhead
}

std::uint64_t getCycleEstimateForReduceVertex(
    const poplar::VertexIntrospector &vertex, const poplar::Target &target,
    const poplar::Type &partialsType, const poplar::Type &outType,
    const popops::Operation operation, bool isUpdate,
    popops::ReductionSpecialisation specialisation) {

  std::vector<unsigned> numPartialEdges;
  std::vector<size_t> partialsPerEdge;
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  if (specialisation == ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT ||
      specialisation == ReductionSpecialisation::SINGLE_OUTPUT_REGION) {
    // single edge case
    // paritalsPerEdge takes the number of partials for the corresponding
    // output edge
    numPartialEdges.emplace_back(1);
    partialsPerEdge.emplace_back(partials.size());
  } else if (specialisation ==
             ReductionSpecialisation::ALL_REGIONS_CONTINUOUS) {
    CODELET_SCALAR_VAL(numPartials, unsigned);
    CODELET_SCALAR_VAL(numOutputs, unsigned);
    return getCycleEstimateReduceAllRegionsContinuous(
        numPartials, numOutputs, target.getVectorWidth(partialsType), isUpdate);
  } else {
    // partials is a 2D edge
    CODELET_VECTOR_VALS(numPartials, unsigned);
    numPartialEdges = numPartials;
    partialsPerEdge = fieldSizes(partials);
  }

  return getCyclesEstimateForReduce(
      partialsPerEdge, fieldSizes(out), numPartialEdges,
      target.getVectorWidth(partialsType), partialsType, outType, operation,
      isUpdate, specialisation);
}

std::uint64_t getCycleEstimateReducePartialsEqualSize(
    const unsigned outSize, const unsigned partialsSize,
    const unsigned numPartials, const unsigned outVectorWidth, bool isScale) {
  // Estimate based on the code structure, inner loop outwards
  std::uint64_t cycles = 4 * numPartials;
  cycles = (cycles + 5) * partialsSize;
  cycles = (cycles + 6) * outSize;
  cycles = cycles + 15;
  cycles = cycles + 2 * outSize * (outVectorWidth - 1);
  if (isScale) {
    cycles = cycles + 1;
  }
  return cycles + 7; // Call / return overhead
}
std::uint64_t getCycleEstimateForReducePartialsEqualSizeVertex(
    const poplar::VertexIntrospector &vertex, const poplar::Target &target,
    const poplar::Type &partialsType, const poplar::Type &outType,
    const popops::Operation operation, bool isUpdate, bool isScale) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  CODELET_SCALAR_VAL(outCount, short);
  CODELET_SCALAR_VAL(partialsSizeM1, short);

  return getCycleEstimateReducePartialsEqualSize(
      outCount, partialsSizeM1 + 1, partials.size(),
      target.getVectorWidth(outType), isScale);
}

} // namespace popops
