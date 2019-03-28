#include "CycleEstimationFunctions.hpp"

#include <poputil/VertexTemplates.hpp>
#include <poplibs_support/cyclesTables.hpp>
#include "ReductionConnection.hpp"

namespace popops {

namespace {

// Get the sizes of a Vector<Vector<>> vertex field.
std::vector<std::size_t> fieldSizes(const poplar::FieldData &field) {

  std::vector<std::size_t> sizes(field.size());
  for (std::size_t i = 0; i < sizes.size(); ++i)
    sizes[i] = field[i].size();

  return sizes;
}

} // anonymous namespace

std::uint64_t
getCyclesEstimateForReduce(const std::vector<std::size_t> &partialsSizes,
                           const std::vector<std::size_t> &outSizes,
                           const std::vector<unsigned> &numPartials,
                           unsigned vectorWidth,
                           const poplar::Type &partialsType,
                           const poplar::Type &outType,
                           popops::Operation operation,
                           bool isUpdate,
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
    cycles += numPartials[0] / vectorWidth; // inner loop
    return cycles;
  }
  if (specialisation == ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT) {
    // This is for the ASM vertices. The C vertices are worse
    cycles += 8;
    // other init / checking
    cycles += 11;
    auto accVectorWidth = 2 * vectorWidth;
    if (numPartials[0] < 6) {
      cycles += 6;
    } else {
      // 2 cycles per vector to avoid requiring 128B loads
      cycles += (numPartials[0] / accVectorWidth) * 2;
      // 1 cycle per element for the remainder
      cycles += numPartials[0] % accVectorWidth;
    }
    if (outType != poplar::FLOAT)
      cycles += 5;
    return cycles;
  }

  if (operation == popops::Operation::ADD ||
      operation == popops::Operation::SQUARE_ADD) { // Or ABS_ADD
    // VectorList costs 7 or 9 cycles to load n+base+descriptorPtr.
    // These vertices have two VectorList::DELTAN so we'll have one of each
    // and save a cycle (basemem only created once)
    cycles += 7 + 8 - 1;

    // Partial index.
    unsigned pi = 0;

    for (unsigned r = 0; r < numReductions; ++r) {
      cycles += 6; // Copied from above.

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

      for (unsigned i = 0; i < numPartials[r]; ++i) {
        auto numVectorWidths =
            (partialsSizes[pi] + 2 * vectorWidth - 1) / (2 * vectorWidth);

        cycles += (2 * 1 + 1 + 3 + scaleAndUpdateCycles + conversionCyles)
                  * numVectorWidths;
        ++pi;
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

        cycles += (2 * 1 + 1 + 3 + scaleAndUpdateCycles + conversionCyles)
                  * numVectorWidths;
        ++pi;
      }
    }
  }
  return cycles;
}


std::uint64_t
getCycleEstimateForReduceVertex(const poplar::VertexIntrospector &vertex,
                           const poplar::Target &target,
                           const poplar::Type &partialsType,
                           const poplar::Type &outType,
                           const popops::Operation operation,
                           bool isUpdate,
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
  } else {
    // partials is a 2D edge
    CODELET_VECTOR_VALS(numPartials, unsigned);
    numPartialEdges = numPartials;
    partialsPerEdge = fieldSizes(partials);
  }

  return getCyclesEstimateForReduce(partialsPerEdge,
                                    fieldSizes(out),
                                    numPartialEdges,
                                    target.getVectorWidth(partialsType),
                                    partialsType,
                                    outType,
                                    operation,
                                    isUpdate,
                                    specialisation);
}

}
