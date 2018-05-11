#include "CycleEstimationFunctions.hpp"

#include <poputil/VertexTemplates.hpp>
#include <poplibs_support/cyclesTables.hpp>

namespace popops {

namespace {

// Get the sizes of a Vector<Vector<>> vertex field.
std::vector<std::size_t> fieldSizes(const poplar::FieldData &field) {

  std::vector<std::size_t> sizes(field.size());
  for (std::size_t i = 0; i < sizes.size(); ++i)
    sizes[i] = field[i].size();

  return sizes;
}

// Return true if every partial is the same size as its corresponding output.
bool partialsAreOutputSize(const std::vector<std::size_t> &partialsSizes,
                           const std::vector<std::size_t> &outSizes,
                           const std::vector<unsigned> &numPartials) {
  unsigned pi = 0;
  for (unsigned oi = 0; oi < outSizes.size(); ++oi) {
    for (unsigned i = 0; i < numPartials[oi]; ++i) {
      if (partialsSizes[pi] != outSizes[oi])
        return false;
      ++pi;
    }
  }
  return true;
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
                           bool isScale) {

  // If we know that the partials for each region are always exactly the same
  // size as the output region size, we can use a faster implementation.
  bool equalPartials = partialsAreOutputSize(partialsSizes,
                                             outSizes,
                                             numPartials);

  // Total number of reductions.
  std::size_t numReductions = outSizes.size();

  // Additional cycles required for scaling, updating and format conversion.
  unsigned scaleAndUpdateCycles = isUpdate * 2 + isScale * 1;
  unsigned conversionCyles = outType == partialsType ? 0 : 1;

  // Total execution cycles.
  std::uint64_t cycles = 0;

  if (operation == popops::Operation::ADD ||
      operation == popops::Operation::SQUARE_ADD) { // Or ABS_ADD
    cycles = 5 + 1;
    // VectorList costs 7 or 9 cycles to load n+base+descriptorPtr.
    // These vertices have two VectorList::DELTAN so we'll have one of each and
    // save a cycle (basemem only created once)
    cycles += 7 + 8 - 1;

    if (equalPartials) {
      // Based on the previous reduceCycleEstimate() "version 1".

      // Innermost loop accumulates vector across all input tiles

      // This estimate based on float->float code
      // Inner loop processes 128bits/2cycles
      // Inner loop cycles would halve for strided data given
      // f32v4add IS addition
      for (unsigned r = 0; r < numReductions; ++r) {
        cycles += 6;
        auto numVectorWidths =
            (outSizes[r] + 2 * vectorWidth - 1) / (2 * vectorWidth);
        cycles += (2 * numPartials[r] + 1 + 3 + scaleAndUpdateCycles
                   + conversionCyles)
                  * numVectorWidths;
      }
    } else {
      // Similar but we need an inner inner loop. And it will be slower
      // for some output sizes since we can't guarantee alignment.
      // E.g. if the output size is 9 and the partial size is 27.

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
    }
  } else {
    // Non-add code.
    cycles = 8;
    // VectorList costs 7 or 9 cycles to load n+base+descriptorPtr.
    // These vertices have two VectorList::DELTAN so we'll have one of each and
    // save a cycle (basemem only created once)
    cycles += 7 + 8 - 1;

    if (equalPartials) {
      // Based on the previous reduceOpsCycleEstimate().

      for (unsigned r = 0; r < numReductions; ++r) {
        // Overhead for each reduction
        cycles += 5;
        auto numElem = outSizes[r];
        const unsigned numVectors = (numElem + vectorWidth - 1) / vectorWidth;

        // process vectorWidth at a time with ld2xstpace. This may not be the
        // best option if numVectors is small overhead of 5: ld ptrs, rpt,
        // brnzdec
        cycles += (numPartials[r] + 5 + scaleAndUpdateCycles + conversionCyles)
                  * numVectors;
      }

    } else {
      // Non-equal partials, non-add.

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
                           bool isScale) {

  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  CODELET_VECTOR_VALS(numPartials, unsigned);

  return getCyclesEstimateForReduce(fieldSizes(partials),
                                          fieldSizes(out),
                                          numPartials,
                                          target.getVectorWidth(partialsType),
                                          partialsType,
                                          outType,
                                          operation,
                                          isUpdate,
                                          isScale);
}

}
