#include "popopsCycleEstimators.hpp"
#include "PerformanceEstimation.hpp"

using namespace poplar;

namespace popops {

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceAdd)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &outType,
                                     const Type &partialsType) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  std::vector<unsigned> outSizes;
  for (auto i = 0u; i < out.size(); ++i) outSizes.push_back(out[i].size());

  return reduceCycleEstimate(outSizes,
                             partials.size(),
                             dataPathWidth,
                             false, false,
                             outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceAddUpdate)(const VertexIntrospector &vertex,
                                           const Target &target,
                                           const Type &outType,
                                           const Type &partialsType) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  std::vector<unsigned> outSizes;
  for (auto i = 0u; i < out.size(); ++i) outSizes.push_back(out[i].size());

  return reduceCycleEstimate(outSizes,
                             partials.size(),
                             dataPathWidth,
                             true, false,
                             outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceAddScale)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &outType,
                                          const Type &partialsType) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  std::vector<unsigned> outSizes;
  for (auto i = 0u; i < out.size(); ++i) outSizes.push_back(out[i].size());

  return reduceCycleEstimate(outSizes,
                             partials.size(),
                             dataPathWidth,
                             false, true,
                             outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceMul)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &outType,
                                     const Type &partialsType) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  std::vector<unsigned> outSizes;
  for (auto i = 0u; i < out.size(); ++i) outSizes.push_back(out[i].size());

  return reduceOpsCycleEstimate(outSizes,
                                partials.size(),
                                dataPathWidth,
                                outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceMax)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &outType,
                                     const Type &partialsType) {
  return
    MAKE_CYCLE_ESTIMATOR_NAME(ReduceMul)(vertex, target, outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceMin)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &outType,
                                     const Type &partialsType) {
  return
    MAKE_CYCLE_ESTIMATOR_NAME(ReduceMul)(vertex, target, outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceAnd)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &outType,
                                     const Type &partialsType) {
  return
    MAKE_CYCLE_ESTIMATOR_NAME(ReduceMul)(vertex, target, outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceOr)(const VertexIntrospector &vertex,
                                    const Target &target,
                                    const Type &outType,
                                    const Type &partialsType) {
  return
    MAKE_CYCLE_ESTIMATOR_NAME(ReduceAnd)(vertex, target, outType, partialsType);
}



/* Cycle cost computation for basic operations */
static uint64_t basicOpLoopCycles(unsigned overhead,
                                  unsigned numElems,
                                  unsigned vectorSize,
                                  unsigned cyclesPerVector) {
  return overhead + (numElems + vectorSize - 1) / vectorSize  * cyclesPerVector;
}

/* Cycles for comparison operations which result in bool as output.
 * For boolean inputs the number of cycles depend on the type of operation
 * as some ops have to be synthesized from the available instruction set
 */
static uint64_t comparisonOpsCycles(unsigned dataPathWidth,
                                    unsigned numElems,
                                    unsigned boolInputComputeCycles,
                                    Type type) {
  if (type == FLOAT) {
    unsigned vectorWidth = dataPathWidth / 32;
    if (sizeof(bool) == 4) {
      // for dataPathWidth = 64:
      // ld64/cmp, ldst64/and on aux
      return basicOpLoopCycles(5, numElems, vectorWidth, 2);
    } else if (sizeof(bool) == 2) {
      // for dataPathWidth = 64:
      // ld64/cmp, ld64/and, st32/sort16
      return basicOpLoopCycles(5, numElems, vectorWidth, 3);
    } else if (sizeof(bool) == 1) {
      // for dataPathWidth = 64:
      // (ld64/cmp, ld64/and, sort16, atom) * 2 on aux
      //   shuf8, shl16, or, st32 on main
      return basicOpLoopCycles(5, numElems, 4 / vectorWidth,
                               (4 / vectorWidth) * 4 + 5);
    }
  } else if (type == HALF) {
    unsigned vectorWidth = dataPathWidth / 32;
    if (sizeof(bool) == 4) {
      // for dataPathWidth = 64:
      // ld64/cmp, ld64/and
      // sort16, sort16/st64
      return basicOpLoopCycles(5, numElems, vectorWidth, 2 + 2 * vectorWidth);
    } else if (sizeof(bool) == 2) {
      // ldst64/cmp, ld64/amp
      return basicOpLoopCycles(5, numElems, vectorWidth, 2);
    } else if (sizeof(bool) == 1) {
      // for dataPathWidth = 64:
      // (ld64/cmp, ld64/and, sort16, atom) * 2 on aux
      //   shuf8, shl16, or, st32 on main
      return basicOpLoopCycles(5, numElems, 4 / vectorWidth,
                               (4 / vectorWidth) * 4 + 2);
    }
  } else if (type == INT) {
    if (sizeof(bool) == 4) {
      return basicOpLoopCycles(5, numElems, 1, 4);
    } else if (sizeof(bool) == 2) {
      // (ld32, ld32, cmp) * 2, sort16, sort16, st32
      return basicOpLoopCycles(5, numElems, 2, 9);
    } else if (sizeof(bool) == 1) {
      // (ld32, ld32, cmp) * 4, sort16, sort16, sort8, st32
      return basicOpLoopCycles(5, numElems, 4, 16);
    }
  } else if (type == BOOL) {
    unsigned vectorWidth = dataPathWidth / sizeof(bool);
    // ld64/ xor(and), ld64st64
    return basicOpLoopCycles(5, numElems, vectorWidth, boolInputComputeCycles);
  }
  assert(0 && "Bool size not supported");
  return 0;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledAdd)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &type) {
  CODELET_FIELD(deltas);
  uint64_t cycles = 5;
  const auto data = vertex.getFieldInfo("data");
  assert(data.size() == deltas.size());
  for (unsigned i = 0; i < data.size(); ++i) {
    unsigned numElem = data[i].size();
    assert(data[i].size() == deltas[i].size());
    unsigned vectorWidth = 1;
    unsigned cyclesPerVector = 1;
    if (type == FLOAT) {
      vectorWidth = target.getDataPathWidth() / 32;
    }
    else if (type == HALF) {
      vectorWidth = target.getDataPathWidth() / 16;
    }
    else {// integer types are not vectorisable
      cyclesPerVector = 4; //ld/mpy/add/st
      vectorWidth = 1;
    }
    // Inner loop uses the axpy instruction.
    cycles += 5 + cyclesPerVector *
        (1 + (numElem + vectorWidth - 1) / vectorWidth);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(HadamardProd)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &type) {
  uint64_t cycles = 5;
  const auto A = vertex.getFieldInfo("A");
  CODELET_FIELD(B);
  assert(A.size() == B.size());
  for (unsigned i = 0; i < A.size(); ++i) {
    assert(A[i].size() == B[i].size());
    unsigned numElem = A[i].size();
    bool isFloat = type == FLOAT;
    unsigned vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
    unsigned numVectors = (numElem + vectorWidth - 1) / vectorWidth;
    cycles += 5 + (1 + numVectors * 2);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Zero)(const VertexIntrospector &vertex,
                                const Target &target,
                                const Type &type) {
  // TODO: make this more accurate
  const auto out = vertex.getFieldInfo("out");
  bool isFloat = type == FLOAT;
  const auto vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
  std::uint64_t cycles = 2 // run
                         + 5; // vertex overhead
  for (unsigned i=0; i<out.size(); ++i) {
    auto zeroCycles = (out[i].size() + vectorWidth - 1) / vectorWidth;
    auto const loopOverhead = 3;
    cycles += loopOverhead + zeroCycles;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Zero2d)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &type) {
  const auto dst = vertex.getFieldInfo("out");
  // These are not valid for integer and boolean casts
  const auto floatVectorWidth = target.getDataPathWidth() / 32;
  return (dst.size() + floatVectorWidth - 1) / floatVectorWidth + 5;
}

// TODO: popops::Cast* cycle estimators do not depend on template type
// of the codelet. (a) This may change. (b) It will introduce an annoying
// special case at estimator registration time as we can't automatically
// lookup based on the template name. (c) INSTANTIATE_TEMPLATE_CYCLE_ESTIMATOR
// doesn't handle funcs with more than one template parameter.
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Cast)(const VertexIntrospector &vertex,
                                const Target &target,
                                const Type &fromType,
                                const Type &toType) {
  // These are not valid for integer and boolean casts
  const auto dst = vertex.getFieldInfo("dst");
  const auto floatVectorWidth = target.getDataPathWidth() / 32;
  return (dst.size() + floatVectorWidth - 1) / floatVectorWidth + 5;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Cast2d)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &fromType,
                                  const Type &toType) {
  const auto floatVectorWidth = target.getDataPathWidth() / 32;
  std::uint64_t cycles = 5;
  const auto dst = vertex.getFieldInfo("dst");
  CODELET_FIELD(src);
  assert(src.size() == dst.size());
  for (unsigned i = 0; i != dst.size(); ++i) {
    assert(src[i].size() == dst[i].size());
    // Estimate based on 6 cycles of loop overhead per src / dst pointer pair:
    //
    // 1: load src
    // 2: load dst
    // 3: load length
    // 4: load src[0]
    // 5: { load src[1] ; convert src[0] }
    // 6: repeat
    // These are not valid for integer and boolean casts
    cycles += 6 + (dst[i].size() + floatVectorWidth - 1) / floatVectorWidth;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Absolute)(const VertexIntrospector &vertex,
                                    const Target &target,
                                    const Type &type) {
  uint64_t cycles = 5;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i < in.size(); ++i) {
    assert (in[i].size() == out[i].size());
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;

    if (type == FLOAT) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (type == HALF) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (type == INT) {
      // ld, abs, st
      cyclesPerVector = 3;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Add)(const VertexIntrospector &vertex,
                                const Target &target,
                                const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;
    if (type == FLOAT) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (type == HALF) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (type == INT) {
      // ld, ld, add, st
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Atan2)(const VertexIntrospector &vertex,
                                 const Target &target,
                                 const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;
    if (type == FLOAT) {
      vectorWidth = 1;
      cyclesPerVector = 25;
    } else if (type == HALF) {
      vectorWidth = 1;
      cyclesPerVector = 25 + 3;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
            cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BitwiseAnd)(const VertexIntrospector &vertex,
                                      const Target &target,
                                      const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    unsigned numElem = in1[i].size();
    unsigned overhead = 6;
    unsigned vectorWidth = 2;
    unsigned cyclesPerVector = 1;

    // AND in parallel with ld2xstpace
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BitwiseNot)(const VertexIntrospector &vertex,
                                       const Target &target,
                                       const Type &type) {
  uint64_t cycles = 7;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i != in.size(); ++i) {
    assert (in[i].size() == out[i].size());
    unsigned numElem = in[i].size();
    unsigned overhead = 6;
    unsigned vectorWidth = 2;
    unsigned cyclesPerVector = 1;

    // NOT on AUX side, ldst64pace
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(BitwiseOr)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    unsigned numElem = in1[i].size();
    unsigned overhead = 6;
    unsigned vectorWidth = 2;
    unsigned cyclesPerVector = 1;

    // OR on AUX side, ld2xstpace
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Ceil)(const VertexIntrospector &vertex,
                                const Target &target,
                                const Type &type) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i != in.size(); ++i) {
    assert (in[i].size() == out[i].size());
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    bool isFloat = type == FLOAT;
    // use mul with 1.0 and use correct rounding mode
    unsigned cyclesPerVector = 1;
    unsigned vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Cos)(const VertexIntrospector &vertex,
                               const Target &target,
                               const Type &type) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i != in.size(); ++i) {
    assert (in[i].size() == out[i].size());
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;
    unsigned cyclesPerVector = 1;
    if (type == FLOAT) {
      vectorWidth = 1;
      cyclesPerVector = 150;
    } else if (type == HALF) {
      vectorWidth = 1;
      cyclesPerVector = 100;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Divide)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;
    if (type == FLOAT) {
      cyclesPerVector = 1;
    } else if (type == HALF) {
      // Convert to f32 using v2 and divide and convert back to f16
      vectorWidth = 2;
      cyclesPerVector = 4;
    } else if (type == INT) {
      // ld into aux, ld into aux, div, st
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Equal)(const VertexIntrospector &vertex,
                                 const Target &target,
                                 const Type &type) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    // E = A and B, F = A or B, G = F andc E, result = 1 andc G
    const auto numBoolOpCycles = type == BOOL ? 4 : 0;
    cycles += comparisonOpsCycles(target.getDataPathWidth(),
                                  in1.size(),
                                  numBoolOpCycles,
                                  type);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Exponent)(const VertexIntrospector &vertex,
                                    const Target &target,
                                    const Type &type) {
  uint64_t cycles = 5;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i != in.size(); ++i) {
    assert (in[i].size() == out[i].size());
    unsigned numElem = in[i].size();
    bool isFloat = type == FLOAT;
    unsigned vectorWidth = 1;
    unsigned cyclesPerVector = 3;
    unsigned overhead = 6;

    if(!isFloat) {
      // Use f16v2exp
      vectorWidth = 2;
      cyclesPerVector = 2;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Floor)(const VertexIntrospector &vertex,
                                const Target &target,
                                const Type &type) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i != in.size(); ++i) {
    assert (in[i].size() == out[i].size());
    const unsigned overhead = 6;
    unsigned numElem = in[i].size();
    bool isFloat = type == FLOAT;

    // Use mul with 1.0 and use correct rounding mode
    unsigned vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
    unsigned cyclesPerVector = 1;
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(GreaterThan)(const VertexIntrospector &vertex,
                                       const Target &target,
                                       const Type &type) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    // same as B < A
    // E = A and B, result = A andc E
    const auto numBoolOpCycles = type == BOOL ? 2 : 0;
    cycles += comparisonOpsCycles(target.getDataPathWidth(),
                                  in1.size(),
                                  numBoolOpCycles,
                                  type);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(GreaterThanEqual)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            const Type &type) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    // same as B <= A
    // E = 1 andc B, result = E or A
    const auto numBoolOpCycles = type == BOOL ? 2 : 0;
    cycles += comparisonOpsCycles(target.getDataPathWidth(),
                                  in1.size(),
                                  numBoolOpCycles,
                                  type);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(IsFinite)(const VertexIntrospector &vertex,
                                    const Target &target,
                                    const Type &type) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i != in.size(); ++i) {
    assert (in[i].size() == out[i].size());
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    bool isFloat = type == FLOAT;
    unsigned vectorWidth = 2;

    // 1 for v==v
    // 1 for v!=INFINITY
    // 1 for anding the two together
    // 1 for converting a match from 0xffff to 0x0001
    // 1 to convert the 32/16bit individual results to 8bits each
    unsigned cyclesPerVector = 5;
    if (!isFloat) {
      vectorWidth = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(LessThan)(const VertexIntrospector &vertex,
                                    const Target &target,
                                    const Type &type) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());    // E = A and B, result = B andc E
    const auto numBoolOpCycles = type == BOOL ? 2 : 0;
    cycles += comparisonOpsCycles(target.getDataPathWidth(),
                                  in1.size(),
                                  numBoolOpCycles,
                                  type);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(LessThanEqual)(const VertexIntrospector &vertex,
                                         const Target &target,
                                         const Type &type) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());    // E = 1 andc A, result = E or B
    const auto numBoolOpCycles = type == BOOL ? 2 : 0;
    cycles += comparisonOpsCycles(target.getDataPathWidth(),
                                  in1.size(),
                                  numBoolOpCycles,
                                  type);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Logarithm)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &type) {
  uint64_t cycles = 5;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i < in.size(); ++i) {
    assert(in[i].size() == out[i].size());
    bool isFloat = type == FLOAT;
    unsigned cyclesPerVector = 6;
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;

    if(!isFloat) {
      // used f16v2 variant
      cyclesPerVector = 2;
      vectorWidth = 2;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(LogicalAnd)(const VertexIntrospector &vertex,
                                      const Target &target,
                                      const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    unsigned numElem = in1[i].size();
    unsigned overhead = 6;
    unsigned vectorWidth = target.getDataPathWidth() / sizeof(bool);
    unsigned cyclesPerVector = 2;

    // Use AND on AUX side
    // Assume ld2xst64 cannot be used
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(LogicalNot)(const VertexIntrospector &vertex,
                                      const Target &target,
                                      const Type &type) {
  uint64_t cycles = 7;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i < in.size(); ++i) {
    assert(in[i].size() == out[i].size());
    unsigned numElem = in[i].size();
    unsigned overhead = 6;
    unsigned vectorWidth = target.getDataPathWidth() / sizeof(bool);
    unsigned cyclesPerVector = 1;

    // XOR on aux side
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(LogicalOr)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    unsigned numElem = in1[i].size();
    unsigned overhead = 6;
    unsigned vectorWidth = target.getDataPathWidth() / sizeof(bool);
    unsigned cyclesPerVector = 2;

    // OR on the aux side
    // Assume ld2xst64 cannot be used
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Maximum)(const VertexIntrospector &vertex,
                                   const Target &target,
                                   const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;

    if (type == FLOAT) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (type == HALF) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (type == INT) {
      // ld, ld, max, st
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Minimum)(const VertexIntrospector &vertex,
                                   const Target &target,
                                   const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;

    if (type == FLOAT) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (type == HALF) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (type == INT) {
      // ld, ld, min, st
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Multiply)(const VertexIntrospector &vertex,
                                    const Target &target,
                                    const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;
    if (type == FLOAT) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (type == HALF) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (type == INT) {
      // ld, ld, mul, st
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(NotEqual)(const VertexIntrospector &vertex,
                                    const Target &target,
                                    const Type &type) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    // E = A and B, F = A or B, result = F andc E
    const auto numBoolOpCycles = type == BOOL ? 3 : 0;
    cycles += comparisonOpsCycles(target.getDataPathWidth(),
                                  in1.size(),
                                  numBoolOpCycles,
                                  type);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Negate)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &type) {
      uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i < in.size(); ++i) {
    assert(in[i].size() == out[i].size());
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;
    if (type == FLOAT) {
      vectorWidth = target.getDataPathWidth() / 32;
    } else if (type == HALF) {
      vectorWidth = target.getDataPathWidth() / 16;
    } else if (type == INT) {
      // ld, sub, st
      cyclesPerVector = 3;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Power)(const VertexIntrospector &vertex,
                                 const Target &target,
                                 const Type &type) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    bool isFloat = type == FLOAT;
    unsigned vectorWidth = 1;
    unsigned cyclesPerVector = 100;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();

    // This cycles are wrong
    // Accuracy concerns using ln
    // pow(a,b) = exp(b * log(a))
    // Doesn't handle negative values yet
    if(!isFloat) {
      // used f16v4 variant: Accuracy converns using half precision log
      vectorWidth = target.getDataPathWidth() / 16;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Remainder)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;

    if (type == FLOAT) {
      vectorWidth = target.getDataPathWidth() / 32;
    } else if (type == HALF) {
      // Convert to f32 using v2 and divide and convert back to f16
      vectorWidth = 2;
      cyclesPerVector = 4;
    } else if (type == INT) {
      // load on aux side, mod and store result from aux
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Round)(const VertexIntrospector &vertex,
                                 const Target &target,
                                 const Type &type) {
  uint64_t cycles = 5;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i < in.size(); ++i) {
    assert(in[i].size() == out[i].size());
    unsigned cyclesPerVector = 2;
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ShiftLeft)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    unsigned numElem = in1[i].size();
    unsigned overhead = 6;
    unsigned vectorWidth = 1;
    unsigned cyclesPerVector = 3;

    // OR on AUX side, ld2xstpace
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ShiftRight)(const VertexIntrospector &vertex,
                                      const Target &target,
                                      const Type &type) {
  return MAKE_CYCLE_ESTIMATOR_NAME(ShiftLeft)(vertex, target, type);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ShiftRightSignExtend)(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &type) {
  return MAKE_CYCLE_ESTIMATOR_NAME(ShiftLeft)(vertex, target, type);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Signum)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &type) {
  // extra cycles to form constants
  uint64_t cycles = 7;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i < in.size(); ++i) {
    assert(in[i].size() == out[i].size());
    unsigned overhead = 6;
    unsigned numElem = in[i].size();

    // default value for int:
    // ld32 in
    // cmpslt a, mzero, in
    // cmpslt b, in, mzero
    // sub c, a, b
    // st32 c
    unsigned cyclesPerVector = 5;
    unsigned vectorWidth = 1;

    if (type != INT) {
      bool isFloat = type == FLOAT;
      vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
      // For float and half:
      // use clamp (f16v4 or f32v2)
      cyclesPerVector = 1;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Sin)(const VertexIntrospector &vertex,
                               const Target &target,
                               const Type &type) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i < in.size(); ++i) {
    assert(in[i].size() == out[i].size());
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;
    unsigned cyclesPerVector = 1;
    if (type == FLOAT) {
      vectorWidth = 1;
      cyclesPerVector = 150;
    } else if (type == HALF) {
      vectorWidth = 1;
      cyclesPerVector = 100;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Subtract)(const VertexIntrospector &vertex,
                                    const Target &target,
                                    const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;
    if (type == FLOAT) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (type == HALF) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (type == INT) {
      // ld, ld, sub, st
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Tanh)(const VertexIntrospector &vertex,
                                const Target &target,
                                const Type &type) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i < in.size(); ++i) {
    assert(in[i].size() == out[i].size());
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;
    unsigned cyclesPerVector = 1;
    if (type == FLOAT) {
      // f32tanh
      vectorWidth = 1;
      cyclesPerVector = 7;
    } else if (type == HALF) {
      // f16v2tanh
      vectorWidth = 2;
      cyclesPerVector = 2;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Sqrt)(const VertexIntrospector &vertex,
                                const Target &target,
                                const Type &type) {
  uint64_t cycles = 5;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i < in.size(); ++i) {
    assert(in[i].size() == out[i].size());
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;
    if (type == FLOAT) {
      cyclesPerVector = 5;
    } else if (type == HALF) {
      // f32sqrt + conversions f16<->f32
      cyclesPerVector = 7;
    } else if (type == INT) {
      // ld, mul, st
      cyclesPerVector = 10; // placeholder
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Square)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &type) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  CODELET_FIELD(out);
  assert(in.size() == out.size());
  for (unsigned i = 0; i < in.size(); ++i) {
    assert(in[i].size() == out[i].size());
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    bool isFloat = type == FLOAT;
    unsigned vectorWidth = 1;
    unsigned cyclesPerVector = 1;
    if (!isFloat) {
      vectorWidth = target.getDataPathWidth() / 16;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Select)(const VertexIntrospector &vertex,
                                  const Target &target,
                                  const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  assert(in3.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    assert(in3[i].size() == in1[i].size());
    unsigned cyclesPerVector = 5;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;
    // ld in1, ld in2, ld in3, movz, st
    // it may be possible to load on the Aux side but then would
    // depend on bool size. If Aux side is used masks must be created after
    // expanding bools to match the input datum size
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Clamp)(const VertexIntrospector &vertex,
                                 const Target &target,
                                 const Type &type) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  CODELET_FIELD(in2);
  CODELET_FIELD(in3);
  CODELET_FIELD(out);
  assert(in1.size() == out.size());
  assert(in2.size() == in1.size());
  assert(in3.size() == in1.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    assert(in1[i].size() == out[i].size());
    assert(in2[i].size() == in1[i].size());
    assert(in3[i].size() == in1[i].size());
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;
    if (type == FLOAT) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (type == HALF) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (type == INT) {
      // ld, ld, ld, cmp, movz, cmp, st
      cyclesPerVector = 7;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(DynamicSlice)(const VertexIntrospector &vertex,
                                         const Target &target,
                                         const Type &type) {
  const auto baseT = vertex.getFieldInfo("baseT");
  const unsigned numBaseElements =
    vertex.getFieldInfo("numBaseElements").getInitialValue<unsigned>(target);
  const unsigned numSubElements =
    vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);

  unsigned vectorWidth = target.getDataPathWidth() / (sizeof(type) * 8);
  auto numRegions = baseT.size() / numBaseElements;
  auto cycles = 5;
  for (unsigned r = 0; r != numRegions; ++r) {
    auto regionSize = baseT[r * numBaseElements].size();
    unsigned nVectors = (regionSize + vectorWidth - 1) / vectorWidth;
    cycles += (4 + nVectors) * numSubElements + 4;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(DynamicUpdateSlice)(const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &type) {
  const auto baseT = vertex.getFieldInfo("baseT");
  const unsigned numBaseElements =
    vertex.getFieldInfo("numBaseElements").getInitialValue<unsigned>(target);
  const unsigned numSubElements =
    vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);

  unsigned vectorWidth = target.getDataPathWidth() / (sizeof(type) * 8);
  auto numRegions = baseT.size() / numBaseElements;
  auto cycles = 5;
  for (unsigned r = 0; r != numRegions; ++r) {
    auto regionSize = baseT[r * numBaseElements].size();
    unsigned nVectors = (regionSize + vectorWidth - 1) / vectorWidth;
    cycles += (4 + nVectors) * numSubElements + 4;
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(DynamicSlice2d)(const VertexIntrospector &vertex,
                                           const Target &target,
                                           const Type &type) {
  unsigned vectorWidth = target.getDataPathWidth() / (sizeof(type) * 8);
  const unsigned numSubElements =
    vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);
  const unsigned elementsPerWorker =
    vertex.getFieldInfo("elementsPerWorker").getInitialValue<unsigned>(target);
  const unsigned numWorkers = target.getNumWorkerContexts();

  auto cycles = 5;
  unsigned nVectors = (elementsPerWorker + vectorWidth - 1) / vectorWidth;
  cycles += (4 + nVectors) * numSubElements + 4;
  cycles *= numWorkers;
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(DynamicUpdateSlice2d)(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &type) {
  return MAKE_CYCLE_ESTIMATOR_NAME(DynamicSlice2d)(vertex, target, type);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(AllTrue)(const VertexIntrospector &vertex,
                                   const Target &target) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
    unsigned cyclesPerVector = 1;
    unsigned overhead = 11;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = target.getDataPathWidth() / (sizeof(bool) * 8);
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CircBufIncrIndex)(const VertexIntrospector &vertex,
                                            const Target &target) {
  return 8;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(CircOffset)(const VertexIntrospector &vertex,
                                      const Target &target) {
  return 10;
}


poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  return {
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceOr, BOOL, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, ReduceAnd, BOOL, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, ReduceMin, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceMin, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceMin, INT, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, ReduceMax, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceMax, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceMax, INT, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, ReduceMul, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceMul, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceMul, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceMul, INT, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, ReduceAddScale, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceAddScale, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceAddScale, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceAddScale, INT, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, ReduceAddUpdate, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceAddUpdate, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceAddUpdate, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceAddUpdate, INT, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, ReduceAdd, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceAdd, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceAdd, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, ReduceAdd, INT, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd, UNSIGNED_INT),
    CYCLE_ESTIMATOR_ENTRY(popops, ScaledAdd, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, HadamardProd, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, HadamardProd, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, Zero, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Zero, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Zero, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Zero, UNSIGNED_INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Zero2d, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Zero2d, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, FLOAT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, FLOAT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, FLOAT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, FLOAT, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, HALF, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, HALF, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast, INT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, INT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, INT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, INT, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast, BOOL, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, BOOL, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, BOOL, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast, BOOL, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, FLOAT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, FLOAT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, FLOAT, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, HALF, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, HALF, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, INT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, INT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, INT, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, INT, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, BOOL, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, BOOL, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, BOOL, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cast2d, BOOL, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Absolute, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Absolute, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Absolute, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Add, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Add, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Add, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Add, UNSIGNED_INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Atan2, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Atan2, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, BitwiseAnd, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, BitwiseNot, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, BitwiseOr, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Ceil, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Ceil, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, Cos, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Cos, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, Divide, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Divide, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Divide, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Equal, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Equal, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Equal, BOOL),
    CYCLE_ESTIMATOR_ENTRY(popops, Equal, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Exponent, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Exponent, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, Floor, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Floor, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, GreaterThan, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, GreaterThan, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, GreaterThan, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, GreaterThan, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, GreaterThanEqual, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, GreaterThanEqual, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, GreaterThanEqual, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, GreaterThanEqual, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, IsFinite, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, IsFinite, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, LessThan, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, LessThan, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, LessThan, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, LessThan, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, LessThanEqual, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, LessThanEqual, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, LessThanEqual, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, LessThanEqual, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Logarithm, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Logarithm, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, LogicalAnd, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, LogicalNot, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, LogicalOr, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Maximum, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Maximum, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Maximum, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Minimum, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Minimum, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Minimum, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Multiply, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Multiply, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Multiply, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, NotEqual, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, NotEqual, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, NotEqual, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, NotEqual, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Negate, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Negate, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Negate, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Power, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Power, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, Remainder, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Remainder, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Remainder, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Round, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Round, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, ShiftLeft, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, ShiftRight, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, ShiftRightSignExtend, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Signum, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Signum, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Signum, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Sin, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Sin, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, Subtract, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Subtract, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Subtract, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Subtract, UNSIGNED_INT),

    CYCLE_ESTIMATOR_ENTRY(popops, Tanh, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Tanh, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, Sqrt, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Sqrt, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, Square, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Square, HALF),

    CYCLE_ESTIMATOR_ENTRY(popops, Select, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Select, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Select, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, Select, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, Clamp, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, Clamp, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, Clamp, INT),

    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2d, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2d, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2d, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicSlice2d, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2d, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2d, HALF),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2d, INT),
    CYCLE_ESTIMATOR_ENTRY(popops, DynamicUpdateSlice2d, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popops, AllTrue),
    CYCLE_ESTIMATOR_ENTRY(popops, CircBufIncrIndex),
    CYCLE_ESTIMATOR_ENTRY(popops, CircOffset)

  };
};

} // end namespace popreduce
