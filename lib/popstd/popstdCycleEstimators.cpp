#include <poplar/HalfFloat.hpp>
#include <popstdCycleEstimators.hpp>

using namespace poplar;

namespace popstd {

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
template<typename InType>
static uint64_t comparisonOpsCycles(unsigned dataPathWidth,
                                    unsigned numElems,
                                    unsigned boolInputComputeCycles) {
  if (std::is_same<InType, float>::value) {
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
  } else if (std::is_same<InType, half>::value) {
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
  } else if (std::is_same<InType, int>::value) {
    if (sizeof(bool) == 4) {
      return basicOpLoopCycles(5, numElems, 1, 4);
    } else if (sizeof(bool) == 2) {
      // (ld32, ld32, cmp) * 2, sort16, sort16, st32
      return basicOpLoopCycles(5, numElems, 2, 9);
    } else if (sizeof(bool) == 1) {
      // (ld32, ld32, cmp) * 4, sort16, sort16, sort8, st32
      return basicOpLoopCycles(5, numElems, 4, 16);
    }
  } else if (std::is_same<InType, bool>::value) {
    unsigned vectorWidth = dataPathWidth / sizeof(bool);
    // ld64/ xor(and), ld64st64
    return basicOpLoopCycles(5, numElems, vectorWidth, boolInputComputeCycles);
  }
  assert(0 && "Bool size not supported");
  return 0;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(ScaledAdd, vertex, target) {
    uint64_t cycles = 5;
    const auto data = vertex.getFieldInfo("data");
    for (unsigned i = 0; i < data.size(); ++i) {
      unsigned numElem = data[i].size();
      unsigned vectorWidth = 1;
      unsigned cyclesPerVector = 1;
      if (std::is_same<InType, float>::value) {
        vectorWidth = target.getDataPathWidth() / 32;
      }
      else if (std::is_same<InType, half>::value) {
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

template <class FPType>
MAKE_CYCLE_ESTIMATOR(HadamardProd, vertex, target) {
  uint64_t cycles = 5;
  const auto A = vertex.getFieldInfo("A");
  for (unsigned i = 0; i < A.size(); ++i) {
    unsigned numElem = A[i].size();
    bool isFloat = std::is_same<FPType, float>::value;
    unsigned vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
    unsigned numVectors = (numElem + vectorWidth - 1) / vectorWidth;
    cycles += 5 + (1 + numVectors * 2);
  }
  return cycles;
}

template <class FPType>
MAKE_CYCLE_ESTIMATOR(Zero, vertex, target) {
  // TODO: make this more accurate
  const auto out = vertex.getFieldInfo("out");
  bool isFloat = std::is_same<FPType, float>::value;
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

MAKE_CYCLE_ESTIMATOR(Zero2d, vertex, target) {
  const auto dst = vertex.getFieldInfo("out");
  // These are not valid for integer and boolean casts
  const auto floatVectorWidth = target.getDataPathWidth() / 32;
  return (dst.size() + floatVectorWidth - 1) / floatVectorWidth + 5;
}

// TODO: popstd::Cast* cycle estimators do not depend on template type
// of the codelet. (a) This may change. (b) It will introduce an annoying
// special case at estimator registration time as we can't automatically
// lookup based on the template name. (c) INSTANTIATE_TEMPLATE_CYCLE_ESTIMATOR
// doesn't handle funcs with more than one template parameter.
MAKE_CYCLE_ESTIMATOR(Cast, vertex, target) {
  // These are not valid for integer and boolean casts
  const auto dst = vertex.getFieldInfo("dst");
  const auto floatVectorWidth = target.getDataPathWidth() / 32;
  return (dst.size() + floatVectorWidth - 1) / floatVectorWidth + 5;
}

MAKE_CYCLE_ESTIMATOR(Cast2d, vertex, target) {
  const auto floatVectorWidth = target.getDataPathWidth() / 32;
  std::uint64_t cycles = 5;
  const auto dst = vertex.getFieldInfo("dst");
  for (unsigned i = 0; i != dst.size(); ++i) {
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(Absolute, vertex, target) {
  uint64_t cycles = 5;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;

    if (std::is_same<InType, float>::value) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, half>::value) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, int>::value) {
      // ld, abs, st
      cyclesPerVector = 3;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Add, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;
    if (std::is_same<InType, float>::value) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, half>::value) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, int>::value) {
      // ld, ld, add, st
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(BitwiseAnd, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(BitwiseNot, vertex, target) {
  uint64_t cycles = 7;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(BitwiseOr, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(Ceil, vertex, target) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    bool isFloat = std::is_same<InType, float>::value;
    // use mul with 1.0 and use correct rounding mode
    unsigned cyclesPerVector = 1;
    unsigned vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Cos, vertex, target) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;
    unsigned cyclesPerVector = 1;
    if (std::is_same<InType, float>::value) {
      vectorWidth = 1;
      cyclesPerVector = 150;
    } else if (std::is_same<InType, half>::value) {
      vectorWidth = 1;
      cyclesPerVector = 100;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Divide, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;
    if (std::is_same<InType, float>::value) {
      cyclesPerVector = 1;
    } else if (std::is_same<InType, half>::value) {
      // Convert to f32 using v2 and divide and convert back to f16
      vectorWidth = 2;
      cyclesPerVector = 4;
    } else if (std::is_same<InType, int>::value) {
      // ld into aux, ld into aux, div, st
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Equal, vertex, target) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    // E = A and B, F = A or B, G = F andc E, result = 1 andc G
    const auto numBoolOpCycles = std::is_same<InType, bool>::value ? 4 : 0;
    cycles += comparisonOpsCycles<InType>(target.getDataPathWidth(),
                                          in1.size(),
                                          numBoolOpCycles);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Exponent, vertex, target) {
  uint64_t cycles = 5;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
    unsigned numElem = in[i].size();
    bool isFloat = std::is_same<InType, float>::value;
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(Floor, vertex, target) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
    const unsigned overhead = 6;
    unsigned numElem = in[i].size();
    bool isFloat = std::is_same<InType, float>::value;

    // Use mul with 1.0 and use correct rounding mode
    unsigned vectorWidth = target.getDataPathWidth() / (isFloat ? 32 : 16);
    unsigned cyclesPerVector = 1;
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(GreaterThan, vertex, target) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    // same as B < A
    // E = A and B, result = A andc E
    const auto numBoolOpCycles = std::is_same<InType, bool>::value ? 2 : 0;
    cycles += comparisonOpsCycles<InType>(target.getDataPathWidth(),
                                          in1.size(),
                                          numBoolOpCycles);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(GreaterThanEqual, vertex, target) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    // same as B <= A
    // E = 1 andc B, result = E or A
    const auto numBoolOpCycles = std::is_same<InType, bool>::value ? 2 : 0;
    cycles += comparisonOpsCycles<InType>(target.getDataPathWidth(),
                                          in1.size(),
                                          numBoolOpCycles);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(IsFinite, vertex, target) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    bool isFloat = std::is_same<InType, float>::value;
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(LessThan, vertex, target) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    // E = A and B, result = B andc E
    const auto numBoolOpCycles = std::is_same<InType, bool>::value ? 2 : 0;
    cycles += comparisonOpsCycles<InType>(target.getDataPathWidth(),
                                          in1.size(),
                                          numBoolOpCycles);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(LessThanEqual, vertex, target) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    // E = 1 andc A, result = E or B
    const auto numBoolOpCycles = std::is_same<InType, bool>::value ? 2 : 0;
    cycles += comparisonOpsCycles<InType>(target.getDataPathWidth(),
                                          in1.size(),
                                          numBoolOpCycles);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Logarithm, vertex, target) {
  uint64_t cycles = 5;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
    bool isFloat = std::is_same<InType, float>::value;
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(LogicalAnd, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(LogicalNot, vertex, target) {
  uint64_t cycles = 7;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(LogicalOr, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(Maximum, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;

    if (std::is_same<InType, float>::value) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, half>::value) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, int>::value) {
      // ld, ld, max, st
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Minimum, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {

    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;

    if (std::is_same<InType, float>::value) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, half>::value) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, int>::value) {
      // ld, ld, min, st
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Multiply, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;
    if (std::is_same<InType, float>::value) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, half>::value) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, int>::value) {
      // ld, ld, mul, st
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(NotEqual, vertex, target) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    // E = A and B, F = A or B, result = F andc E
    const auto numBoolOpCycles = std::is_same<InType, bool>::value ? 3 : 0;
    cycles += comparisonOpsCycles<InType>(target.getDataPathWidth(),
                                          in1.size(),
                                          numBoolOpCycles);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Negate, vertex, target) {
      uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {

    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;
    if (std::is_same<InType, float>::value) {
      vectorWidth = target.getDataPathWidth() / 32;
    } else if (std::is_same<InType, half>::value) {
      vectorWidth = target.getDataPathWidth() / 16;
    } else if (std::is_same<InType, int>::value) {
      // ld, sub, st
      cyclesPerVector = 3;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Power, vertex, target) {
  uint64_t cycles = 7;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    bool isFloat = std::is_same<InType, float>::value;
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(Remainder, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;

    if (std::is_same<InType, float>::value) {
      vectorWidth = target.getDataPathWidth() / 32;
    } else if (std::is_same<InType, half>::value) {
      // Convert to f32 using v2 and divide and convert back to f16
      vectorWidth = 2;
      cyclesPerVector = 4;
    } else if (std::is_same<InType, int>::value) {
      // load on aux side, mod and store result from aux
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Round, vertex, target) {
  uint64_t cycles = 5;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
    unsigned cyclesPerVector = 2;
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(ShiftLeft, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(ShiftRight, vertex, target) {
  return MAKE_CYCLE_ESTIMATOR_NAME(ShiftLeft)<InType>(vertex,target);
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(ShiftRightSignExtend, vertex, target) {
  return MAKE_CYCLE_ESTIMATOR_NAME(ShiftLeft)<InType>(vertex,target);
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Signum, vertex, target) {
  // extra cycles to form constants
  uint64_t cycles = 7;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
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

    if (!std::is_same<InType, int>::value) {
      bool isFloat = std::is_same<InType, float>::value;
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(Sin, vertex, target) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;
    unsigned cyclesPerVector = 1;
    if (std::is_same<InType, float>::value) {
      vectorWidth = 1;
      cyclesPerVector = 150;
    } else if (std::is_same<InType, half>::value) {
      vectorWidth = 1;
      cyclesPerVector = 100;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Subtract, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;
    if (std::is_same<InType, float>::value) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, half>::value) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, int>::value) {
      // ld, ld, sub, st
      cyclesPerVector = 4;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Tanh, vertex, target) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth;
    unsigned cyclesPerVector;
    if (std::is_same<InType, float>::value) {
      // f32tanh
      vectorWidth = 1;
      cyclesPerVector = 7;
    } else if (std::is_same<InType, half>::value) {
      // f16v2tanh
      vectorWidth = 2;
      cyclesPerVector = 2;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Sqrt, vertex, target) {
  uint64_t cycles = 5;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    unsigned vectorWidth = 1;
    if (std::is_same<InType, float>::value) {
      cyclesPerVector = 5;
    } else if (std::is_same<InType, half>::value) {
      // f32sqrt + conversions f16<->f32
      cyclesPerVector = 7;
    } else if (std::is_same<InType, int>::value) {
      // ld, mul, st
      cyclesPerVector = 10; // placeholder
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(Square, vertex, target) {
  uint64_t cycles = 6;
  const auto in = vertex.getFieldInfo("in");
  for (unsigned i = 0; i < in.size(); ++i) {
    unsigned overhead = 6;
    unsigned numElem = in[i].size();
    bool isFloat = std::is_same<InType, float>::value;
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(Select, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
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

template <class InType>
MAKE_CYCLE_ESTIMATOR(Clamp, vertex, target) {
  uint64_t cycles = 5;
  const auto in1 = vertex.getFieldInfo("in1");
  for (unsigned i = 0; i < in1.size(); ++i) {
    unsigned cyclesPerVector = 1;
    unsigned overhead = 6;
    unsigned numElem = in1[i].size();
    unsigned vectorWidth = 1;
    if (std::is_same<InType, float>::value) {
      vectorWidth = target.getDataPathWidth() / 32;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, half>::value) {
      vectorWidth = target.getDataPathWidth() / 16;
      cyclesPerVector = 2;
    } else if (std::is_same<InType, int>::value) {
      // ld, ld, ld, cmp, movz, cmp, st
      cyclesPerVector = 7;
    }
    cycles += basicOpLoopCycles(overhead, numElem, vectorWidth,
                                cyclesPerVector);
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(DynamicSelect, vertex, target) {
  const auto baseT = vertex.getFieldInfo("baseT");
  const unsigned numBaseElements =
    vertex.getFieldInfo("numBaseElements").getInitialValue<unsigned>(target);
  const unsigned numSubElements =
    vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);

  unsigned vectorWidth = target.getDataPathWidth() / (sizeof(InType) * 8);
  auto numRegions = baseT.size() / numBaseElements;
  auto cycles = 5;
  for (unsigned r = 0; r != numRegions; ++r) {
    auto regionSize = baseT[r * numBaseElements].size();
    unsigned nVectors = (regionSize + vectorWidth - 1) / vectorWidth;
    cycles += (4 + nVectors) * numSubElements + 4;
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(DynamicUpdateSlice, vertex, target) {
  const auto baseT = vertex.getFieldInfo("baseT");
  const unsigned numBaseElements =
    vertex.getFieldInfo("numBaseElements").getInitialValue<unsigned>(target);
  const unsigned numSubElements =
    vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);

  unsigned vectorWidth = target.getDataPathWidth() / (sizeof(InType) * 8);
  auto numRegions = baseT.size() / numBaseElements;
  auto cycles = 5;
  for (unsigned r = 0; r != numRegions; ++r) {
    auto regionSize = baseT[r * numBaseElements].size();
    unsigned nVectors = (regionSize + vectorWidth - 1) / vectorWidth;
    cycles += (4 + nVectors) * numSubElements + 4;
  }
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(DynamicSelect2d, vertex, target) {
  unsigned vectorWidth = target.getDataPathWidth() / (sizeof(InType) * 8);
  const unsigned numSubElements =
    vertex.getFieldInfo("numSubElements").getInitialValue<unsigned>(target);
  const unsigned elementsPerWorker =
    vertex.getFieldInfo("elementsPerWorker").getInitialValue<unsigned>(target);
  const unsigned numWorkers =
    vertex.getFieldInfo("numWorkers").getInitialValue<unsigned>(target);

  auto cycles = 5;
  unsigned nVectors = (elementsPerWorker + vectorWidth - 1) / vectorWidth;
  cycles += (4 + nVectors) * numSubElements + 4;
  cycles *= numWorkers;
  return cycles;
}

template <class InType>
MAKE_CYCLE_ESTIMATOR(DynamicUpdateSlice2d, vertex, target) {
  return MAKE_CYCLE_ESTIMATOR_NAME(DynamicSelect2d)<InType>(vertex, target);
}

MAKE_CYCLE_ESTIMATOR(AllTrue, vertex, target) {
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

MAKE_CYCLE_ESTIMATOR(CircBufIncrIndex, vertex, target) {
  return 8;
}

MAKE_CYCLE_ESTIMATOR(CircOffset, vertex, target) {
  return 10;
}

poplibs::CycleEstimatorTable cyclesFunctionTable = {
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, ScaledAdd, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, ScaledAdd, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, ScaledAdd, unsigned int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, ScaledAdd, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, HadamardProd, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, HadamardProd, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Zero, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Zero, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Zero, int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Zero, unsigned int),

  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Zero2d, float),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Zero2d, half),

  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, float, float),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, float, half),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, float, int),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, float, int),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, float, bool),

  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, half, float),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, half, half),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, half, int),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, half, bool),

  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, int,float),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, int,half),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, int,int),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, int,bool),

  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, bool,float),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, bool,half),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, bool,int),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast, bool,bool),

  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, float, float),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, float, half),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, float, int),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, float, bool),

  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, half, float),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, half, half),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, half, int),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, half, bool),

  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, int,float),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, int,half),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, int,int),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, int,bool),

  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, bool,float),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, bool,half),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, bool,int),
  TYPED_CYCLE_ESTIMATOR_ENTRY(popstd, Cast2d, bool,bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Absolute, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Absolute, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Absolute, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Add, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Add, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Add, int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Add, unsigned int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, BitwiseAnd, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, BitwiseNot, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, BitwiseOr, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Ceil, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Ceil, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Cos, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Cos, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Divide, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Divide, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Divide, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Equal, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Equal, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Equal, bool),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Equal, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Exponent, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Exponent, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Floor, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Floor, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, GreaterThan, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, GreaterThan, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, GreaterThan, int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, GreaterThan, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, GreaterThanEqual, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, GreaterThanEqual, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, GreaterThanEqual, int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, GreaterThanEqual, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, IsFinite, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, IsFinite, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, LessThan, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, LessThan, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, LessThan, int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, LessThan, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, LessThanEqual, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, LessThanEqual, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, LessThanEqual, int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, LessThanEqual, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Logarithm, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Logarithm, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, LogicalAnd, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, LogicalNot, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, LogicalOr, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Maximum, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Maximum, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Maximum, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Minimum, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Minimum, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Minimum, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Multiply, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Multiply, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Multiply, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, NotEqual, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, NotEqual, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, NotEqual, int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, NotEqual, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Negate, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Negate, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Negate, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Power, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Power, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Remainder, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Remainder, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Remainder, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Round, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Round, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, ShiftLeft, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, ShiftRight, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, ShiftRightSignExtend, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Signum, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Signum, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Signum, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Sin, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Sin, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Subtract, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Subtract, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Subtract, int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Subtract, unsigned int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Tanh, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Tanh, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Sqrt, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Sqrt, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Square, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Square, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Select, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Select, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Select, int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Select, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Clamp, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Clamp, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, Clamp, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicSelect, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicSelect, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicSelect, int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicSelect, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicUpdateSlice, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicUpdateSlice, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicUpdateSlice, int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicUpdateSlice, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicSelect2d, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicSelect2d, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicSelect2d, int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicSelect2d, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicUpdateSlice2d, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicUpdateSlice2d, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicUpdateSlice2d, int),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popstd, DynamicUpdateSlice2d, bool),

  CYCLE_ESTIMATOR_ENTRY(popstd, AllTrue),
  CYCLE_ESTIMATOR_ENTRY(popstd, CircBufIncrIndex),
  CYCLE_ESTIMATOR_ENTRY(popstd, CircOffset)
};

} // end namespace popstd
