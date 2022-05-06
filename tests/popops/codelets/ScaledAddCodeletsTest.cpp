// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// Tests one or more of the ScaledAdd family of codelets:
//
//     VERTEX NAME:                   COMPUTES:
//
//     ScaledAddSupervisor            A + (b x B)
//     ScaledAdd2D
//
//     ScaledSubtractSupervisor       A - (b x B)
//     ScaledSubtract2D
//
//     aXPlusbYSupervisor            (a x A) + (b x B)
//     aXPlusbY2D
//
//     aXMinusbYSupervisor           (a x A) - (b x B)
//     aXMinusbY2D
//
//     XMinusaXPlusbYSupervisor       A - (a x A) - (b x B)
//     XMinusaXPlusbY2D
//
// Where 'A' and 'B' are tensors; 'a', 'b' are constants or scalar tensors.
//
// One or more combinations of operation/data type can be specified for the
// vertices under test.
//
// There is also an option to compare cycles reported when running the vertex
// on two different devices.
//
// See description string in main() for details and examples of usage.

#include "BinaryCodeletsTest.hpp"

#include <poplar/Engine.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include "../lib/popops/ExprOpUtil.hpp"
#include <poplibs_test/Util.hpp>
#include <poputil/TileMapping.hpp>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <optional>
#include <regex>
#include <type_traits>

using namespace poplar;
using namespace poputil;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace popops;
using namespace poplibs_support;

// All vertices that can be tested by this code
const std::vector<std::string> verticesNames = {
    "ScaledAddSupervisor", "ScaledAdd2D",        "ScaledSubtractSupervisor",
    "ScaledSubtract2D",    "aXPlusbYSupervisor", "aXPlusbY2D",
    "aXMinusbYSupervisor", "aXMinusbY2D",        "XMinusaXPlusbYSupervisor",
    "XMinusaXPlusbY2D",
};

//*************************************************************************
// This contains the name of the vertex under test and various flags that
// characterise it, used in different places during the test.
struct VertexDesc {
  std::string name;

  // Types for 'dataA' (i.e. 'X'), 'dataB' (i.e. 'Y') and 'scaleA' & 'scaleB'
  // These are template parameters for the vertices, but not all vertices
  // have all the three parameters.
  Type dataAType;
  Type dataBType;
  Type scaleType;

  // Template parameter for the vertex: are scale value(s) constants or tensors?
  // Not all vertices have this flag.
  std::optional<bool> scaleIsConstant;

  // Template parameter : Are A and B tensors constrained to be in different
  // elements?
  bool constrainedAB;

  std::string vClass;    // Full name with template params, for addVertex()
  std::string vClassFmt; // like vClass, but formatted for display

  bool is2D;
  bool isSubtract;
  bool hasScaleA;
  bool hasTolerance;    // Some vertices have a 'tolerance' field
  bool isSubtractFromX; // Is the vertex a 'XMinusaXPlusbY' variant?

  VertexDesc(const std::string &vertexName, const Type &dataAType,
             const Type &dataBType, const Type &scaleType,
             const std::optional<bool> scaleIsConstant,
             std::optional<bool> constrainedAB)
      : name(vertexName), dataAType(dataAType), dataBType(dataBType),
        scaleType(scaleType), scaleIsConstant(scaleIsConstant),
        constrainedAB(constrainedAB) {

    // Extract the flags by looking at the name
    is2D = name.find("2D") != std::string::npos;
    isSubtract = name.find("Subtract") != std::string::npos ||
                 name.find("Minus") != std::string::npos;
    isSubtractFromX =
        (name == "XMinusaXPlusbYSupervisor" || name == "XMinusaXPlusbY2D");

    hasScaleA =
        !(name == "ScaledAddSupervisor" || name == "ScaledAdd2D" ||
          name == "ScaledSubtractSupervisor" || name == "ScaledSubtract2D");

    hasTolerance = dataAType == HALF && dataBType == HALF && scaleType == FLOAT;

    // Build the vertex class string, using appropriate type
    // names & flags
    std::string vName = "popops::" + name;
    if (name == "ScaledAddSupervisor" || name == "ScaledAdd2D") {
      vClass = templateVertex(vName, dataAType, dataBType, scaleType,
                              *constrainedAB);
    } else if (name == "ScaledSubtractSupervisor") {
      vClass = templateVertex(vName, dataAType, dataBType, scaleType,
                              *constrainedAB);
    } else if (name == "ScaledSubtract2D") {
      vClass = templateVertex(vName, dataAType, scaleType, *constrainedAB);
    } else if (name == "aXPlusbYSupervisor" || name == "aXPlusbY2D" ||
               name == "aXMinusbYSupervisor" || name == "aXMinusbY2D") {
      vClass = templateVertex(vName, dataAType, scaleType, *constrainedAB);
    } else if (name == "XMinusaXPlusbYSupervisor" ||
               name == "XMinusaXPlusbY2D") {
      vClass = templateVertex(vName, dataAType, *constrainedAB);
    } else {
      throw std::runtime_error("Invalid vertex name (" + name + ")");
    }

    // Format the name for display, removing namespace
    vClassFmt = vClass;
    boost::erase_all(vClassFmt, "popops::");
    const unsigned FMT_LEN = 72;
    unsigned len = vClassFmt.size();
    unsigned padLen = (len < FMT_LEN) ? (FMT_LEN - len) : 1;
    vClassFmt += std::string(padLen, ' '); // pad for display
  }
};

//*************************************************************************
// Contains information relative to the test for one single vertex
struct TestRecord {
  SizeDesc size;
  std::shared_ptr<VertexDesc> vertex;

  // If not empty, offset in bytes in the device memory between the START of 1st
  // and the 2nd operand; if '0', it means place the two operands back-to-back.
  boost::optional<unsigned> operandOffset;

  TestOperand A;
  TestOperand B;
  // All these vertices are 'in-place', so the output is the same as the first
  // operand ("A"), but we need a TestOperand for 'out' to have a separate
  // 'raw buffer' both for the "A" input and the output
  TestOperand out;

  // We have this as floats even for the vertices that use int/unsigned for
  // simplicity.
  float scaleAValue;
  float scaleBValue;

  float tolerance; // Used only in some vertices.

  // Stream names used to transfer the host data, for the two operands and the
  // output. Must be different for each test that is run in the same graph/CS.
  std::string writeNameA;
  std::string writeNameB;
  std::string readName;

  /// \param[in] v      The vertex (with operation and data type) to test.
  /// \param[in] seq    A sequential index, different for each test
  /// \param[in] sz     Sizes of operands for this test., from cmd line.
  /// \param[in] tolerance used only for vertices that have that field
  /// \param[in] scaleAVal used only for vertices that have it. If 'nullopt', a
  ///                      random value will be generated
  /// \param[in] scaleBVal If 'nullopt', a random value will be generated
  TestRecord(std::shared_ptr<VertexDesc> v, unsigned seq, const SizeDesc &sz,
             boost::optional<unsigned> operandOffset, float tolerance,
             std::optional<float> scaleAVal, std::optional<float> scaleBVal)
      : vertex(std::move(v)), operandOffset(operandOffset),
        tolerance(tolerance) {
    writeNameA = "A_" + to_string(seq);
    writeNameB = "B_" + to_string(seq);
    readName = "Aout_" + to_string(seq);
    // Adjust size from command line parameters, according to test type.
    size = sz.adjust(vertex->is2D);
    if (!vertex->hasScaleA) {
      scaleAValue = 1.0;
    }

    // If we are generating random values for the scale values, and we have half
    // values, we want to avoid doing a subtraction, either a 'Subtract/Minus'
    // vertex or an 'Add/Plus' one with scales of opposite sign.
    // This is because subtraction can give results very close to zero, that
    // can be quite different from the float computation on the host (for
    // verification)
    float minAScale, maxAScale, minBScale, maxBScale;
    if (vertex->dataAType == HALF || vertex->dataBType == HALF ||
        vertex->scaleType == HALF) {
      if (vertex->isSubtractFromX) {
        minAScale = 0;
        maxAScale = 0.5;
        minBScale = 1;
        maxBScale = 10;
      } else if (vertex->isSubtract) {
        minAScale = 1;
        maxAScale = 10;
        minBScale = -10;
        maxBScale = -1;
      } else {
        minAScale = minBScale = 1;
        maxAScale = maxBScale = 10;
      }
    } else {
      if (vertex->dataAType == UNSIGNED_INT) {
        minAScale = minBScale = 1;
      } else {
        minAScale = minBScale = -10;
      }
      maxAScale = maxBScale = 10;
    }
    static RandomEngine rndEng = std::minstd_rand(1);
    std::vector<float> scale(1);
    if (vertex->hasScaleA && scaleAVal == std::nullopt) {
      fillBuffer(FLOAT, rndEng, scale, 10, minAScale, maxAScale, true);
      scaleAValue = scale[0];
    } else {
      scaleAValue = *scaleAVal;
    }
    if (scaleBVal == std::nullopt) {
      fillBuffer(FLOAT, rndEng, scale, 10, minBScale, maxBScale, true);
      scaleBValue = scale[0];
    } else {
      scaleBValue = *scaleBVal;
    }

    // Integer types: round the scales to an integer
    if (!vertex->scaleType.isFloatingPoint()) {
      scaleAValue = round(scaleAValue);
      scaleBValue = round(scaleBValue);
    }
  };

  TestRecord(TestRecord &&) = default;

  std::string toString() {
    std::string scalesStr;
    if (vertex->hasScaleA) {
      scalesStr = (format("a:%g b:%g") % scaleAValue % scaleBValue).str();
    } else {
      scalesStr = (format("b:%g") % scaleBValue).str();
    }
    if (vertex->hasTolerance) {
      scalesStr += (format(" tol:%g") % tolerance).str();
    }
    return vertex->vClassFmt + size.toString(10) + "  " + scalesStr;
  }
};

bool isImplementedCombination(const std::string &name, const Type AType,
                              const Type BType, const Type scaleType,
                              const std::optional<bool> scaleIsConstant,
                              const bool constrainedAB) {
  using std::get;
  using std::nullopt;
  using std::tie;
  using std::tuple;
  struct Combination {
    // Valid triples of [A type, B type , scales type]
    std::vector<tuple<Type, std::optional<Type>, std::optional<Type>>> types;
    // Valid pairs for [constant,constraint]
    std::vector<tuple<std::optional<bool>, bool>> flags;
  };
  // One 'Combinations' contains all valid combos for a specific variant.
  using Combinations = std::vector<Combination>;
  // Check if the types+flags in 'v' match a valid combo in 'combinations'
  auto isValid = [&](const Combinations &combinations) {
    for (Combination cmb : combinations) {
      for (auto t : cmb.types) {
        Type tb = (get<1>(t) == nullopt) ? get<0>(t) : get<1>(t).value();
        Type st = (get<2>(t) == nullopt) ? get<0>(t) : get<2>(t).value();
        if (tie(AType, BType, scaleType) == tie(get<0>(t), tb, st)) {
          for (auto fl : cmb.flags) {
            if (tie(scaleIsConstant, constrainedAB) == fl)
              return true;
          }
        }
      }
    }
    return false;
  };
  // clang-format off
  if (name == "ScaledAddSupervisor" || name == "ScaledAdd2D") {
    static const Combinations cs= {
    //  A      B    scale
    {{{FLOAT, FLOAT, FLOAT},
      {HALF,  HALF,  HALF }},
    // constant  constraint
     {{false,     false},
      {false,     true },
      {true,      false},
      {true,      true }}},

    {{{HALF,  HALF,  FLOAT}},
     {{false,     false},
      {false,     true }}},

    {{{FLOAT, HALF,  HALF },
      {FLOAT, HALF,  FLOAT},
      {HALF,  FLOAT,  HALF},
      {HALF,  FLOAT,  FLOAT},
      {INT,   INT,   INT  },
      {UNSIGNED_INT, UNSIGNED_INT, UNSIGNED_INT}},
     {{false,     false},
      {true,      false}}}
    };
    return isValid(cs);
  } else if (name == "ScaledSubtractSupervisor") {
    static const Combinations cs = {
    {{{FLOAT, FLOAT, FLOAT},
      {HALF,  HALF,  HALF },
      {HALF,  FLOAT, HALF },
      {HALF,  HALF,  FLOAT}},
     {{nullopt,   false},
      {nullopt,   true }}},

    {{{INT,   INT,   INT},
      {UNSIGNED_INT, UNSIGNED_INT, UNSIGNED_INT}},
     {{nullopt,   false}}}};
    return isValid(cs);
  } else if (name == "ScaledSubtract2D") {
    static const Combinations cs = {
    {{{HALF,  nullopt, HALF },
      {FLOAT, nullopt, FLOAT},
      {HALF,  nullopt, FLOAT}},
     {{nullopt,   false},
      {nullopt,   true }}},

    {{{INT,   INT,   INT},
      {UNSIGNED_INT, UNSIGNED_INT, UNSIGNED_INT}},
     {{nullopt,   false}}}};
    return isValid(cs);
  } else if (name == "aXPlusbYSupervisor" ||
             name == "aXPlusbY2D") {
    static const Combinations cs = {
    {{{HALF, nullopt, HALF}},
     {{false,     false},
      {false,     true },
      {true,      false},
      {true,      true }}},

    {{{HALF, nullopt, FLOAT}},
     {{false,     false},
      {false,     true },
      {true,      false}}},

    {{{FLOAT, nullopt, FLOAT}},
     {{false,     false},
      {true,      false}}}
    };
    return isValid(cs);
  } else if (name == "aXMinusbYSupervisor" ||
             name == "aXMinusbY2D") {
    static const Combinations cs = {
    {
     {{HALF, nullopt, HALF },
      {HALF, nullopt, FLOAT}},
     {{false,     false},
      {false,     true }}},

     {{{FLOAT, nullopt, FLOAT}},
     {{false,     false}}}
    };
    return isValid(cs);
  } else if (name == "XMinusaXPlusbYSupervisor" ||
             name == "XMinusaXPlusbY2D") {
    static const Combinations cs = {
    {{{HALF, nullopt, nullopt}},
     {{false,     false},
      {false,     true },
      {true,      false},
      {true,      true }}}};
    return isValid(cs);
  }
  // clang-format on
  return false;
}

//***************************************************************************
/// Check if two values are 'equal enough'.
/// For half and float values.
bool equalValues(const bool isIpuModel, bool dataIsHalf, float expected,
                 float actual) {
  // This is NOT the tolerance field of the vertex, just used here to compensate
  // for the differences in the device precision.
  double tolerance = 0.000001;

  // Horrible contortions to verify result for halves. We should really
  // have a half bit-exact computation library for the host.
  if (dataIsHalf) {
    float clipTreshHalf =
        isIpuModel ? std::numeric_limits<float>::infinity() : 65504.0f;
    float clipValueHalf = 65488.0f;
    if (actual >= clipTreshHalf) {
      return expected >= clipValueHalf;
    } else if (actual <= -clipTreshHalf) {
      return expected <= -clipValueHalf;
    }

    tolerance = 0.003;
  }
  bool isEqual = false;
  double delta = expected - actual;
  if (expected == 0) {
    isEqual = std::abs(delta) < 10e-6;
  } else {
    delta = std::abs(delta / expected);
    isEqual = (delta <= tolerance);
  }
  return isEqual;
}

//***************************************************************************
/// Check if two values are 'equal enough'.
/// For int/unsigned/boolean values, results must be bit exact
template <typename T>
bool equalValues(const bool isIpuModel, bool dataIsHalf, T expected, T actual) {
  return expected == actual;
}

//*************************************************************************
/// Verifies the results of the test.
/// Input data and device results are stored in 'test'.
///
/// \tparam HostType         Type used on the host for A, B and scales.
/// \param[in]    target     Which target.
/// \param[in]    isIpuModel Was the device and IpuModel?
/// \param[in]    test       Describes the test to setup.
///
/// \return true if the values returned by the device match (with appropriate
///         tolerances) the one computed on the host
///
template <typename HostType>
bool verifyTest(const Target &target, bool isIpuModel, const TestRecord &test,
                const MiscOptions &options) {
  const VertexDesc &vertex = *test.vertex;

  // Convert the device data in host format. Also convert back the input data
  // in host format.
  std::vector<HostType> AHost(test.A.totalElems);
  std::vector<HostType> BHost(test.B.totalElems);
  std::vector<HostType> outHost(test.out.totalElems);
  copy(target, vertex.dataAType, test.A.rawBuf.get(), AHost.data(),
       AHost.size());
  copy(target, vertex.dataBType, test.B.rawBuf.get(), BHost.data(),
       BHost.size());
  copy(target, vertex.dataAType, test.out.rawBuf.get(), outHost.data(),
       outHost.size());
  if (options.printBuffers) {
    printBuffer("out", outHost, vertex.dataAType, test.out.rowSizes,
                test.out.offsets);
  }

  bool dataIsHalf = (vertex.dataAType == HALF || vertex.dataBType == HALF);
  // Check for mismatches on computed values
  auto &rowSizes = test.out.rowSizes; // same as test.in.rowSizes
  unsigned errCount = 0;              // how many mismatched elements we find
  unsigned numRows = rowSizes.size();
  for (unsigned row = 0; row < numRows; row++) {
    for (unsigned i = 0; i < rowSizes[row]; i++) {

      HostType valA = AHost[test.A.offsets[row] + i];
      HostType valB = BHost[test.B.offsets[row] + i];

      // Result from device
      HostType actual = outHost[test.out.offsets[row] + i];

      HostType expected;
      if (vertex.isSubtractFromX) {
        expected = valA * (1 - test.scaleAValue) + valB * test.scaleBValue;
      } else if (vertex.isSubtract) {
        expected = valA * test.scaleAValue - valB * test.scaleBValue;
      } else {
        expected = valA * test.scaleAValue + valB * test.scaleBValue;
      }

      if (!equalValues(isIpuModel, dataIsHalf, expected, actual)) {
        // If its is 2D, we want to show row and column where it failed, not
        // just the linear index.
        std::string valAStr = convertToString(valA);
        std::string valBStr = convertToString(valB);
        std::string scaleAStr = convertToString(test.scaleAValue);
        std::string scaleBStr = convertToString(test.scaleBValue);
        std::string opStr;
        if (vertex.isSubtractFromX) {
          opStr = (format(" %s - %s x %s + %s x %s") % valAStr % scaleAStr %
                   valAStr % scaleBStr % valBStr)
                      .str();
        } else if (vertex.hasScaleA && vertex.isSubtract) {
          opStr = (format(" %s x %s - %s x %s") % scaleAStr % valAStr %
                   scaleBStr % valBStr)
                      .str();
        } else if (vertex.hasScaleA && !vertex.isSubtract) {
          opStr = (format(" %s x %s + %s x %s") % scaleAStr % valAStr %
                   scaleBStr % valBStr)
                      .str();
        } else if (vertex.isSubtract) {
          opStr =
              (format(" %s - %s x %s") % valAStr % scaleBStr % valBStr).str();
        } else {
          opStr =
              (format(" %s + %s x %s") % valAStr % scaleBStr % valBStr).str();
        }
        std::cerr << format("out[%s] = %s  =>  expected:%s;  actual:%s\n") %
                         (vertex.is2D ? to_string(row) + "][" + to_string(i)
                                      : to_string(i)) %
                         opStr % convertToString(expected) %
                         convertToString(actual);
        errCount++;
      }
    }
  }
  if (errCount > 0) {
    std::cerr << "Failed: mismatch on " << errCount << " value(s)\n";
  }

  // Check for overwrites past the end of each row
  bool paddingIsNan = (vertex.dataAType == HALF || vertex.dataAType == FLOAT);
  auto overwriteCount =
      test.out.checkPadBytes(target, vertex.is2D, isIpuModel, paddingIsNan);

  return (errCount == 0) && (overwriteCount == 0);
}

//*************************************************************************
// Populate the operand buffers and the scale values
template <typename HostType>
void fillBuffers(TestRecord &test, unsigned randomSeed,
                 std::vector<HostType> &AHost, std::vector<HostType> &BHost) {
  const VertexDesc &vertex = *test.vertex;
  const Type &dataAType = vertex.dataAType;
  const Type &dataBType = vertex.dataBType;
  RandomEngine rndEng;
  if (randomSeed != 0)
    rndEng = std::minstd_rand(randomSeed);

  HostType min = (dataAType == INT) ? -300 : 0;
  HostType max = 300;

  fillBuffer(dataAType, rndEng, AHost, 100, min, max, false);
  fillBuffer(dataBType, rndEng, BHost, 200, min, max, false);
}

//*************************************************************************
/// Setup one vertex test.
///
/// \tparam HostType        Type used on the host for A, B and scales.
/// \param[in]    target    Which target.
/// \param[inout] graph     The graph.
/// \param[inout] upload    A Sequence where we will add the uploading of the
///                         data for this vertex (from the host to the device)
/// \param[inout] alignCS   Compute set containing the 'alignment' vertices.
/// \param[inout] cs        The compute set to add the vertex to.
/// \param[inout] download  A Sequence where we will add the downloading of the
///                         result data (from device to host)
/// \param[inout] streamMap Used to pass the appropriate streams for upload/
///                         download when running on the device.
/// \param[inout] test      Describes the test to setup. Pointers to data and
///                         output buffers are setup here.
/// \param[in]    tile      Which tile to run this test on.
/// \param[in]    options  Global options.
///
template <typename HostType>
static void setupTest(const Target &target, bool isIpuModel, Graph &graph,
                      Sequence &upload, ComputeSet &alignCS, ComputeSet &cs,
                      Sequence &download, StreamMap &streamMap,
                      TestRecord &test, unsigned tile,
                      const MiscOptions &options) {
  const VertexDesc &vertex = *test.vertex;
  const Type &dataAType = vertex.dataAType;
  const Type &dataBType = vertex.dataBType;
  const Type &scaleType = vertex.scaleType;

  if (std::find(std::begin(verticesNames), std::end(verticesNames),
                vertex.name) == std::end(verticesNames)) {
    throw std::runtime_error(vertex.name +
                             " is not a valid vertex name. Maybe you wanted "
                             "to use the --vertexRE option for a regular "
                             "expression?");
  }

  // === Setup offsets for padding
  test.A.setup(target, dataAType, test.size.val, options.alignStart);
  test.B.setup(target, dataBType, test.size.val, options.alignStart);
  test.out.setup(target, dataAType, test.size.val, options.alignStart);

  // === Allocate and initialise host buffers with appropriate values.
  std::vector<HostType> AHost(test.A.totalElems);
  std::vector<HostType> BHost(test.B.totalElems);
  fillBuffers(test, options.randomSeed, AHost, BHost);

  // If requested, print the tensor buffers and scale values
  if (options.printBuffers) {
    if (vertex.hasScaleA) {
      std::cout << "scaleA:" << test.scaleAValue << "\n";
    }
    printBuffer("A", AHost, dataAType, test.A.rowSizes, test.A.offsets);
    std::cout << "scaleB:" << test.scaleBValue << "\n";
    printBuffer("B", BHost, dataBType, test.B.rowSizes, test.B.offsets);
  }

  // === Create graph variables.
  // The two operands can be created in two ways, based on the value of
  // operandOffs:
  //
  // 1. If operandOffs contains a value (is not 'none') a single graph variable
  //    will be created, that is then sliced in two, one slice for each operand.
  //    This allows controlling the offset in device memory between the two
  //    operands, as required by some tests (as long as poplar doesn't do
  //    rearrangements!).
  //    Doing this can slow down significantly the execution when running tests
  //    in groups.
  //    If the offset is specified as 0, the two slices will be placed back-to-
  //    back (no padding, apart from a possible 8 byte alignment).
  //    If the offset (in bytes) is > 0, it must be a multiple of 8 bytes and
  //    we will add padding between the two slices, so that the distance between
  //    the START of first operand and the START of the second is equal to the
  //    specified offset (so it must be greater than the size of the first
  //    operand).
  //
  // 2. If operandOffs is empty ('none') the operands will be created as two
  //    separate graph variables. This let poplar place the operands in memory
  //    as it sees fit and it is much faster.

  Tensor A, B;

  if (!test.operandOffset) {
    // Optional not specified ('none'), let poplar place the operands in memory
    A = graph.addVariable(dataAType, {test.A.totalElems}, "A");
    graph.setTileMapping(A, tile);
    B = graph.addVariable(dataBType, {test.B.totalElems}, "B");
    graph.setTileMapping(B, tile);
    // === Auxiliary vertices required to align the tensors to 8 bytes
    createAlignVertex(graph, alignCS, A, tile);
    createAlignVertex(graph, alignCS, B, tile);
  } else {
    const static unsigned ALIGN = 8; // align sizes/offsets on this num. bytes
    const unsigned dataTypeSize = target.getTypeSize(dataAType);
    assert((ALIGN % dataTypeSize) == 0);
    const unsigned nBytes1 =
        test.A.totalElems * dataTypeSize; // Operand1 size bytes
    const unsigned nBytes1Aligned = gccs::alignNext(nBytes1, ALIGN);

    // Offset optional specified, check it's ok (must be aligned and big enough)
    unsigned offs = *test.operandOffset;
    if ((offs % ALIGN) != 0) {
      throw std::runtime_error("The specified --offset-operands (" +
                               to_string(offs) + ") should be multiple of " +
                               to_string(ALIGN));
    } else if (offs == 0) {
      // '0' means "no padding" between operands, just make sure op2 is aligned
      offs = nBytes1Aligned;
    } else if (offs < nBytes1) {
      // offset is the distance between the START of the two operands, so cannot
      // be smaller than the size of the first one.
      throw std::runtime_error("The specified --offset-operands (" +
                               to_string(offs) +
                               ") is smaller than the size of the first "
                               "operand (" +
                               to_string(nBytes1) + ")");
    }

    // Create one single tensor to be sliced in two
    unsigned offsIn2 = std::max(nBytes1Aligned, offs); // Where operand 2 starts
    unsigned offsIn2Elems = offsIn2 / dataTypeSize; // Now in elems, not bytes
    unsigned totSize = offsIn2Elems + test.B.totalElems;
    Tensor in = graph.addVariable(dataAType, {totSize}, "A_B");
    graph.setTileMapping(in, tile);
    createAlignVertex(graph, alignCS, in, tile);

    A = in.slice(0, test.A.totalElems);
    B = in.slice(offsIn2Elems, offsIn2Elems + test.B.totalElems);
  }

  // === Create the vertex
  auto v = graph.addVertex(cs, vertex.vClass);
  graph.setTileMapping(v, tile);

  // We copy the data in the 'test.out.rawBuf' from where it
  // would be both written to the device and read back with the result.
  // But we also copy it in the 'test.A.rawBuf' to be used later for the
  // verification.
  test.A.rawBuf =
      allocateHostMemoryForTensor(target, A, graph.getReplicationFactor());
  test.out.rawBuf = allocateHostMemoryForTensor(A, test.writeNameA, graph,
                                                upload, download, streamMap);
  copy(target, AHost.data(), AHost.size(), dataAType, test.A.rawBuf.get());
  copy(target, AHost.data(), AHost.size(), dataAType, test.out.rawBuf.get());

  test.B.rawBuf = allocateHostMemoryForTensor(B, test.writeNameB, graph, upload,
                                              boost::none, streamMap);
  copy(target, BHost.data(), BHost.size(), dataBType, test.B.rawBuf.get());

  // Fill the padding space in the input buffers (with NaNs) for overprocessing
  // detection (only for floating point types) and overrrun detection.
  // For non floating types, fill the output buffer padding, to detect overruns
  if (dataAType == FLOAT || dataAType == HALF) {
    test.out.setPadBytes(target, isIpuModel, true);
    test.B.setPadBytes(target, isIpuModel, true);
  } else {
    test.out.setPadBytes(target, isIpuModel);
  }

  // Connect the operands
  TestOperand::OperandType opType = vertex.is2D
                                        ? TestOperand::OperandType::is2D
                                        : TestOperand::OperandType::is1D;
  test.A.connectOperand(graph, v, opType, A, "A");
  test.B.connectOperand(graph, v, opType, B, "B");

  // All vertices have a 'scaleB' field, some have a 'scaleA'
  auto setScale = [&](std::string fieldName, HostType val) {
    if (vertex.scaleIsConstant == true) {
      graph.setInitialValue(v[fieldName], val);
    } else {
      Tensor scale;
      if (vertex.is2D) {
        scale = graph.addConstant(scaleType, {}, val);
      } else {
        scale = graph.addConstant(scaleType, {1}, val);
      }
      graph.connect(v[fieldName], scale);
      graph.setTileMapping(scale, tile);
    }
  };
  setScale("scaleB", test.scaleBValue);
  if (vertex.hasScaleA) {
    setScale("scaleA", test.scaleAValue);
  }

  // Additional fields depend on vertex types
  if (!vertex.is2D) {
    graph.setInitialValue(v["size"], test.A.rowSizes[0]);
  }
  if (vertex.hasTolerance) {
    graph.setInitialValue(v["tolerance"], test.tolerance);
  }
}

// A macro to match the device data and output types with the types used
// on the host. This is used in doSetupTest and doVerifyTest.
// The FLOAT and HALF types both use 'float' buffers on the host; this
// simplifies this macro a lot.
#define SELECT_BY_TYPES()                                                      \
  /* Note that for both HALF and FLOAT the host buffers are 'float' */         \
  SELECT_ONE(HALF, HALF, HALF, float)                                          \
  SELECT_ONE(HALF, HALF, FLOAT, float)                                         \
  SELECT_ONE(HALF, FLOAT, HALF, float)                                         \
  SELECT_ONE(HALF, FLOAT, FLOAT, float)                                        \
  SELECT_ONE(FLOAT, HALF, HALF, float)                                         \
  SELECT_ONE(FLOAT, HALF, FLOAT, float)                                        \
  SELECT_ONE(FLOAT, FLOAT, FLOAT, float)                                       \
  SELECT_ONE(INT, INT, INT, int)                                               \
  SELECT_ONE(UNSIGNED_INT, UNSIGNED_INT, UNSIGNED_INT, unsigned)               \
  /* The combination of types is not valid */                                  \
  throw std::runtime_error(                                                    \
      "The combination of A type (" + vertex.dataAType.toString() +            \
      "), B type (" + vertex.dataBType.toString() + ") and scale type (" +     \
      vertex.scaleType.toString() + ") is not supported");

//*************************************************************************
// Calls the appropriate version of setupTest using the template parameters
// relevant to the data types.
// See 'setupTest()' for parameters
static void doSetupTest(const Target &target, bool isIpuModel, Graph &graph,
                        Sequence &upload, ComputeSet &alignCS, ComputeSet &cs,
                        Sequence &download, StreamMap &streamMap,
                        TestRecord &test, unsigned tile,
                        const MiscOptions &options) {
  VertexDesc &vertex = *test.vertex;
  // Call the appropriate instantiation of the templated function
#define SELECT_ONE(IPU_A_TYPE, IPU_B_TYPE, IPU_SCALE_TYPE, HOST_TYPE)          \
  if (vertex.dataAType == IPU_A_TYPE && vertex.dataBType == IPU_B_TYPE &&      \
      vertex.scaleType == IPU_SCALE_TYPE) {                                    \
    setupTest<HOST_TYPE>(target, isIpuModel, graph, upload, alignCS, cs,       \
                         download, streamMap, test, tile, options);            \
    return;                                                                    \
  }
  SELECT_BY_TYPES()
#undef SELECT_ONE
}

//*************************************************************************
// Calls the appropriate version of verifyTest using the template parameters
// relevant to the data/output types.
// See 'verifyTest()' for parameters
bool doVerifyTest(const Target &target, bool isIpuModel, TestRecord &test,
                  const MiscOptions &options) {
  VertexDesc &vertex = *test.vertex;

#define SELECT_ONE(IPU_A_TYPE, IPU_B_TYPE, IPU_SCALE_TYPE, HOST_TYPE)          \
  if (vertex.dataAType == IPU_A_TYPE && vertex.dataBType == IPU_B_TYPE &&      \
      vertex.scaleType == IPU_SCALE_TYPE) {                                    \
    return verifyTest<HOST_TYPE>(target, isIpuModel, test, options);           \
  }
  SELECT_BY_TYPES()
#undef SELECT_ONE
}

// One of the boolean flags is optional, so we want to be able to read a
// command line parameter that can contain 'true'/'false'/'none'
void convertToOptBool(const std::vector<std::string> strs,
                      std::vector<std::optional<bool>> &optBools) {
  for (auto s : strs) {
    if (s == "true" || s == "1") {
      optBools.push_back(true);
    } else if (s == "false" || s == "0") {
      optBools.push_back(false);
    } else if (s == "none") {
      optBools.push_back(std::nullopt);
    } else {
      throw std::runtime_error("Invalid 'constant-scale' option: it must be "
                               "one of: none, true (1), false (0)");
    }
  }
}

//*************************************************************************
int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  std::vector<Type> dataATypes;
  std::vector<Type> dataBTypes;
  std::vector<Type> scaleTypes;
  std::vector<float> scaleAValues = {};
  std::vector<float> scaleBValues = {};

  std::vector<SizeDesc> sizes = {{false, {25, 12, 21}}};

  std::vector<std::string> scaleIsConstant_; // as strings
  std::vector<std::optional<bool>> scaleIsConstant;

  std::vector<bool> constrainedAB;

  std::vector<float> tolerances = {0.001};

  std::vector<std::string> vertices;
  std::string vertexRE; // regular expression for vertex name

  boost::optional<unsigned> operandOffset;
  MiscOptions options;

  // clang-format off
  const static std::string description =
  "Tests one or more of the ScaledAdd/Subtract/aXPlus/MinusbY vertices with "
  "different data types and template parameter falgs, with a default data size,"
  " or a specified one.\n"
  "If no vertex is specified, all will be tested. Same for data types and "
  "flags\n"
  "Using the --compare-cycles option, cycles reported when running the vertex\n"
  "on two different devices can be compared."
  "\n"
  "\n"
  "Details of options are:";

  po::options_description poDesc(description);

  // Get a string with all vertices names, comma separated
  std::string allVerticesStr;
  for (auto &n : verticesNames)
    allVerticesStr += (allVerticesStr.empty()? "" : ", ") + n;

  poDesc.add_options()
    ("vertex",
     po::value<std::vector<std::string>>(&vertices)->multitoken(),
     ("Vertices to test, one or more of: " + allVerticesStr).c_str())
    ("vertexRE",
     po::value<std::string>(&vertexRE),
     "Regular expression to specify vertex names (alternative to --vertex)")
    ("data-a-type",
     po::value<std::vector<Type>>(&dataATypes)->multitoken(),
     "Data type for the first tensor: one or more of half, float, int, uint")
    ("data-b-type",
     po::value<std::vector<Type>>(&dataBTypes)->multitoken(),
     "Data type for the second tensor: one or more of half, float, int, uint")
    ("scale-type",
     po::value<std::vector<Type>>(&scaleTypes)->multitoken(),
     "Data type for the scale values: one or more of half, float, int, uint")
    ("scale-a",
     po::value<std::vector<float>>(&scaleAValues)->multitoken(),
     "Value for 'scale a' to use (single or multiple)")
    ("scale-b",
     po::value<std::vector<float>>(&scaleBValues)->multitoken(),
     "Value for 'scale b' to use (single or multiple)")
    ("constrainedAB",
     po::value<std::vector<bool>>(&constrainedAB)->multitoken(),
     "Vertex template flag: are the two operands constrained to be in "
     "different memory elements?")
    ("tolerance",
     po::value<std::vector<float>>(&tolerances)->multitoken(),
     "Vertex 'tolerance' field used in mixed types vertices")
    ("size",
     po::value<std::vector<SizeDesc>>(&sizes)->multitoken(),
     "Size(s) for rows of first operand. Single value for a 1D vertex, "
     "square-bracket-delimited, comma-separated list for a 2D vertex")
    ("offset-operands",
     po::value<boost::optional<unsigned>>(&operandOffset)->implicit_value(0),
     "The 2nd operand will be placed in memory so that its start is at the "
     "specified number of bytes from the start of the 1st operand. If 0, the "
     "two operands will be allocated back-to-back. If not present, placement "
     "in memory will be left to poplar")
    ;
  // clang-format on
  addCommonOptions(poDesc, deviceType, options);

  parseOptions(argc, argv, poDesc);

  // === Some parameter checks
  if (!vertexRE.empty() && !vertices.empty()) {
    throw std::runtime_error(
        "Cannot specify both --vertexRE and --vertex option");
  }

  // === If no vertices specified, test 'em all
  if (vertices.empty()) {
    vertices = verticesNames;
  }

  // === If no data type specified, test 'em all
  if (dataATypes.empty()) {
    dataATypes = {HALF, FLOAT, INT, UNSIGNED_INT};
  }
  if (dataBTypes.empty()) {
    dataBTypes = {HALF, FLOAT, INT, UNSIGNED_INT};
  }
  if (scaleTypes.empty()) {
    scaleTypes = {HALF, FLOAT, INT, UNSIGNED_INT};
  }

  // === If the flags are not specified, test both
  if (scaleIsConstant_.empty()) {
    scaleIsConstant = {std::nullopt, false};
  } else {
    convertToOptBool(scaleIsConstant_, scaleIsConstant);
  }
  if (constrainedAB.empty()) {
    constrainedAB = {false, true};
  }

  std::regex vertexRegEx(vertexRE);

  std::vector<std::optional<float>> justOne = {1.0};
  std::vector<std::optional<float>> justNull = {std::nullopt};
  std::vector<std::optional<float>> scaleAValues_ =
      convertToOptionalVector(scaleAValues);
  std::vector<std::optional<float>> scaleBValues_ =
      convertToOptionalVector(scaleBValues);
  std::vector<std::shared_ptr<TestRecord>> tests;
  unsigned numTests = 0;
  unsigned errCount = 0;
  // Loop over all vertices, types and options
  for (std::string vertexName : vertices) {
    // If a regex was specified, see if it matches
    if (vertexRE.empty() || std::regex_search(vertexName, vertexRegEx)) {
      for (auto Atype : dataATypes) {
        for (auto Btype : dataBTypes) {
          for (auto scaleType : scaleTypes) {
            for (auto &constant : scaleIsConstant) {
              for (auto const &constrained : constrainedAB) {
                if (isImplementedCombination(vertexName, Atype, Btype,
                                             scaleType, constant,
                                             constrained)) {
                  auto vertex = std::make_shared<VertexDesc>(
                      vertexName, Atype, Btype, scaleType, constant,
                      constrained);
                  // Adjust the vector of scaleA values based on the type of
                  // vertex (might not have scaleA); if no scale was specified
                  // use std::nullopt
                  for (auto &scaleA : vertex->hasScaleA
                                          ? (scaleAValues_.size() > 0)
                                                ? scaleAValues_
                                                : justNull
                                          : justOne) {
                    // Adjust the vector of scaleB values ; if no scale was
                    // specified use std::nullopt
                    for (auto &scaleB : (scaleBValues_.size() > 0)
                                            ? scaleBValues_
                                            : justNull) {
                      for (auto tolerance : tolerances) {
                        for (auto sz : sizes) {
                          numTests++;
                          auto testRec = std::make_shared<TestRecord>(
                              vertex, numTests, sz, operandOffset, tolerance,
                              scaleA, scaleB);
                          addOneTest<TestRecord, VertexDesc>(
                              tests, testRec, deviceType, errCount, options);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  runAllTests<TestRecord>(tests, numTests, deviceType, errCount, options);
  return (errCount == 0) ? 0 : 1; // returning 1 means an error.
}
