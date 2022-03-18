// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

// Low level codelet test for the ConvPartial1x1Out vertex.
// This allow to test the assembly code without adding the complexity
// of the code in Convolution.cpp

#define BOOST_TEST_MODULE ConvPartial1x1Out
#include "poplibs_test/Util.hpp"
#include "poplin/codelets.hpp"
#include "popops/Cast.hpp"
#include "popops/codelets.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <boost/multi_array.hpp>
#include <functional>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>

namespace utf = boost::unit_test::framework;

using namespace poplar;
using namespace poputil;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poplibs_support;

// Some test cases are to be run only on IPU2
boost::test_tools::assertion_result isIpu2(boost::unit_test::test_unit_id = 0) {
  return TEST_TARGET == DeviceType::Sim2 ||
         TEST_TARGET == DeviceType::IpuModel2;
}

// Some test cases are to be run only on IPU21
boost::test_tools::assertion_result
isIpu21(boost::unit_test::test_unit_id = 0) {
  return TEST_TARGET == DeviceType::Sim21 ||
         TEST_TARGET == DeviceType::IpuModel21;
}

const unsigned LOAD_SIZE = 8; // How many bytes in one 'ld64' IPU instruction

// These definitions limits the testing to the variants of the vertex for which
// the template parameter 'useLimitedVer' is 'true'
using UnsignedType = unsigned short;
using SignedType = short;
const auto worklistEntryType = UNSIGNED_SHORT;

// The worklist ('partition' list) has a triplet of integer values for each
// worker thread, specifying what part of the work the thread has to do.
struct worklistElem {
  // Units of 8 bytes, must contain a multiple of outChansPerGroup values
  unsigned outOffs;
  //  How many 'Output Channel Groups' this worker will compute
  int numFieldElems;
  // Units of 8 bytes, must contain a multiple of intChansPerGroup values
  unsigned inOffs;
};

// This contains all test parameters that specify a single test.
struct testParams {
  testParams()
      : verificationTolerance(0.00001), numConvUnits(8), inStride(1),
        alignedAllocation(false), inType(FLOAT), accumType(FLOAT),
        fieldWidth(1), numConvGroups(1), numOutGroups(1), numInGroups(1),
        outChansPerGroup(8), flipOut(false), use128BitLoad(false),
        weightFp8Format(Fp8Format::QUART143), weightFp8Scale(0),
        inputFp8Format(Fp8Format::QUART143), inputFp8Scale(0) {}
  double verificationTolerance;
  unsigned numConvUnits;
  int inStride;
  bool alignedAllocation;

  Type inType;
  Type accumType;

  unsigned fieldWidth;
  UnsignedType numConvGroups;
  UnsignedType numOutGroups;
  UnsignedType numInGroups;
  UnsignedType outChansPerGroup;
  bool flipOut;
  bool use128BitLoad;

  Fp8Format weightFp8Format;
  int weightFp8Scale;
  Fp8Format inputFp8Format;
  int inputFp8Scale;

  std::vector<worklistElem> worklists;
};

/// Performs the same computation done by the vertex; used to verify results
/// coming from the device.
/// Note that the input parameters are not pre-processed as they are for the
/// vertex.
void vertexCompute(const unsigned numContexts, const unsigned inTypeSize,
                   const unsigned accumTypeSize, const unsigned inChansPerGroup,
                   const boost::multi_array<float, 2> &in,
                   const boost::multi_array<float, 2> &weights,
                   boost::multi_array<float, 2> &out, const testParams &t) {
  for (unsigned cg = 0; cg < t.numConvGroups; ++cg) {
    for (unsigned og = 0; og < t.numOutGroups; ++og) {

      const auto outRow = cg * t.numOutGroups + og;

      for (unsigned ig = 0; ig < t.numInGroups; ++ig) {

        const auto wRow = cg * t.numOutGroups * t.numInGroups +
                          ig * t.numOutGroups + (t.numOutGroups - 1 - og);
        const auto inRow = cg * t.numInGroups + ig;

        for (unsigned context = 0; context < numContexts; ++context) {

          // Adjust the offset fields for this context from the worklist.
          // From units of 8 bytes (LOAD_SIZE) to the number of groups of chans
          const auto outOffset =
              (t.worklists[context].outOffs * LOAD_SIZE / accumTypeSize) /
              t.outChansPerGroup;
          const auto inOffset =
              (t.worklists[context].inOffs * LOAD_SIZE / inTypeSize) /
              inChansPerGroup;

          for (int i = 0; i < t.worklists[context].numFieldElems; ++i) {

            const auto inCol = (inOffset + i * t.inStride) * inChansPerGroup;
            int outCol = t.flipOut ? (outOffset - i) * t.outChansPerGroup
                                   : (outOffset + i) * t.outChansPerGroup;
            if (outCol < 0) {
              throw poputil::poplibs_error("outCol is negative: " +
                                           std::to_string(outCol));
            }
            // Process a whole 'Output Channel Group'
            for (unsigned outChan = 0; outChan < t.outChansPerGroup;
                 ++outChan) {

              // Do a dot product of a 'Input Channel Group' (a segment of a
              // row of 'in[][]' and of 'weight[][]').
              const auto wCol = outChan * inChansPerGroup;
              float sum = 0;
              for (unsigned inChan = 0; inChan < inChansPerGroup; ++inChan) {
                sum += in[inRow][inCol + inChan] * weights[wRow][wCol + inChan];
              }

              // If this is the first time through this output element, set
              // it, to start the accumulation, otherwise accumulate
              if (ig == 0)
                out[outRow][outCol + outChan] = sum;
              else
                out[outRow][outCol + outChan] += sum;
            }
          }
        }
      }
    }
  }
}

/// Fills the data buffers for the convolution with values.
///
/// \param[out] in       The data for the input tensor
///
/// \param[out] weights  The data for the weights tensor
///
/// \param[out] out      The tensor that will contain the result of the vertex.
///                      This will be overwritten when the vertex is run, but
///                      we populate it with non-zero values for debug purposes.
///
void populateData(boost::multi_array<float, 2> &in,
                  boost::multi_array<float, 2> &weights,
                  boost::multi_array<float, 2> &out, bool limitRangeForFp8) {
  for (unsigned i = 0; i < in.shape()[0]; i++) {
    for (unsigned j = 0; j < in.shape()[1]; j++) {
      if (limitRangeForFp8) {
        in[i][j] = ((i + j) % 5);
      } else {
        in[i][j] = i * 0.27182818 - j * 0.31415926;
      }
    }
  }
  float weightVal = 1.0f;
  for (unsigned i = 0; i < weights.shape()[0]; i++) {
    for (unsigned j = 0; j < weights.shape()[1]; j++) {
      if (limitRangeForFp8) {
        weights[i][j] = weightVal;
        weightVal = unsigned((weightVal + 1)) % 6;
      } else {
        weights[i][j] = i * 0.142857 - j * 0.0618034;
      }
    }
  }

  for (unsigned i = 0; i < out.shape()[0]; i++) {
    for (unsigned j = 0; j < out.shape()[1]; j++) {
      out[i][j] = i * 1000 + j;
    }
  }
}

/// Verifies the results computed by the device against the expected results
/// computed by the test code on the host.
///
/// \param[in] result      results computed by the device code.
/// \param[in] expected    results computed by the test code on the host.
/// \param[in] tolerance   passed to 'checkIsClose()'.
///
void compareResults(const boost::multi_array<float, 2> &results,
                    const boost::multi_array<float, 2> &expected,
                    const double tolerance) {

  unsigned errors = 0;
  for (unsigned i = 0; i < results.shape()[0]; i++) {
    for (unsigned j = 0; j < results.shape()[1]; j++) {
      if (!checkIsClose(results[i][j], expected[i][j], tolerance)) {
        std::cout << "out[" << i << "][" << j
                  << "] - computed:" << results[i][j]
                  << ", expected:" << expected[i][j] << "\n";
        errors++;
      }
    }
  }
  BOOST_CHECK(errors == 0);

#define PRINT_RESULTS 0
#if PRINT_RESULTS
  std::cout << "  -- result --\n";
  for (int i = 0; i < results.shape()[0]; i++) {
    for (int j = 0; j < results.shape()[1]; j++) {
      std::cout << std::setprecision(5) << std::setw(7) << results[i][j] << " ";
    }
    std::cout << "\n";
  }
  if (errors > 0) {
    std::cout << "  -- expected --\n";
    for (int i = 0; i < expected.shape()[0]; i++) {
      for (int j = 0; j < expected.shape()[1]; j++) {
        std::cout << std::setprecision(5) << std::setw(7) << expected[i][j]
                  << " ";
      }
      std::cout << "\n";
    }
  }
#endif
}

///
/// Runs a test according to the parameters.
///
void runTest(const struct testParams &t) {
  auto device = createTestDevice(TEST_TARGET);
  const Target &target = device.getTarget();
  Graph graph(target);

  unsigned numContexts = target.getNumWorkerContexts();

  // Check correctness of test parameters
  std::string err;
  if ((t.numConvUnits != 4) && (t.numConvUnits != 8) &&
      (t.numConvUnits != 16)) {
    err = "'numConvUnits' must be 4, 8 or 16 (16 only on IPU2 / IPU21)";
  } else if ((t.numConvUnits == 16) && !isIpu2() && !isIpu21()) {
    err = "'numConvUnits' can be 16 only on IPU2";
  } else if ((t.outChansPerGroup % t.numConvUnits) != 0) {
    err = "'outChansPerGroup' must be a multiple of 'numConvUnits'";
  } else if (t.worklists.size() != numContexts) {
    err = "'worklists' must have " + std::to_string(numContexts) + " elements";
  }
  if (!err.empty()) {
    throw poputil::poplibs_error(err);
  }
  const bool isFp8 = t.inType == QUARTER;

  // This value is hard coded in the vertex assembly because of the AMP
  // hardware structure
  unsigned inChansPerGroup;
  if (t.numConvUnits == 8) {
    inChansPerGroup = (t.inType == HALF) ? 16 : 8;
  } else {
    inChansPerGroup = isFp8 ? 32 : 16;
  }

  // "Host side" buffers
  boost::multi_array<float, 2> inBuf{
      boost::extents[t.numConvGroups * t.numInGroups]
                    [t.fieldWidth * inChansPerGroup]};
  boost::multi_array<float, 2> weightsBuf{
      boost::extents[t.numConvGroups * t.numInGroups * t.numOutGroups]
                    [inChansPerGroup * t.outChansPerGroup]};
  boost::multi_array<float, 2> outBuf{
      boost::extents[t.numConvGroups * t.numOutGroups]
                    [t.fieldWidth * t.outChansPerGroup]};

  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  Sequence prog;

  auto cs = graph.addComputeSet("cs");

  populateData(inBuf, weightsBuf, outBuf, isFp8);

  // The vertex wants the worklist as a flat vector of 6x3 UnsignedType.
  // Also, the 'numFieldElems' needs to be adjusted by subtracting 3 to its
  // value, as the assembly wants it. So, it can be negative; but the type
  // required by the vertex definition is 'UnsignedType'!
  const unsigned WLIST_ELEM_SIZE = 3;
  std::vector<UnsignedType> worklistsBuf(WLIST_ELEM_SIZE * numContexts);
  for (unsigned i = 0; i < numContexts; i++) {
    worklistsBuf[WLIST_ELEM_SIZE * i + 0] = t.worklists[i].outOffs;
    worklistsBuf[WLIST_ELEM_SIZE * i + 1] = t.worklists[i].numFieldElems - 3;
    worklistsBuf[WLIST_ELEM_SIZE * i + 2] = t.worklists[i].inOffs;
  }

  // Extract the (2-D) shapes, for creating the Tensors
  std::vector<unsigned long> inShape{inBuf.shape(), inBuf.shape() + 2};
  std::vector<unsigned long> wShape{weightsBuf.shape(), weightsBuf.shape() + 2};
  std::vector<unsigned long> outShape{outBuf.shape(), outBuf.shape() + 2};

  // Buffer for the results computed on host later, for verification. We set
  // it with the same initialisation values as the one we pass to the device
  // for the check
  boost::multi_array<float, 2> outHost{outShape};
  outHost = outBuf;

  const auto hostInType = isFp8 ? HALF : t.inType;
  const unsigned inTypeSize = target.getTypeSize(hostInType);
  Tensor in = graph.addVariable(hostInType, inShape, "in");
  Tensor out = graph.addVariable(t.accumType, outShape, "out");
  Tensor weights = graph.addVariable(hostInType, wShape, "weights");

  graph.setTileMapping(in, 0);
  graph.setTileMapping(weights, 0);
  graph.setTileMapping(out, 0);

  const std::string vertexName =
      templateVertex("poplin::ConvPartial1x1Out", t.inType, t.accumType, true,
                     t.use128BitLoad, t.numConvUnits, false);
  std::cout << "========= " << vertexName << "\n";

  auto v = graph.addVertex(cs, vertexName);
  graph.setTileMapping(v, 0);

  graph.connect(v["out"], out);
  auto worklists = graph.addConstant(worklistEntryType, {worklistsBuf.size()},
                                     worklistsBuf.data(), "worklists");
  graph.setTileMapping(worklists, 0);
  graph.connect(v["worklists"], worklists);

  if (isFp8) {
    // Applying a scale in metadata should result in casting half->fp8 and
    // applying 2^(-scale) for both weights and in, so the values stored as fp8
    // are smaller.
    // But then when doing the convolution 2^(scaleWeights + scaleIn) should be
    // applied so we get the same result.  (Subject to values being
    // representable in each number format)
    auto weightsMetadata =
        createFp8MetadataTensor(graph, t.weightFp8Format, t.weightFp8Scale);
    auto inMetadata =
        createFp8MetadataTensor(graph, t.inputFp8Format, t.inputFp8Scale);

    // TODO - T57103 won't need an on-IPU cast once we can copy data to the IPU
    auto inFp8 = popops::cast(graph, in, QUARTER, inMetadata, prog, "CastIn");
    auto weightsFp8 = popops::cast(graph, weights, QUARTER, weightsMetadata,
                                   prog, "CastWeights");
    graph.connect(v["in"], inFp8);
    graph.connect(v["weights"], weightsFp8);
  } else {
    graph.connect(v["in"], in);
    graph.connect(v["weights"], weights);
  }

  graph.createHostWrite("in", in);
  graph.createHostWrite("out", out);
  graph.createHostRead("out", out);
  graph.createHostWrite("weights", weights);

  prog.add(Execute(cs));

  // Populate the 'composite' vertex fields
  auto loadElems = LOAD_SIZE / inTypeSize;
  SignedType transformedInStride =
      (t.inStride - 1) * inChansPerGroup / loadElems + 1;
  unsigned strideAdj = (t.accumType == FLOAT) ? -6 : -4;
  if (t.numConvUnits > 8) {
    strideAdj -= 8;
  }
  SignedType transformedOutStride =
      strideAdj + (t.flipOut ? -t.outChansPerGroup : t.outChansPerGroup);

  graph.setInitialValue(v["numConvGroupsM1"], t.numConvGroups - 1);
  graph.setInitialValue(v["numOutGroupsM1"], t.numOutGroups - 1);
  graph.setInitialValue(v["numInGroups"], t.numInGroups);
  graph.setInitialValue(v["transformedInStride"], transformedInStride);
  graph.setInitialValue(v["outChansPerGroup"], t.outChansPerGroup);
  graph.setInitialValue(v["transformedOutStride"], transformedOutStride);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);

  Engine e(graph, prog);
  device.bind([&](const Device &d) {
    // A couple of local functions to write/read tensors with the appropriate
    // conversions between HALF and FLOAT, if necessary.
    auto writeTensorData = [&](const Type &type, const std::string &name,
                               const boost::multi_array<float, 2> &buf) {
      if (type == FLOAT) {
        e.writeTensor(name, buf.data(), buf.data() + buf.num_elements());
      } else if (type == HALF) {
        unsigned nBytes = target.getTypeSize(HALF) * buf.num_elements();
        std::vector<char> tmpBuf(nBytes);
        copy(target, buf.data(), buf.num_elements(), type, tmpBuf.data());
        e.writeTensor(name, tmpBuf.data(), tmpBuf.data() + nBytes);
      }
    };
    auto readTensorData = [&](const Type &type, const std::string &name,
                              boost::multi_array<float, 2> &buf) {
      if (type == FLOAT) {
        e.readTensor(name, buf.data(), buf.data() + buf.num_elements());
      } else if (type == HALF) {
        unsigned nBytes = target.getTypeSize(HALF) * buf.num_elements();
        std::vector<char> tmpBuf(nBytes);
        e.readTensor(name, tmpBuf.data(), tmpBuf.data() + nBytes);
        copy(target, type, tmpBuf.data(), buf.data(), buf.num_elements());
      }
    };
    e.load(d);
    writeTensorData(hostInType, "in", inBuf);
    writeTensorData(hostInType, "weights", weightsBuf);
    writeTensorData(t.accumType, "out", outBuf);
    e.run();
    readTensorData(t.accumType, "out", outBuf);
  });

  vertexCompute(numContexts, target.getTypeSize(t.inType),
                target.getTypeSize(t.accumType), inChansPerGroup, inBuf,
                weightsBuf, outHost, t);

  compareResults(outBuf, outHost, t.verificationTolerance);
}

// ====================== FLOAT FLOAT ======================
BOOST_AUTO_TEST_SUITE(float_float)
BOOST_AUTO_TEST_CASE(t1) {
  testParams t;
  t.inType = FLOAT;
  t.accumType = FLOAT;
  t.fieldWidth = 4;
  t.numConvGroups = 1;
  t.numOutGroups = 1;
  t.numInGroups = 1;
  t.outChansPerGroup = 8;
  // Worklist 'outOffs', 'inOffs' are in units of 2 float values,
  // 'numFieldElems' is in units of outChansPerGroup (8 float values)
  t.worklists = {{0, 2, 0}, {8, 2, 8}, {0, 0, 0},
                 {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  runTest(t);
}
BOOST_AUTO_TEST_CASE(t2) {
  testParams t;
  t.inType = FLOAT;
  t.accumType = FLOAT;
  t.fieldWidth = 4;
  t.numConvGroups = 1;
  t.numOutGroups = 2;
  t.numInGroups = 1;
  t.outChansPerGroup = 8;
  t.flipOut = true;
  // Worklist 'outOffs', 'inOffs' are in units of 2 float values,
  // 'numFieldElems' is in units of outChansPerGroup (8 float values)
  t.worklists = {{0, 1, 12}, {12, 3, 0}, {0, 0, 0},
                 {0, 0, 0},  {0, 0, 0},  {0, 0, 0}};
  runTest(t);
}
BOOST_AUTO_TEST_CASE(t3) {
  testParams t;
  t.inType = FLOAT;
  t.accumType = FLOAT;
  t.fieldWidth = 7;
  t.numConvGroups = 1;
  t.numOutGroups = 1;
  t.numInGroups = 1;
  t.outChansPerGroup = 16;
  t.flipOut = true;
  // Worklist 'outOffs', 'inOffs' are in units of 2 float values,
  // 'numFieldElems' is in units of outChansPerGroup (16 float values)
  t.worklists = {{48, 1, 0},  {40, 1, 4},  {32, 1, 8},
                 {24, 1, 12}, {16, 1, 16}, {8, 2, 20}};
  runTest(t);
}
BOOST_AUTO_TEST_CASE(t4) {
  testParams t;
  t.inType = FLOAT;
  t.accumType = FLOAT;
  t.fieldWidth = 4;
  t.numConvGroups = 1;
  t.numOutGroups = 1;
  t.numInGroups = 1;
  t.outChansPerGroup = 8;
  t.flipOut = true;
  // Worklist 'outOffs', 'inOffs' are in units of 2 float values,
  // 'numFieldElems' is in units of outChansPerGroup (8 float values)
  t.worklists = {{0, 0, 0}, {0, 0, 0}, {12, 2, 8},
                 {0, 0, 0}, {0, 0, 0}, {4, 2, 0}};
  runTest(t);
}
BOOST_AUTO_TEST_SUITE_END()

// ====================== HALF HALF ======================
BOOST_AUTO_TEST_SUITE(half_half)
BOOST_AUTO_TEST_CASE(t1) {
  testParams t;
  t.inType = HALF;
  t.accumType = HALF;
  t.verificationTolerance = 0.005;
  t.fieldWidth = 7;
  t.numConvGroups = 2;
  t.numOutGroups = 2;
  t.numInGroups = 1;
  t.outChansPerGroup = 8;
  t.flipOut = true;
  // Worklist 'outOffs', 'inOffs' are in units of 4 half values,
  // 'numFieldElems' is in units of outChansPerGroup (8 float values)
  t.worklists = {{0, 1, 24}, {2, 1, 20}, {4, 1, 16},
                 {6, 1, 12}, {8, 1, 8},  {12, 2, 0}};
  runTest(t);
}
BOOST_AUTO_TEST_SUITE_END()

// ====================== HALF FLOAT ======================
BOOST_AUTO_TEST_SUITE(half_float)
BOOST_AUTO_TEST_CASE(t1) {
  testParams t;
  t.inType = HALF;
  t.accumType = FLOAT;
  t.verificationTolerance = 0.005;
  t.fieldWidth = 9;
  t.numConvGroups = 2;
  t.numOutGroups = 2;
  t.numInGroups = 1;
  t.outChansPerGroup = 8;
  // Worklist 'outOffs' are in units of 2 float values,
  // 'inOffs' are in units of 4 half values,
  // 'numFieldElems' is in units of outChansPerGroup (8 float values)
  t.worklists = {{0, 1, 0},   {4, 1, 4},   {8, 1, 8},
                 {12, 2, 12}, {20, 2, 20}, {28, 2, 28}};
  runTest(t);
}
BOOST_AUTO_TEST_SUITE_END()

// ====================== HALF HALF DUAL ======================
BOOST_AUTO_TEST_SUITE(half_half_dual, *boost::unit_test::precondition(isIpu2))
BOOST_AUTO_TEST_CASE(t1) {
  testParams t;
  t.inType = HALF;
  t.accumType = HALF;
  t.numConvUnits = 16;
  t.verificationTolerance = 0.005;
  t.fieldWidth = 7;
  t.numConvGroups = 1;
  t.numOutGroups = 1;
  t.numInGroups = 1;
  t.outChansPerGroup = 16;
  t.flipOut = true;
  // Worklist 'outOffs', 'inOffs' are in units of 4 half values,
  // 'numFieldElems' is in units of outChansPerGroup (8 float values)
  t.worklists = {{0, 1, 24},  {4, 1, 20}, {8, 1, 16},
                 {12, 1, 12}, {16, 1, 8}, {24, 2, 0}};
  runTest(t);
}
BOOST_AUTO_TEST_SUITE_END()

// ====================== QUARTER HALF ======================
BOOST_AUTO_TEST_SUITE(quarter_half, *boost::unit_test::precondition(isIpu21))
BOOST_AUTO_TEST_CASE(t1) {
  testParams t;
  t.inType = QUARTER;
  t.accumType = HALF;
  t.verificationTolerance = 0.005;
  t.fieldWidth = 256;
  t.numConvGroups = 1;
  t.numOutGroups = 2;
  t.numInGroups = 2;
  t.outChansPerGroup = 16;
  t.flipOut = false;
  t.numConvUnits = 16;

  // Worklist 'outOffs' are in units of 4 half values,
  // 'inOffs' are in units of 8 quarter values,
  // 'numFieldElems' is in units of outChansPerGroup (16 half values)
  t.worklists = {{0, 1, 0}, {4, 1, 16}, {8, 2, 32},
                 {0, 0, 0}, {0, 0, 0},  {0, 0, 0}};
  runTest(t);
}
BOOST_AUTO_TEST_CASE(t2) {
  testParams t;
  t.inType = QUARTER;
  t.accumType = HALF;
  t.verificationTolerance = 0.005;
  t.fieldWidth = 256;
  t.numConvGroups = 1;
  t.numOutGroups = 2;
  t.numInGroups = 2;
  t.outChansPerGroup = 16;
  t.flipOut = false;
  t.numConvUnits = 16;
  t.use128BitLoad = false;

  // Worklist 'outOffs' are in units of 4 half values,
  // 'inOffs' are in units of 8 quarter values,
  // 'numFieldElems' is in units of outChansPerGroup (16 half values)
  t.worklists = {{0, 1, 0}, {4, 1, 16}, {8, 2, 32},
                 {0, 0, 0}, {0, 0, 0},  {0, 0, 0}};
  runTest(t);
}

BOOST_AUTO_TEST_CASE(t3) {
  testParams t;
  t.inType = QUARTER;
  t.accumType = HALF;
  t.verificationTolerance = 0.005;
  t.fieldWidth = 32;
  t.numConvGroups = 1;
  t.numOutGroups = 1;
  t.numInGroups = 1;
  t.outChansPerGroup = 16;
  t.flipOut = true;
  t.numConvUnits = 16;
  t.use128BitLoad = true;
  t.inStride = 2;

  // Worklist 'outOffs' are in units of 4 half values,
  // 'inOffs' are in units of 8 quarter values,
  // 'numFieldElems' is in units of outChansPerGroup (16 half values)
  t.worklists = {{4, 2, 16}, {0, 0, 0}, {0, 0, 0},
                 {0, 0, 0},  {0, 0, 0}, {0, 0, 0}};
  runTest(t);
}

BOOST_AUTO_TEST_CASE(t4) {
  testParams t;
  t.inType = QUARTER;
  t.accumType = HALF;
  t.verificationTolerance = 0.005;
  t.fieldWidth = 64;
  t.numConvGroups = 1;
  t.numOutGroups = 1;
  t.numInGroups = 1;
  t.outChansPerGroup = 16;
  t.numConvUnits = 16;
  t.use128BitLoad = true;
  t.inStride = 1;
  t.weightFp8Scale = -1;
  t.inputFp8Scale = 2;
  t.weightFp8Format = Fp8Format::QUART152;

  // Worklist 'outOffs' are in units of 4 half values,
  // 'inOffs' are in units of 8 quarter values,
  // 'numFieldElems' is in units of outChansPerGroup (16 half values)
  t.worklists = {{0, 2, 0}, {8, 2, 16}, {0, 0, 0},
                 {0, 0, 0}, {0, 0, 0},  {0, 0, 0}};
  runTest(t);
}

BOOST_AUTO_TEST_SUITE_END()
