// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popops_codelets_CodeletsTestCommon_hpp
#define popops_codelets_CodeletsTestCommon_hpp

// Definitions/declarations used in elementwise unary/binary operation test
// code.

#include <poplar/Type.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/TempDir.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>

#include <gccs/Algorithm.hpp>

#include <pva/pva.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/token_functions.hpp>
#include <boost/tokenizer.hpp>

#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poplibs_support;
using namespace poplibs_test::util;
using namespace poplar_test;

using boost::format;
using std::to_string;

namespace po = boost::program_options;

static const std::string estimatedStr = "estimated";

// Overloaded & templated convertToString functions to print correctly
// from inside the templated 'verifyResult()'  function

template <typename IntType> std::string intToHexStr(IntType val) {
  std::stringstream ss;
  ss << val << " (0x" << std::hex << val << ")";
  return ss.str();
}

std::string convertToString(int val) { return intToHexStr(val); }

std::string convertToString(unsigned val) { return intToHexStr(val); }

std::string convertToString(short val) { return intToHexStr(val); }

std::string convertToString(unsigned short val) { return intToHexStr(val); }

std::string convertToString(char val) {
  std::stringstream ss;
  int ival = val;
  ss << ival << " (0x" << std::hex << (ival & 0xff) << ")";
  return ss.str();
}

std::string convertToString(signed char val) {
  return convertToString(char(val));
}

std::string convertToString(unsigned char val) {
  return intToHexStr(unsigned(val));
}

template <typename T> std::string convertToString(T val) {
  std::stringstream ss;
  ss << val;
  return ss.str();
}

// Prints the specified buffer containing one operand
template <typename HostType>
void printBuffer(const std::string &name, const std::vector<HostType> &buf,
                 const Type IPUType, const std::vector<unsigned> &sizes,
                 const std::vector<unsigned> &rowOffsets) {
  unsigned nRows = sizes.size();
  std::cout << name << ": {" << ((nRows > 1) ? "\n" : "");
  for (unsigned row = 0; row < nRows; row++) {
    std::cout << ((nRows > 1) ? "{" : "");
    unsigned n = sizes[row];
    for (unsigned i = 0; i < n; i++) {
      HostType val = buf.at(rowOffsets[row] + i);
      if (IPUType == BOOL) {
        std::cout << int(val);
      } else {
        std::cout << convertToString(val);
      }
      if (i < n - 1) {
        std::cout << ", ";
      }
    }
    std::cout << ((nRows > 1) ? "},\n" : "");
  }
  std::cout << "}\n";
}

// A 'random generator' engine which might not be there (in which case it means
// that we are not using random values to fill the data buffers).
using RandomEngine = std::optional<std::minstd_rand>;

// Filling one of the operand buffers for 32 bit integer data.
void fillBufferInt(const Type &dataType, RandomEngine &rndEng, int *data,
                   unsigned n, int i, int min, int max, bool nonZero) {
  std::uniform_int_distribution<int> d(min, max);
  for (unsigned k = 0; k < n; k++) {
    if (rndEng) {
      do {
        data[k] = d(*rndEng);
      } while (nonZero && data[k] == 0);
    } else {
      if (max != 0 && i > max)
        i = 0;
      data[k] = i++;
    }
  }
}

void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<int> &buf, int i, int min, int max, bool nonZero) {
  fillBufferInt(dataType, rndEng, buf.data(), buf.size(), i, min, max, nonZero);
}

void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<unsigned> &buf, unsigned i, unsigned min,
                unsigned max, bool nonZero) {
  // The 'int' filling is good for 'unsigned' as well
  fillBufferInt(dataType, rndEng, (int *)(buf.data()), buf.size(), i, min, max,
                nonZero);
}

// Filling one of the operand buffers for 32 bit integer data.
void fillBufferLongLong(const Type &dataType, RandomEngine &rndEng,
                        long long *data, unsigned n, long long i, long long min,
                        long long max, bool nonZero) {
  std::uniform_int_distribution<long long> d(min, max);
  for (unsigned k = 0; k < n; k++) {
    if (rndEng) {
      do {
        data[k] = d(*rndEng);
      } while (nonZero && data[k] == 0);
    } else {
      if (max != 0 && i > max)
        i = 0;
      data[k] = i++;
    }
  }
}

void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<long long> &buf, long long i, long long min,
                long long max, bool nonZero) {
  fillBufferLongLong(dataType, rndEng, buf.data(), buf.size(), i, min, max,
                     nonZero);
}

void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<unsigned long long> &buf, unsigned long long i,
                unsigned long long min, unsigned long long max, bool nonZero) {
  // The 'long long' filling is good for 'unsigned long long' as well
  fillBufferLongLong(dataType, rndEng, (long long *)(buf.data()), buf.size(), i,
                     min, max, nonZero);
}

// Filling one of the operand buffers for 16 bit integer data.
void fillBufferShort(const Type &dataType, RandomEngine &rndEng, short *data,
                     unsigned n, short i, short min, short max, bool nonZero) {
  std::uniform_int_distribution<short> d(min, max);
  for (unsigned k = 0; k < n; k++) {
    if (rndEng) {
      do {
        data[k] = d(*rndEng);
      } while (nonZero && data[k] == 0);
    } else {
      if (max != 0 && i > max)
        i = 0;
      data[k] = i++;
    }
  }
}
void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<short> &buf, short i, short min, short max,
                bool nonZero) {
  fillBufferShort(dataType, rndEng, buf.data(), buf.size(), i, min, max,
                  nonZero);
}

void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<unsigned short> &buf, unsigned short i,
                unsigned short min, unsigned short max, bool nonZero) {
  // The 'short' filling is good for 'unsigned short' as well
  fillBufferShort(dataType, rndEng, (short *)(buf.data()), buf.size(), i, min,
                  max, nonZero);
}

// Filling one of the operand buffers for 8 bit integer data.
void fillBufferChar(const Type &dataType, RandomEngine &rndEng, char *data,
                    unsigned n, char i, char min, char max, bool nonZero) {
  std::uniform_int_distribution<char> d(min, max);
  for (unsigned k = 0; k < n; k++) {
    if (rndEng) {
      do {
        data[k] = d(*rndEng);
      } while (nonZero && data[k] == 0);
    } else {
      if (max != 0 && i > max)
        i = 0;
      data[k] = i++;
    }
  }
}
void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<char> &buf, char i, char min, char max,
                bool nonZero) {
  fillBufferChar(dataType, rndEng, buf.data(), buf.size(), i, min, max,
                 nonZero);
}
void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<signed char> &buf, signed char i, signed char min,
                signed char max, bool nonZero) {
  fillBufferChar(dataType, rndEng, (char *)(buf.data()), buf.size(), i, min,
                 max, nonZero);
}
void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<unsigned char> &buf, unsigned char i,
                unsigned char min, unsigned char max, bool nonZero) {
  // This could be either boolean (0/1) or proper 'unsigned char' (0..255)
  if (dataType == BOOL) {
    std::bernoulli_distribution d(0.5);

    for (auto &x : buf)
      x = d(*rndEng);
  } else {
    fillBufferChar(dataType, rndEng, (char *)(buf.data()), buf.size(), i, min,
                   max, nonZero);
  }
}

// Filling one of the operand buffers for FLOAT, HALF or QUARTER data (all use
// 'float' buffers on the host)
void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<float> &buf, float i, float min, float max,
                bool nonZero,
                const std::optional<QuarterMetadata> metadata = std::nullopt) {
  std::uniform_real_distribution<float> d(min, max);
  if (dataType == QUARTER) {
    assert(metadata != std::nullopt);
    // Pick values that are exact for the FP8 format selected.
    // QUART 143 has 3 mantissa bits + 1 lead bit = 4 bits, supports 0..16
    // QUART 152 has 2 mantissa bits + 1 lead bit = 3 bits, supports 0..8
    const unsigned modulo =
        (metadata->getFormat() == QuarterMetadata::Format ::F143) ? 17 : 9;
    for (unsigned i = 0; i < buf.size(); i++) {
      buf[i] = (i + 1) % modulo;
    }
  } else {
    for (auto &x : buf) {
      if (rndEng) {
        do {
          x = d(*rndEng);
        } while (nonZero && x == 0);
      } else {
        if (i > max)
          i = 0;
        x = i;
        i += 1.0;
      }
    }
  }
}

// This contains the size of an operand for a UnaryOp, Cast or BinaryOp test.
// It can contain:
//
//   1. A single value (for a 1D/Supervisor vertex). In this case:
//      isRowsByCols==false, val[] has a single element
//
//   2. Multiple values, for a 2D vertex. Each value represent the length of the
//      2D 'row' (subvector). In this case:
//      isRowsByCols==false, val[] has multiple elements
//
//   3. A 'ROW x COL' value (for a VectorOuter vertex or a 2D vertex):
//      sRowsByCols==true, val[] has exactly two values (rows, cols)
struct SizeDesc {
  bool isRowsByCols = false;
  std::vector<unsigned> val;
  // Return a new DescSize adjusted for a 'standard' vertex that can be only
  // 1D or 2D. Note that the Broadcast/VectorInner/VectorOuter have a more
  // complex bespoke 'adjustment' of the command-line-specified size.
  SizeDesc adjust(bool vertexIs2D) const {
    SizeDesc adjusted = *this;
    if (isRowsByCols) {
      // If specified on the command line as "NxM", for instance "7x5", we will
      // change it:
      //  If 2D into:  [7,7,7,7,7]
      //  If 1D into:  [35]
      adjusted.isRowsByCols = false;
      unsigned rows = val.at(0);
      unsigned cols = val.at(1);
      adjusted.val.clear();
      if (vertexIs2D) {
        adjusted.val.resize(rows, cols);
      } else {
        adjusted.val.push_back(rows * cols);
      }
    } else {
      // Otherwise, if it the vertex is 1D, we keep only the first size
      // (for example, if we start with [21,12,5], we change into [21]).
      // Dropping silently the 'extra' row sizes make it easier to test both
      // 1D and 2D vertices with a single command.
      if (!vertexIs2D) {
        adjusted.val.resize(1);
      }
    }
    return adjusted;
  }
  std::string toString(unsigned minWidth = 0) const {
    std::string s;
    unsigned n = val.size();
    if (isRowsByCols) {
      if (n != 2)
        throw std::runtime_error("Invalid SizeDesc: 'isRowsByCols' is set, but "
                                 "'val' has " +
                                 to_string(n) + "element (instead of 2)");
      s = to_string(val.at(0)) + "x" + to_string(val.at(1));
    } else {
      s = "[";
      for (unsigned i = 0; i < n - 1; i++)
        s += to_string(val[i]) + ",";
      if (n > 0)
        s += to_string(val[n - 1]);
      s += "]";
    }
    int padLen = minWidth - s.size();
    if (padLen > 0)
      s += std::string(padLen, ' ');
    return s;
  }
};

// Utility function to read a SizeDesc from a text stream. Three formats are
// valid in the input text:
//
//   "1234"          - A single positive integer value
//
//   "[11,22,33,44]" - A list of integers, comma separated, with square brackets
//
//   "666x333"       - A 'ROW x COL' value (using the "x" character)
//
std::istream &operator>>(std::istream &in, SizeDesc &sd) {
  skipSpaces(in);
  auto c = in.peek();
  if (c == '[') {
    // If it starts with '[' must be a comma separated list with square brackets
    in.ignore();
    skipSpaces(in);
    auto c = in.peek();
    if (c == ']') {
      in.ignore();
    } else {
      while (true) {
        sd.val.push_back(poplibs_test::util::detail::readValue<unsigned>(in));
        skipSpaces(in);
        c = in.get();
        if (c == ']') {
          skipSpaces(in);
          break;
        } else if (c != ',') {
          throw std::runtime_error(
              "Invalid size descriptor; expected ',' or ']'");
        }
        skipSpaces(in);
      }
    }
    return in;
  } else {
    // If it doesn't start with '[' must be a single number or <row>x<col>
    if (!std::isdigit(c)) {
      throw std::runtime_error(
          "Invalid size descriptor; expected '[' or digit");
    }
    sd.val.push_back(poplibs_test::util::detail::readValue<unsigned>(in));
    skipSpaces(in);
    in.clear();
    c = in.peek();
    if (c == 'x') {
      sd.isRowsByCols = true;
      in.ignore();
      skipSpaces(in);
      if (!std::isdigit(in.peek())) {
        throw std::runtime_error("Invalid size descriptor; expected a digit");
      }
      sd.val.push_back(poplibs_test::util::detail::readValue<unsigned>(in));
      in.clear();
      skipSpaces(in);
    }
  }
  return in;
}

// Printing a SizeDesc to a stream
std::ostream &operator<<(std::ostream &os, const SizeDesc &sd) {
  os << sd.toString();
  return os;
}

// Returns the smallest divisor of n. If n is prime returns 1
unsigned findDivisor(unsigned n) {
  unsigned max = sqrt(n);
  if ((n % 2) == 0) {
    return 2;
  } else {
    for (unsigned d = 3; d < max; d += 2) {
      if ((n % d) == 0) {
        return d;
      }
    }
    return 1;
  }
}

// Used to pass the appropriate streams for upload/download between the setup
// and running phases.
using StreamMap = std::vector<std::pair<std::string, char *>>;

// Various options that are global (not specific for each test)
struct MiscOptions {
  // Do we want to print a poplar report on stdout?
  bool report = false;

  // If true, floating point exceptions are disabled (using
  // "debug.floatPointOpException")
  bool disableFpExceptions = false;

  // Do we want to skip verification of results?
  bool ignoreData = false;

  // Do we want to print on stdout the values in the input and output buffers?
  bool printBuffers = false;

  // The random seed for data; '0' means 'not random' (might not work with
  // some operands!)
  unsigned randomSeed = 1;

  // How many test to run in a single graph/compute set (one per tile)
  unsigned groupTests = 0;

  // Maximum number of tiles to use for a single graph/run of tests.
  // NOTE: Default was set empirically to reduce test times.
  unsigned numTiles = 0;

  // print the number of cycles used by the vertex.
  bool printCycles = false;

  // If not null, we want to compare the execution cycles between the 'current'
  // device and the device whose name is specified here
  boost::optional<std::string> cycleCompareDevice;

  // Percent of (absolute value of) difference above which a cycle comparison
  // test will fail (if cycleCompareDevice is not null))
  float cycleCompareThreshold = 10;

  // The start of inputs and outputs variables is normally aligned on an 8 byte
  // boundary (by using the NopAlignVertex). If one or more non-zero values
  // (1..7) are specified here, we will add this many bytes to the start so
  // that the alignment will be to 8 byte boundary + N. This value N must be a
  // multiple of the natural size of the variable (for instance 2 for 'half'),
  // if not it will be adjusted.
  // If multiple values are specified, we test them all.
  std::vector<unsigned> startPadBytes = {0};

  QuarterMetadata::Format fp8Format = QuarterMetadata::Format::F143;
  int fp8Scale = 0;
};

//*************************************************************************
/// Create a tensor in a graph, creatiign also the metadata if necessary,
/// based on 'options'
Tensor createTensor(Graph &graph, const Type &dataType,
                    const MiscOptions &options, unsigned numElems,
                    const std::string &name) {
  if (dataType == QUARTER) {
    auto metadata = poputil::createVariableMetadataTensor(
        graph, options.fp8Format, options.fp8Scale);
    return graph.addVariable(dataType, metadata, {numElems}, name);
  } else {
    return graph.addVariable(dataType, {numElems}, name);
  }
}

std::string formatStartPadBytes(unsigned startPadBytes) {
  if (startPadBytes == 0) {
    return "";
  } else {
    return (format("  offs:%s") % startPadBytes).str();
  }
}

//*************************************************************************
/// Runs the tests specified by 'tests[offs:offs+numTests]', all in a single
/// compute set.
///
/// \param[inout] tests    A vector containing the tests to run. Note that this
///                        function will modify the test records that are run,
///                        adding the data and output buffers.
/// \param[in]    offs     Offset into 'tests' specifying the first test to run
/// \param[in]    numTests   How many tests from 'tests[offs]' we want to run
/// \param[in]    deviceType What type of device to run the tests on
/// \param[in]    options    Global options.
/// \param[in]    verbose    Do we want to print a description of each test as
///                          it is setup/verified?
/// \param[out]   cycles     If requested in 'options', will be populated with
///                          the cycles spent running the vertex.
/// \param[out]   estimatedCycles If requested in 'options', will be populated
///                               with the cycles estimation for the vertex.
/// \return   number of FAILED tests.
template <typename TestRecord>
unsigned runTests(std::vector<std::shared_ptr<TestRecord>> &tests,
                  unsigned offs, unsigned numTests,
                  const DeviceType &deviceType, TestDevice &device,
                  const Target &target, const MiscOptions &options,
                  bool verbose, uint64_t &cycles, uint64_t &estimatedCycles) {
  // The name of the compute set where we run the vertex under test.
  const static std::string computeSetName = "vertexComputeSet";

  Graph graph(target);
  popops::addCodelets(graph);
  ComputeSet alignCS = graph.addComputeSet("alignCS");
  ComputeSet cs = graph.addComputeSet(computeSetName);
  Sequence program, upload, download;
  bool ipuModel = isIpuModel(deviceType);

  StreamMap streamMap;
  for (unsigned i = 0; i < numTests; i++) {
    // If running only one test, we print the info about that test before
    // doing the setup. This will help debug any problem arising during the
    // setup itself. We also do the same in the case we are not running the
    // data verification.
    if (verbose && (numTests == 1 || options.ignoreData)) {
      std::cout << tests[offs + i]->toString() << std::endl;
    }
    doSetupTest(target, ipuModel, graph, upload, alignCS, cs, download,
                streamMap, *tests[offs + i], i % target.getNumTiles(), options);
  }
  program.add(Execute(alignCS));
  program.add(Execute(cs));

  // === Run the program
  OptionFlags engOpts;
  bool reportCycles = options.printCycles || options.cycleCompareDevice;
  std::optional<TempDir> tempDir;
  if (options.report || reportCycles) {
    tempDir.emplace(TempDir::create());
    engOpts.set("autoReport.outputExecutionProfile", "true");
    engOpts.set("autoReport.directory", tempDir->getPath());
  }
  if (options.cycleCompareDevice == estimatedStr) {
    engOpts.set("profiler.includeCycleEstimates", "true");
  }
  if (options.disableFpExceptions) {
    engOpts.set("debug.floatPointOpException", "false");
  }
  Engine engine(graph, Sequence{upload, program, download}, engOpts);
  attachStreams(engine, streamMap);

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run();

    if (options.report) {
      OptionFlags opt;
      opt.set("showExecutionSteps", "true");
      engine.printProfileSummary(std::cout, opt);
    }
  });
  unsigned errCount = 0;
  if (options.ignoreData) {
    std::cout << "Result not checked for correctness\n";
  } else {
    for (unsigned i = 0; i < numTests; i++) {
      // If we are running grouped tests, we print the test description before
      // verification of each test. This helps finding out which specific test
      // might have failed
      if (verbose && numTests > 1) {
        std::cout << tests[offs + i]->toString() << std::endl;
      }

      if (!doVerifyTest(target, ipuModel, *tests[offs + i], options))
        errCount++;
    }
  }
  if (reportCycles) {
    const pva::Report &report = engine.getReport();

    // Get the cycles by searching the "steps"/"computeSets" vector for a
    // compute set having the name we used.
    for (auto const &s : report.execution().steps()) {
      auto const &computeSets = s.computeSets();
      for (auto const &cs : computeSets) {
        if (cs.name() == computeSetName) {
          auto cyc = cs.cyclesByTile();
          cycles = (cyc.size() > 0) ? cyc[0] : 0;
          auto est = cs.estimatedCyclesByTile();
          estimatedCycles = (est.size() > 0) ? est[0] : 0;
        }
      }
    }
  }
  return errCount;
}

//*************************************************************************
// Creates the device, possibly adjusting the number of tiles and group
// size inside 'options'.
TestDevice setupDevice(const DeviceType deviceType, MiscOptions &options) {
  // Default values for Sim and IpuModel
  const unsigned DEFAULT_TILES = 4;
  const unsigned DEFAULT_GROUP_SIZE = 400;
  boost::optional<unsigned> requestedTilesPerIPU;
  bool readNumTiles = false;
  bool setGroupFromTiles = false;

  if (isHw(deviceType)) {
    // If it's an HW device and no specfic number of tiles/group size has
    // been specified, we want to use all tiles (it's faster)
    if (options.numTiles == 0) {
      readNumTiles = true;
      setGroupFromTiles = (options.groupTests == 0);
    }
  } else {
    if (options.numTiles == 0) {
      options.numTiles = DEFAULT_TILES;
    }
    requestedTilesPerIPU = options.numTiles;
  }
  if (!setGroupFromTiles && options.groupTests == 0) {
    options.groupTests = DEFAULT_GROUP_SIZE;
  }

  TestDevice device = createTestDevice(deviceType, 1, requestedTilesPerIPU);

  // If needed, now we can find out how many tiles are there on the device
  if (readNumTiles) {
    options.numTiles = device.getTarget().getTilesPerIPU();
  }
  if (setGroupFromTiles) {
    options.groupTests = options.numTiles;
  }
  return device;
}

//*************************************************************************
/// Compare the cycles used when running the vertex with the two specified
/// devices. Prints result on standard output.
///
/// \param[in] deviceType          The 'main' device for which to run the test
/// \param[in] test                Test record defining the specific vertex and
///                                operand(s) size.
/// \param[in] options             Global options.
///
/// \return true   if both run returned successfully and the difference is less
///                than 'compareThreshold' % of the run with the first device.
template <typename TestRecord, typename VertexDesc>
bool compareCycles(DeviceType deviceType, std::shared_ptr<TestRecord> test,
                   MiscOptions &options) {
  VertexDesc &vertex = *test->vertex;

  std::stringstream devName[2]; // Strings with the name of selected devices
  std::cout << vertex.vClassFmt << std::flush;

  bool ok[2];
  uint64_t cycles[2] = {0, 0};

  std::vector<std::shared_ptr<TestRecord>> testVect = {test};

  // If was explicitly requested to compare with the "estimated" cycles, we can
  // just run the test once, and get both values.
  if (options.cycleCompareDevice == estimatedStr) {
    devName[0] << deviceType;
    devName[1] << "estimated";
    TestDevice device = setupDevice(deviceType, options);
    ok[0] = ok[1] = (runTests<TestRecord>(testVect, 0, 1, deviceType, device,
                                          device.getTarget(), options, false,
                                          cycles[0], cycles[1]) == 0);
  } else {
    // If an explicit device was specified for the comparison, run with the two
    // devices and get the cycles from both.
    DeviceType compDeviceType;
    std::istringstream is(*options.cycleCompareDevice);
    is >> compDeviceType;
    DeviceType deviceTypes[2] = {deviceType, compDeviceType};
    for (unsigned i = 0; i < 2; i++) {
      devName[i] << deviceTypes[i];
      uint64_t cyc = 0;
      uint64_t estimatedCyc = 0;
      TestDevice device = setupDevice(deviceTypes[i], options);
      ok[i] = runTests<TestRecord>(testVect, 0, 1, deviceTypes[i], device,
                                   device.getTarget(), options, false, cyc,
                                   estimatedCyc) == 0;
      if (!ok[i]) {
        std::cout << "Failed on device " << devName[i].str()
                  << " (see stderr)\n";
        return false;
      }
      cycles[i] = isIpuModel(deviceTypes[i]) ? estimatedCyc : cyc;
    }
  }

  float diff = static_cast<float>(cycles[1]) - static_cast<float>(cycles[0]);
  float diffPerc = diff / cycles[0] * 100;
  bool compareOk = abs(diffPerc) < options.cycleCompareThreshold;
  std::cout << format("%s:%8u;  %s:%8u;   diff =%4u  %7.2f%%%s\n") %
                   devName[0].str() % cycles[0] % devName[1].str() % cycles[1] %
                   diff % diffPerc % (compareOk ? "" : " <<== FAIL");
  return compareOk;
}

//*************************************************************************
/// Add one test record to the array of tests to perform, or execute it
/// straight away (if required)
///
/// \param[inout] tests      A vector where to add the test. Could be 'nullopt'
///                          if we need to run the test straight away
/// \param[in]    newTest    The new tests to add (or run)
/// \param[in]    deviceType Device to run on
/// \param[inout] errCount   If the test is run immediately, will be increased
///                          by one if the test fails
/// \param[in]    options    Global options.
template <typename TestRecord, typename VertexDesc>
void addOneTest(std::vector<std::shared_ptr<TestRecord>> &tests,
                std::shared_ptr<TestRecord> newTest, DeviceType deviceType,
                unsigned &errCount, MiscOptions &options) {
  // If:
  //   1. we are running a cycle comparison, or
  //   2. we have specfied to run test individually (groupTests == 1), or
  //   3. we want to print the vertex execution cycles
  // then we run the new test straight away here,
  // otherwise we add this test record to 'tests[]', to be run afterwards.
  if (options.cycleCompareDevice) {
    errCount +=
        compareCycles<TestRecord, VertexDesc>(deviceType, newTest, options) ? 0
                                                                            : 1;
  } else if (options.groupTests == 1 || options.printCycles) {
    std::vector<std::shared_ptr<TestRecord>> testVect = {newTest};
    uint64_t cycles;
    uint64_t estimatedCycles;
    bool verbose = !options.printCycles;
    TestDevice device = setupDevice(deviceType, options);
    errCount += runTests(testVect, 0, 1, deviceType, device, device.getTarget(),
                         options, verbose, cycles, estimatedCycles);
    if (options.printCycles) {
      std::cout << newTest->toString() << "  cycles:"
                << (isIpuModel(deviceType) ? estimatedCycles : cycles)
                << std::endl;
    }
  } else {
    tests.emplace_back(std::move(newTest));
  }
}

//*************************************************************************
/// Run all tests that have been accumulated in 'tests[]' (if any), or do
/// nothing if the test have already been run individually
/// \param[inout] tests      A vector with the tests to run.
/// \param[in]    numTests   How many tests are in 'tests[]'. If 'tests' is
///                          empty, this is the total tests run already.
/// \param[in]    deviceType What type of device to run the tests on
/// \param[inout] errCount   Will be updated with the number of failed tests.
/// \param[in]    options    Global options.
///
template <typename TestRecord>
void runAllTests(std::vector<std::shared_ptr<TestRecord>> &tests,
                 unsigned numTests, DeviceType deviceType, unsigned &errCount,
                 MiscOptions &options) {
  if (numTests == 0) {
    throw std::runtime_error("The specified vertex, operand(s) and data "
                             "type(s) do not match any valid combination");
  }
  if (tests.size() > 0) {
    assert(numTests == tests.size());
    // Run the tests in batches of up to 'groupTests' together on a single
    // graph/ single CS, each test on a different tile. We limit grouping tests
    // on multiple tile to 'groupTests' because when running on the simulators,
    // if too many tiles are used, execution is slower.
    TestDevice device = setupDevice(deviceType, options);
    unsigned offs = 0, n = numTests;
    while (n) {
      uint64_t cycles;
      uint64_t estimatedCycles;
      unsigned l = n > options.groupTests ? options.groupTests : n;
      errCount +=
          runTests(tests, offs, l, deviceType, device, device.getTarget(),
                   options, true, cycles, estimatedCycles);
      offs += l;
      n -= l;
    }
  }
  if (numTests > 1) {
    std::cout << numTests << " codelets tests run in total; "
              << ((errCount == 0) ? "All passed\n"
                                  : to_string(errCount) + " failed\n");
  }
}

// Just a way to have a common fatal error message when we select an operation
// where we have a data+output type pair that is not supported.
class invalid_types : public std::runtime_error {
public:
  invalid_types(const Type &src, const Type &dst)
      : std::runtime_error("Combination of source type (" + src.toString() +
                           ") and destination type (" + dst.toString() +
                           ") not supported") {}
};

/// Add all command line options tha are common among all vertex test
/// executables.
void addCommonOptions(po::options_description &poDesc, DeviceType &deviceType,
                      MiscOptions &options) {
  // clang-format off
  poDesc.add_options()
    ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(DeviceType::Sim2),
     "Device type")
    ("align-start",
     po::value<std::vector<unsigned>>(&options.startPadBytes)->multitoken(),
     "Align all inputs and the output on a 8+N byte boundary. If the "
     "vertex requires stricter alignment then specified, setting this will "
     "cause rearrangements, and overwrite detection won't work")
    ("cycles",
     po::value<bool>(&options.printCycles)->implicit_value(true),
     "Print the number of cycles used by the vector")
    ("compare-cycles",
     po::value<boost::optional<std::string>>(&options.cycleCompareDevice)->
                                       implicit_value(std::string("estimated")),
     "For each specified vertex, compare the cycles reported by the device ("
     "--device-type option) and another device specified by this option")
    ("cycle-threshold",
     po::value<float>(&options.cycleCompareThreshold)->
                                  default_value(options.cycleCompareThreshold),
     "Percent threshold when running the --compare-cycle option. An (absolute) "
     "cycle difference greater than this threshold will make the test fail")
    ("report",
     po::value<bool>(&options.report)->implicit_value(true),
     "Provide a poplar report")
    ("disable-fp-exceptions",
     po::value<bool>(&options.disableFpExceptions)->
                                    default_value(options.disableFpExceptions),
     "Disable floating point exceptions when running on device")
    ("options-file",
     po::value<std::string>(),
     "A file containing options, with the same syntax as the command line; "
     "can be also specified with '@options_file_name'")
    ("random-seed",
     po::value<unsigned>(&options.randomSeed)->
                                            implicit_value(options.randomSeed),
     "Seed for random data. Value of 0 means 'don't use random data'")
    ("ignore-data",
     po::value<bool>(&options.ignoreData)->implicit_value(true),
     "Do not check correctness of result, useful for benchmarking without "
     "overhead of host-side computation")
    ("group-tests",
     po::value<unsigned>(&options.groupTests)->
                                             implicit_value(options.groupTests),
     "Multiple tests will be run together to increase speed. Set to 1 to run "
     "each test individually, to find which specific test is failing. Set to "
     "0 to automatically select the fastest group size for Hw, Sim, IpuModel")
    ("num-tiles",
     po::value<unsigned>(&options.numTiles)->default_value(options.numTiles),
     "Maximum number of tiles to use for one batch of tests")
    ("print-buffers",
     po::value<bool>(&options.printBuffers)->implicit_value(true),
     "Print the input and output buffers")
    ("fp8-scale",
     po::value<int>(&options.fp8Scale)->implicit_value(options.fp8Scale),
     "Exponent scale for fp8 type conversion")
    ("fp8-format",
     po::value<QuarterMetadata::Format >(&options.fp8Format)->
                                              implicit_value(options.fp8Format),
     "Format for fp8 type conversion")
    ;
  // clang-format on

  // Print a warning if these two options are both specified
  if (options.cycleCompareDevice && (options.groupTests > 1)) {
    std::cout << "When running with --compare-cycle option, the --group-tests "
                 "option is ignored\n";
  }
}

// Utility function to parse the command line options so that you can also
// specify an 'option' text file, using "--options-file <file-name>" or
// @<file-name>, that contains options.
void parseOptions(int argc, char **argv, po::options_description &desc) {
  po::variables_map vm;
  try {
    // Additional command line parser to interpret an argument '@filename' as
    // a option "config-file" with the value "filename"
    auto at_option_parser = [](std::string const &s) {
      if ('@' == s[0]) {
        return std::make_pair(std::string("options-file"), s.substr(1));
      } else {
        return std::pair<std::string, std::string>();
      }
    };

    po::store(po::command_line_parser(argc, argv)
                  .options(desc)
                  .extra_parser(at_option_parser)
                  .run(),
              vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      std::exit(0);
    }
    // If there is a file to read the options from, do it
    if (vm.count("options-file")) {
      std::string filename = vm["options-file"].as<std::string>();
      std::ifstream ifs(filename.c_str());
      if (!ifs) {
        throw std::runtime_error("Could not open options file <" + filename +
                                 ">");
      }
      // Read the whole file into a stringstream
      std::stringstream ss;
      ss << ifs.rdbuf();
      // Split the file content into tokens, using spaces/newlines/tabs
      boost::char_separator<char> sep(" \t\n\r");
      std::string sstr = ss.str();
      boost::tokenizer<boost::char_separator<char>> tok(sstr, sep);
      std::vector<std::string> args;
      std::copy(tok.begin(), tok.end(), back_inserter(args));
      // Parse the file and store the options
      po::store(po::command_line_parser(args).options(desc).run(), vm);
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    std::exit(1);
  }
}

// A descriptor to keep information about which tile to store a slice of
// a tensor on
struct MappingDesc {
  bool isConst = false;
  std::optional<float> constVal;
  unsigned tile;
  std::vector<size_t> slice;
};

// Map operands on tiles according to 'mapping[]'. First it is mapped linearly
// on all tiles (if required), then the specific 'mappings[]' are applied (in
// order). Note that each mapping can/will override the previous. This makes
// easier to obtain arbitrary mappings.
void mapTensor(Graph &graph, const Tensor &t, bool mapLinearly,
               const std::vector<MappingDesc> &mapping) {
  if (mapLinearly || (mapping.size() == 0)) {
    poputil::mapTensorLinearly(graph, t);
  }
  for (auto m : mapping) {
    if (m.slice.size() == 0) {
      graph.setTileMapping(t, m.tile);
    } else {
      std::vector<size_t> ends;
      for (auto i : m.slice) {
        ends.push_back(i + 1);
      }
      graph.setTileMapping(t.slice(m.slice, ends), m.tile);
    }
  }
}

// Utility function to read a MappingDesc from a stream
std::istream &operator>>(std::istream &in, MappingDesc &md) {
  char c = in.peek();
  if (c == 'c') {
    in >> c; // flush the peeked char
    md.isConst = true;

    in.seekg(0, in.end);
    int uiLength = in.tellg();
    if (uiLength > 1) {
      in.seekg(1, in.beg);
      char delim = 0;
      in >> delim;
      if (delim == ':') {
        float constVal = 0;
        in >> constVal;
        md.constVal = constVal;
      }
    }
  } else {
    in >> md.tile;
    in >> c;
    if (c != ':') {
      throw std::runtime_error("Invalid shape; expected ':'after tile number'");
    }
    ShapeOption<size_t> slice;
    in >> slice;
    md.slice = slice.val;
  }
  return in;
}

// Utility function to write a MappingDesc to a stream
std::ostream &operator<<(std::ostream &os, const MappingDesc &md) {
  if (md.isConst) {
    return os << "const";
  } else {
    os << md.tile << ":{";
    for (auto x : md.slice)
      os << x << ",";
    return os << "}";
  }
}

// This collects together information about one operand of a Unary/BinaryOp
struct OperandDescriptor {
  std::vector<size_t> shape;    // Shape, as defined on command line.
  std::vector<size_t> shapeExt; // Shape, rank-extended.
  std::vector<MappingDesc> map; // Indicates where to map this operand
};

// This extends to rank 'n' a given tensor shape.
// Returns a shape having rank 'n', obtained by prepending '1's at the left
// ('n' must be >= shape.size()).
// I.e. if shape is {6,1} and 'n' is 4, it returns {1,1,6,1}.
std::vector<size_t> extendShape(const std::vector<size_t> &shape, unsigned n) {
  unsigned m = shape.size();
  assert(n >= m);
  std::vector<size_t> shapeExt(n, 1);
  for (unsigned k = 0; k < m; k++) {
    shapeExt[n - m + k] = shape[k];
  }
  return shapeExt;
}

// Given a linear array 'data' (one of the host buffers) which represent a
// tensor with specified 'shape', get the element with indices specified by
// 'i[]', using broadcasting rules.
// Basically this returns:  data[ i[0], i[1], ... ].
template <typename T>
T get(const T data[], const std::vector<size_t> shape,
      const std::vector<unsigned> i) {
  unsigned offs = 0;
  for (unsigned k = 0; k < i.size(); k++) {
    // Need to keep into account broadcasting rules: if a certain
    // dimension is 1, then the corresponding index does not matter (i.e.
    // the effective index to use is 0)
    offs = offs * shape[k] + ((shape[k] == 1) ? 0 : i[k]);
  }
  return data[offs];
}

// Contains details of how one operand (or the result), possibly 2D, is stored
// inside a linear Tensor, and in a linear buffer in the host, including the
// position of the padding used for overprocessing/overwrite detection.
struct TestOperand {
  // FILL value is chosen so that we can detect overprocessing of 64 bit loads
  // with a stride of 6.
  // ALIGN is set to 8 to avoid rearrangements, as no vertex field requires more
  // than 8 bytes alignment.
  // Both must be a multiple of ANY possible type size (1,2,4), so
  // that the total padding always contains a whole number of elements.
  unsigned static constexpr FILL = 48; // In bytes
  unsigned static constexpr ALIGN = 8; // In bytes

  // Linear buffer where the in/out data is stored, in device format.
  // Each row (a single one for Supervisor vertices) has FILL bytes added at
  // the end. The end is then further padded, so that it aligns to ALIGN bytes.
  //
  // For instance, for a 1D test with a size of 3 half floats (if FILL was 4):
  //                             FILL=4 bytes       Pad to ALIGN
  //   <------ 6 bytes ------> <----- 4 -----> <--------------------->
  //   0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
  //  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
  //  | rowSizes[0] = 3 half  |###|###|###|###|###|###|###|###|###|###|
  //  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
  //
  //
  // All rows are stored contiguously. For instance, for a 2D test with
  // rows of length 3, 2, 1, 4 (half floats), still with FILL=4:
  //   0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
  //  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
  //  |rowSizes[0]=3 half     |###|###|###|###|###|###|###|###|###|###|
  //  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
  //  |rowSizes[1]=2  |###|###|###|###|1 half |###|###|###|###|###|###|
  //  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
  //  |rowSizes[3]=4 half             |###|###|###|###|###|###|###|###|
  //  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
  //
  // Additionally, we can also add some padding at the very start (startPad)
  // (not shown above).
  //
  // The content of 'rawBuf' is what is uploaded to the device as a single 1D
  // tensor, both for the input and output tensors. The operand, or the 2D
  // operand subvectors (rows) are connected to the appropriate slices of the 1D
  // uploaded tensor.
  //
  // All the padding (in the input and output buffer/tensor) is filled with a
  // (nonzero) byte pattern. that is checked after execution, for past-the-end
  // overwrites in the output. For float/half, the pattern (which contains NaNs)
  // is also written in the input tensors, to detect overprocessing
  std::unique_ptr<char[]> rawBuf;

  // Size of each row in elements, just the data, excluding the padding
  std::vector<unsigned> rowSizes;

  // Offsets in rawBuf/tensor where each rows begins (in elements, not bytes).
  // Will be populated with 'number of rows + 1' elements, the last one being
  // one past the end of the padding of the last row.
  std::vector<unsigned> offsets;

  Type type;

  // Total size (in elements, not bytes) of the rawBuf/tensor, after adding
  // alignment & padding
  unsigned totalElems = 0;

  // In elements. This can be added at the very start (before the first row) to
  // test vertices that handle different initial alignment. All rows start will
  // be offset by this.
  unsigned startPad = 0;

  /// Setup the offset and sizes relative to this operand
  ///
  /// \param[in] rowSz
  /// \param[in] startPadBytes Number of bytes to add at the start. Might be
  ///                          adjusted if is not a multiple of data size
  /// \param[in] warnOnAdjust  Print/do not print a message if startPad needs
  ///                          adjusting because of dataType size
  void setup(const Target &target, Type dataType,
             const std::vector<unsigned> &rowSz, unsigned &startPadBytes,
             bool warnOnAdjust = true) {
    type = dataType;
    const unsigned typeSize = target.getTypeSize(type);
    startPad = startPadBytes / typeSize;
    if ((startPadBytes % typeSize) != 0 && warnOnAdjust) {
      std::cout << "Warning: start alignment of " << startPadBytes << " is not "
                << "valid for type " << type.toString() << ". Alignment set to "
                << startPad * typeSize << "\n";
    }
    startPad = startPadBytes / typeSize;
    startPadBytes = startPad * typeSize;

    rowSizes = rowSz;
    unsigned numRows = rowSizes.size();
    offsets.resize(numRows + 1);
    totalElems = startPad;
    offsets[0] = startPad;
    for (unsigned i = 0; i < numRows; i++) {
      // Compute aligned/padded size of row (and offset of next row)
      unsigned nBytes = rowSizes[i] * typeSize;
      nBytes = gccs::alignNext(nBytes + FILL, ALIGN);
      unsigned sizePadded = nBytes / typeSize;
      totalElems += sizePadded;
      offsets[i + 1] = offsets[i] + sizePadded;
    }
  }

  enum OperandType { isScalar, is1D, is2D };

  // Calls graph.connect() appropriately for the correct slice(s) of the 1D
  // tensor that has been allocated with the padding.
  void connectOperand(Graph &graph, VertexRef &v, OperandType opType, Tensor &t,
                      const std::string &fieldName) {
    switch (opType) {
    case is2D: {
      unsigned numRows = rowSizes.size();
      graph.setFieldSize(v[fieldName], numRows);
      for (unsigned i = 0; i < numRows; i++) {
        graph.connect(v[fieldName][i],
                      t.slice(offsets[i], offsets[i] + rowSizes[i]));
      }
    } break;
    case isScalar:
      graph.connect(v[fieldName], t[offsets[0]]);
      break;
    default:
      graph.connect(v[fieldName],
                    t.slice(offsets[0], offsets[0] + rowSizes[0]));
    }
  };

  // Iterates over all padding bytes, for each of them calling:
  //   processByte(row, padOffs, i, padByteValue)
  // where:
  //   row    : the number of the row [-1..nRow). -1 means the initial padding.
  //   padOffs: offset in bytes inside rawBuf where the pad for 'row' starts
  //   i      : index for the pad byte for 'row' 0..nPad
  //   padByteValue: the value for the pad byte at: rawBuf[padOffs + i]
  //
  // This diagram represents how the buffer is structured (includes also an
  // initial padding). We need to scan all bytes marked '#'.
  //                   rowSizes[0]         rowSizes[1]        rowSizes[2]
  //                  <----------->        <---------->     <------------->
  //            +----+-------------+------+------------+---+---------------+--+
  //     rawBuf |####|             |######|            |###|               |##|
  //            +----+-------------+------+------------+---+---------------+--+
  // offsets[0] |--->|                    |                |                  |
  // offsets[1] |------------------------>|                |                  |
  // offsets[2] |----------------------------------------->|                  |
  // offsets[3] |------------------------------------------------------------>|
  //
  // padOffs=0  |----------------->|
  //            |                  |--> i
  //
  // The values for the pad bytes are chosen so that:
  //   1. Strings of adjacent bytes are all different, as much as possible.
  //   2. No pad byte is zero or one (which are valid boolean values).
  //   3. For half/float types, if 'useNan' is set, an aligned groups of 2/4
  //      bytes will represents a signalling NaN:
  //        Half NaN : aaaaaaaa s111110b                   (low to high addr)
  //        Float NaN: aaaaaaaa bbbbbbbb 10cccccc s1111111 (low to high addr)
  //      where at least one of the 'a', 'b', 'c' bits must be 1, and 's' can
  //      be 0 or 1
  //
  void processPadBytes(
      const Target &target,
      std::function<void(int, unsigned, unsigned, unsigned char)> processByte,
      bool isIpuModel, bool useNan) const {
    int numRows = rowSizes.size();
    unsigned typeSize = target.getTypeSize(type);
    unsigned char k = 2; // a running counter 2..123.
    unsigned prevOffs = 0;
    for (int row = 0; row < numRows + 1; row++) {
      unsigned n = offsets[row] - prevOffs;
      unsigned offsByte = prevOffs * typeSize;
      if (row < numRows)
        prevOffs += n + rowSizes[row];
      n *= typeSize;
      for (unsigned i = 0; i < n; i++) {
        // The padding bytes are the sequence 2, 3, 4, ... 123, 2, 3, ...
        // The value 123 is chosen to avoid having 124 (0x7c) which is the top
        // byte of a half NaN which is undesirable for the IpuModel.
        unsigned char padByteValue = k;
        k = (k < 122) ? k + 1 : 2;
        if (useNan) {
          // But if we want NaNs, we massage groups pf 2/4 bytes to be NaNs
          if (type == HALF) {
            if (isIpuModel) {
              // When running on the IpuModel, half signalling NaNs are
              // normalized to 0x7c01, so we can use that one value only
              padByteValue = (i & 1) ? 0x7c : 1;
            } else {
              if (i & 1) {
                // Alternate 0/1 in MS and LS bits so that the bytes at odd
                // offsets cycle between the values 7c, 7d, fc, fd
                unsigned j = i >> 1;
                padByteValue = 0x7c | (j & 1) | ((j & 2) << 6);
              }
            }
          } else {
            switch (i % 4) {
            case 2:
              padByteValue = (padByteValue & 0xc0) | 0x80;
              break;
            case 3:
              padByteValue =
                  0x7f | ((i & 0x04) << 5); // alternate 0/1 in sign bit
              break;
            default:;
            }
          }
        }
        processByte(row - 1, offsByte, i, padByteValue);
      }
    }
  }

  // Set the pad bytes in the raw buffer
  void setPadBytes(const Target &target, bool isIpuModel, bool useNan = false) {
    processPadBytes(
        target,
        [&](int row, unsigned rowEndOffs, unsigned i, unsigned char padByte) {
          rawBuf[rowEndOffs + i] = padByte;
        },
        isIpuModel, useNan);
  }

  // Verifies that the pad bytes in the raw buffer are the same we set.
  unsigned checkPadBytes(const Target &target, bool is2D, bool isIpuModel,
                         bool useNan = false) const {
    unsigned errCount = 0;
    processPadBytes(
        target,
        [&](int row, unsigned rowEndOffs, unsigned i, unsigned char expected) {
          unsigned char actual = (unsigned char)rawBuf[rowEndOffs + i];
          if (actual != expected) {
            std::string posStr =
                is2D ? to_string(row) + "," + to_string(i) : to_string(i);
            std::string rowStr = (row < 0)
                                     ? "in initial padding"
                                     : "past the end of row " + to_string(row);
            std::cerr << "ovewrite " << rowStr << ", at offset " << i
                      << ": expected:" << convertToString(expected)
                      << ";  actual:" << convertToString(actual) << "\n";
            errCount++;
          }
        },
        isIpuModel, useNan);
    if (errCount > 0) {
      std::cerr << "Failed: overwrite detected on " << errCount << " byte(s)\n";
    }
    return errCount;
  }
};

//*************************************************************************
// Converts a vector to a vector of optionals (of the same type)
template <typename T>
std::vector<std::optional<T>> convertToOptionalVector(const std::vector<T> &v) {
  std::vector<std::optional<T>> optVec;
  for (const T &x : v)
    optVec.push_back(std::optional<T>(x));
  return optVec;
}

// Create the auxiliary vertex to ensure alignment of the tensor 't'
void createAlignVertex(Graph &graph, ComputeSet &cs, const Tensor &t,
                       unsigned tile) {
  auto v = graph.addVertex(cs,
                           poputil::templateVertex("popops::NopAlignVertex",
                                                   TestOperand::ALIGN,
                                                   t.elementType()),
                           {{"t", t}});
  graph.setTileMapping(v, tile);
  graph.setPerfEstimate(v, 0);
}
#endif
