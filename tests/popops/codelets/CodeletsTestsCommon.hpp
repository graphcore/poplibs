// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popops_codelets_CodeletsTestCommon_hpp
#define popops_codelets_CodeletsTestCommon_hpp

// Definitions/declarations used in elementwise unarey/binary operation test
// code.

#include <poplar/Type.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

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

using boost::format;
using std::to_string;

namespace po = boost::program_options;

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
                 const Type IPUType, const std::vector<unsigned> &sizes) {
  unsigned nRows = sizes.size();
  unsigned j = 0;
  std::cout << name << ": {" << ((nRows > 1) ? "\n" : "");
  for (unsigned k = 0; k < nRows; k++) {
    std::cout << ((nRows > 1) ? "{" : "");
    unsigned n = sizes[k];
    for (unsigned i = 0; i < n; i++) {
      if (IPUType == BOOL) {
        std::cout << int(buf.at(j++));
      } else {
        std::cout << convertToString(buf.at(j++));
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

// Filling one of the operand buffers for FLOAT and HALF data (both use 'float'
// buffers on the host)
void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<float> &buf, float i, float min, float max,
                bool nonZero) {
  std::uniform_real_distribution<float> d(min, max);
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
  std::string toString() const {
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
        sd.val.push_back(detail::readValue<unsigned>(in));
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
    sd.val.push_back(detail::readValue<unsigned>(in));
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
      sd.val.push_back(detail::readValue<unsigned>(in));
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

  float cycleCompareThreshold = 10; // percent
};

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
/// \param[inout] cycles     If not null, we want to get back the cycles used
///                          when running the set of tests.
/// \return   number of FAILED tests.
template <typename TestRecord>
unsigned runTests(std::vector<std::shared_ptr<TestRecord>> &tests,
                  unsigned offs, unsigned numTests, const DeviceType deviceType,
                  const MiscOptions &options, bool verbose,
                  uint64_t *cycles = nullptr) {
  // The name of the compute set where we run the vertex under test.
  const static std::string computeSetName = "vertexComputeSet";

  TestDevice device = createTestDevice(deviceType, 1, numTests + 1);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);
  ComputeSet cs = graph.addComputeSet(computeSetName);
  Sequence program, upload, download;

  StreamMap streamMap;
  for (unsigned i = 0; i < numTests; i++) {
    // If running only one test, we print the info about that test before
    // doing the setup. This will help debug any problem arising during the
    // setup itself. We also do the same in the case we are not running the
    // data verification.
    if (verbose && (numTests == 1 || options.ignoreData)) {
      std::cout << tests[offs + i]->toString() << std::endl;
    }
    doSetupTest(target, graph, upload, cs, download, streamMap,
                *tests[offs + i], i, options);
  }
  program.add(Execute(cs));

  // === Run the program
  OptionFlags engOpts;
  if (options.report || cycles) {
    engOpts.set("debug.instrumentCompute", "true");
  }
  if (options.disableFpExceptions) {
    engOpts.set("debug.floatPointOpException", "false");
  }
  Engine engine(graph, Sequence(upload, program, download), engOpts);
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

      if (!doVerifyTest(target, isIpuModel(deviceType), *tests[offs + i],
                        options))
        errCount++;
    }
  }
  if (cycles) {
    // Get the cycles by searching the "simulation"/"steps" vector for an
    // "OnTileExecute" element having the compute set name we used.
    poplar::ProfileValue execProfile = engine.getExecutionProfile();
    for (auto s : execProfile["simulation"]["steps"].asVector()) {
      if (s["type"] == "OnTileExecute" && s["name"] == computeSetName) {
        *cycles = s["cycles"].asUint();
      }
    }
  }
  return errCount;
}

//*************************************************************************
/// If we are going to compare cycles between two devices, return the second
/// device to use, using the string specified on the command line. If that
/// string was "default", then we try to match 'Sim1' with 'IpuModel1' or
/// 'Sim2' with 'IpuModel2', otherwise we just use the specified device.
///
/// \param mainDevice        The main (first) device to run on, from cmd line
/// \param compareDeviceStr  The 'cycle-compare' device string, from cmd line
///
/// \return The device to use as second device in the comparison
DeviceType getCycleCompareDevice(const DeviceType &mainDevice,
                                 const std::string &compareDeviceStr) {
  std::optional<DeviceType> compDev;
  if (compareDeviceStr == "default") {
    if (mainDevice == DeviceType::Sim1) {
      compDev = DeviceType::IpuModel1;
    } else if (mainDevice == DeviceType::Sim2) {
      compDev = DeviceType::IpuModel2;
    }
  }
  if (!compDev) {
    std::istringstream is(compareDeviceStr);
    is >> *compDev;
  }
  return *compDev;
}

//*************************************************************************
/// Compare the cycles used when running the vertex with the two specified
/// devices. Prints result on standard output.
///
/// \param[in]    devices  Two devices for which we will run the test
/// \param[in]    test     Test record defining the specific vertex and
///                        operand(s) size.
/// \param[in]    options  Global options.
///
/// \return true   if both run returned successfully and the difference is less
///                than 'compareThreshold' % of the run with the first device.
template <typename TestRecord, typename VertexDesc>
bool compareCycles(const std::vector<DeviceType> devices,
                   std::shared_ptr<TestRecord> test,
                   const MiscOptions &options) {
  assert(devices.size() == 2);

  VertexDesc &vertex = *test->vertex;

  std::stringstream devName[2]; // Strings with the name of selected devices
  std::cout << vertex.vClassFmt << std::flush;

  bool ok[2];
  uint64_t cycles[2];
  // Run with the two devices and get the cycles
  for (unsigned i = 0; i < 2; i++) {
    devName[i] << devices[i];
    uint64_t cyc = 0;
    std::vector<std::shared_ptr<TestRecord>> testVect = {test};
    ok[i] = runTests<TestRecord>(testVect, 0, 1, devices[i], options, false,
                                 &cyc) == 0;
    if (!ok[i]) {
      std::cout << "Failed on device " << devName[i].str() << " (see stderr)\n";
      return false;
    }
    cycles[i] = cyc;
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
/// \param[in]    devices    Normally contains a single device, or two if we are
///                          running a cycle comparison.
/// \param[inout] errCount   If the test is run immediatley, will be increased
///                          by one if the test fails
/// \param[in]    options    Global options.
template <typename TestRecord, typename VertexDesc>
void addOneTest(std::optional<std::vector<std::shared_ptr<TestRecord>>> &tests,
                std::shared_ptr<TestRecord> newTest,
                std::vector<DeviceType> devices, unsigned &errCount,
                const MiscOptions &options) {
  // If we run a cycle comparison (devices has 2 elements), or if have specfied
  // to run test individually ('tests[]' is empty), we run the one test
  // straight away here, otherwise we add this test records in 'tests[]' to be
  // run afterwards.
  if (devices.size() == 2) {
    errCount += compareCycles<TestRecord, VertexDesc>(devices, newTest, options)
                    ? 0
                    : 1;
  } else if (tests == std::nullopt) {
    assert(devices.size() == 1);
    std::vector<std::shared_ptr<TestRecord>> testVect = {newTest};
    errCount += runTests(testVect, 0, 1, devices[0], options, true);
  } else {
    tests->emplace_back(std::move(newTest));
  }
}

//*************************************************************************
/// Run all tests that have been accumulated in 'tests[]' (if any), or do
/// nothing if the test have already been run individually
/// \param[inout] tests      A vector with the tests to run. Could be 'nullopt'
///                          if we have already run the tests
/// \param[in]    numTests   How many tests are in 'tests[]'. If tests was
///                          empty, the total tests run already.
/// \param[in]    groupTests how many tests to group together in a single
///                          graph and compute set
/// \param[in]    deviceType What type of device to run the tests on
/// \param[inout] errCount   Will be updated with the number of failed tests.
/// \param[in]    options    Global options.
///
template <typename TestRecord>
void runAllTests(std::optional<std::vector<std::shared_ptr<TestRecord>>> &tests,
                 unsigned numTests, unsigned groupTests, DeviceType deviceType,
                 unsigned &errCount, const MiscOptions &options) {
  if (numTests == 0) {
    throw std::runtime_error("The specified vertex, operand(s) and data "
                             "type(s) do not match any valid combination");
  }
  if (tests) {
    assert(numTests == tests->size());
    // Run the tests in batches of up to 'groupTests' together on a single
    // graph/ single CS, each test on a different tile. We limit grouping tests
    // on multiple tile to 'groupTests' because when running on the simulators,
    // if too many tiles are used, execution is slower.
    unsigned offs = 0, n = numTests;
    while (n) {
      unsigned l = n > groupTests ? groupTests : n;
      errCount += runTests(*tests, offs, l, deviceType, options, true);
      offs += l;
      n -= l;
    }
  }
  if (numTests > 1) {
    std::cout << "BinaryCodeletsTest: " << numTests << " tests run in total; "
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
                      boost::optional<std::string> &cycleCompareDevice,
                      std::vector<Type> &dataTypes, unsigned &groupTests,
                      MiscOptions &options) {
  // clang-format off
  poDesc.add_options()
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(DeviceType::Sim2),
     "Device type")
    ("data-type",
     po::value<std::vector<Type>>(&dataTypes)->multitoken(),
     "Data type: one or more of half, float, int, uint, short, ushort, bool, "
     "char, schar, uchar")
    ("compare-cycles",
     po::value<boost::optional<std::string>>(&cycleCompareDevice)->
                                         implicit_value(std::string("default")),
     "For each specified vertex, compare the cycles reported by the device ("
     "--device-type option) and another device specified by this option")
    ("cycle-threshold",
     po::value<float>(&options.cycleCompareThreshold)->
                                  default_value(options.cycleCompareThreshold),
     "Percent threshold when running the --compare-cycle option. An (absolute) "
     "cycle difference greater than this threshold will make the test fail.")
    ("report",
     po::value<bool>(&options.report)->implicit_value(true),
     "Provide a poplar report")
    ("disable-fp-exceptions",
     po::value<bool>(&options.disableFpExceptions)->
                                    default_value(options.disableFpExceptions),
     "Disable floating point exceptions when running on device.")
    ("options-file",
     po::value<std::string>(),
     "A file containing options, with the same syntax as the command line; "
     "can be also specified with '@options_file_name'")
    ("random-seed",
     po::value<unsigned>(&options.randomSeed)->
                                            implicit_value(options.randomSeed),
     "Seed for random data. Value of 0 means 'no random data'")
    ("ignore-data",
     po::value<bool>(&options.ignoreData)->implicit_value(true),
     "Do not check correctness of result, useful for benchmarking without "
     "overhead of host-side computation")
    ("group-tests",
     po::value<unsigned>(&groupTests)->implicit_value(100),
     "Run multiple tests together in a single graph and single compute set, "
     "each test on a separate tile, to increase execution speed")
    ("print-buffers",
     po::value<bool>(&options.printBuffers)->implicit_value(true),
     "Print the input and output buffers")
    ("help", "Print help")
    ;
  // clang-format on
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

#endif
