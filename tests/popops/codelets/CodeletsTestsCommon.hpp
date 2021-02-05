// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popops_codelets_CodeletsTestCommon_hpp
#define popops_codelets_CodeletsTestCommon_hpp

// Definitions/declarations used in elementwise unarey/binary operation test
// code.

#include <poplar/Type.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/ElementWise.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/token_functions.hpp>
#include <boost/tokenizer.hpp>

#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>

using namespace poplar;
using namespace popops;
using namespace poplibs_support;
using namespace poplibs_test::util;

namespace po = boost::program_options;

// Overloaded & templated convertToString functions to print correctly
// from inside the templated 'verifyResult()'  function

template <typename INT_T> std::string convertIntToHexStr(INT_T val) {
  std::stringstream ss;
  ss << val << " (0x" << std::hex << val << ")";
  return ss.str();
}

std::string convertToString(char val) { return convertIntToHexStr(int(val)); }

std::string convertToString(signed char val) {
  return convertIntToHexStr(int(val));
}

std::string convertToString(unsigned char val) {
  return convertIntToHexStr(int(val));
}

std::string convertToString(int val) { return convertIntToHexStr(val); }

std::string convertToString(unsigned val) { return convertIntToHexStr(val); }

std::string convertToString(short val) { return convertIntToHexStr(val); }

std::string convertToString(unsigned short val) {
  return convertIntToHexStr(val);
}

std::string convertToString(bool val) { return convertToString(unsigned(val)); }

template <typename T> std::string convertToString(T val) {
  std::stringstream ss;
  ss << val;
  return ss.str();
}

// A 'random generator' engine which might not be there (in which case it means
// that we are not using random values to fill the data buffers).
using RandomEngine = std::optional<std::minstd_rand>;

// Filling one of the operand buffers for int data.
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

// Filling one of the operand buffers for short/unsigned short data.
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

// Filling one of the operand buffers for short/unsigned short data.
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

// Parses the command line options.
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

// This contains the size of an operand for a UnaryOp, Cast or BinaryOp test.
// It can contain a single value (for a 1D vertex), multiple values, for a 2D
// vertex, or a 'ROW x COL' value (for a VectorOuter vertex or a 2D vertex)
struct SizeDesc {
  bool isRowsByCols = false;
  std::vector<unsigned> val;
};

// Utility function to read a SizeDesc from a stream. Three formats are valid:
//
//   1234          - A single positive integer value
//
//   [11,22,33,44] - A list of integers
//
//   666x333       - A 'ROW x COL' value
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
  unsigned n = sd.val.size();
  if (sd.isRowsByCols) {
    if (n != 2)
      throw std::runtime_error("Invalid SizeDesc: 'isRowsByCols' is set, but "
                               "'val' has " +
                               std::to_string(n) + "element (instead of 2)");
    os << sd.val.at(0) << "x" << sd.val.at(1);
  } else {
    os << "[";
    for (unsigned i = 0; i < n - 1; i++)
      os << sd.val[i] << ",";
    if (n > 0)
      os << sd.val[0];
    os << "]";
  }
  return os;
}

// If we are comparing cycles, return the second device. If it was unspecified
// ("default") we try to match Sim1 with IpuModel1 or Sim2 with IpuModel2,
// otherwise we just use the specified device.
// \param mainDevice        The main device to run on, from cmd line
// \param compareDeviceStr  The 'cycle-compare' device string, from cmd line
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

#endif
