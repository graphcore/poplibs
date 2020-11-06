// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popops_codelets_CodeletsTestCommon_hpp
#define popops_codelets_CodeletsTestCommon_hpp

// Definitions/declarations used in elementwise unarey/binary operation test
// code.

#include <poplar/Type.hpp>
#include <poplibs_support/TestDevice.hpp>
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
namespace po = boost::program_options;

// We use 'unsigned char' on the host instead of 'bool', because
// std::vector<bool> gets specialised with 1 bit per element instead of 1 byte
typedef unsigned char HostBool;

// Overloaded & templated convertToString functions to print correctly
// from inside the templated 'verifyResult()'  function

template <typename INT_T> std::string convertIntToHexStr(INT_T val) {
  std::stringstream ss;
  ss << val << " (0x" << std::hex << val << ")";
  return ss.str();
}

std::string convertToString(int val) { return convertIntToHexStr(val); }

std::string convertToString(unsigned val) { return convertIntToHexStr(val); }

std::string convertToString(short val) { return convertIntToHexStr(val); }

std::string convertToString(unsigned short val) {
  return convertIntToHexStr(val);
}

std::string convertToString(HostBool val) {
  return convertToString(unsigned(val));
}

template <typename T> std::string convertToString(T val) {
  std::stringstream ss;
  ss << val;
  return ss.str();
}

// A 'random generator' engine which might not be there (in which case it means
// that we are not using random values to fill the data buffers).
using RandomEngine = std::optional<std::minstd_rand>;

// Filling one of the operand buffers, for boolean data. We always fill it with
// random booleans, (ignoring 'i', 'min', 'max' and 'nonZero')
void fillBuffer(const Type &dataType, RandomEngine &rndEng,
                std::vector<HostBool> &buf, int i, HostBool min, HostBool max,
                bool nonZero) {
  std::bernoulli_distribution d(0.5);

  for (auto &x : buf)
    x = d(*rndEng);
}

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
#endif
