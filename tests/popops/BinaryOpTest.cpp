// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE BinaryOpTest
//
// Perform a binary operation between two tensors with any desired shape, each
// mapped in any desired way among tiles.

#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include "../lib/popops/ExprOpUtil.hpp"
#include "popops/ElementWise.hpp"
#include <poplibs_test/Util.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/token_functions.hpp>
#include <boost/tokenizer.hpp>

#include <exception>
#include <fstream>
#include <sstream>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popops;
using namespace poplibs_support;

const poplar::OptionFlags options{{"debug.instrumentCompute", "true"}};

// A descriptor to keep information about which tile to store a slice of
// a tensor on
struct MappingDesc {
  unsigned tile;
  std::vector<size_t> slice;
};

// This extends to rank 'n' a given tensor shape.
// Returns a shape having rank 'n', obtained by prepending '1's at the left
// ('n' must be >= shape.size()).
// I.e. if shape is {6,1} and 'n' is 4, it returns {1,1,6,1}.
static std::vector<size_t> extendShape(const std::vector<size_t> &shape,
                                       unsigned n) {
  unsigned m = shape.size();
  assert(n >= m);
  std::vector<size_t> shapeExt(n, 1);
  for (unsigned k = 0; k < m; k++) {
    shapeExt[n - m + k] = shape[k];
  }
  return shapeExt;
}

//*************************************************************************
/// Verifies on the host the result of the operation performed on the device.
///
/// \param deviceType          The device used.
/// \param dataType            The data type used (float, half).
/// \param in1Host             Data buffer for the first operand.
/// \param shape1Ext           Shape for first operand, rank-extended.
/// \param in2Host, shape1Ext  Data and shape for second operand.
/// \param outHost, outShape   Data (and shape) for result, obtained from
///                            device and converted to host float.
/// \param operation           Operation performed on device.
static bool verifyResult(const DeviceType &deviceType, const Type &dataType,
                         const std::vector<float> &in1Host,
                         const std::vector<size_t> &shape1Ext,
                         const std::vector<float> &in2Host,
                         const std::vector<size_t> &shape2Ext,
                         const std::vector<float> &outHost,
                         const std::vector<size_t> &shapeOut,
                         const expr::BinaryOpType operation) {
  unsigned errCount = 0; // how many mismatched elements we find
  double maxDelta = 0;

  // For float values, the results computed on the host will match exactly
  // the ones on the device, while for half we need to do some approximations.
  float clipTreshHalf = (isIpuModel(deviceType))
                            ? std::numeric_limits<float>::infinity()
                            : 65504.0f;
  float clipValueHalf = 65488.0f;

  auto equalValues = [&](float expected, float actual) {
    if (dataType == FLOAT) {
      return expected == actual;
    } else if (dataType == HALF) {
      // horrible contortions to verify result for halves. We should really
      // have a half bit-exact computation library for the host.
      if (actual >= clipTreshHalf) {
        return expected >= clipValueHalf;
      } else if (actual <= -clipTreshHalf) {
        return expected <= -clipValueHalf;
      } else {
        double delta = std::abs(expected - actual);
        bool isEqual = false;
        if (expected == 0) {
          isEqual = (expected == actual);
        } else {
          delta = delta / expected;
          isEqual = (delta < 0.002);
        }
        maxDelta = (delta > maxDelta) ? delta : maxDelta;
        return isEqual;
      }
    }
    return false;
  };

  unsigned n = shapeOut.size(); // How many dimensions we have

  // Given a linear array 'data' (one of the host buffers) which represent a
  // tensor with specified shape, get the element with indices specified by
  // 'i[]', using broadcasting rules.
  // Basically this returns:  data[ i[0], i[1], ... ]
  auto get = [&](const float data[], const std::vector<size_t> shape,
                 const std::vector<unsigned> i) {
    unsigned offs = 0;
    for (unsigned k = 0; k < n; k++) {
      // Need to keep into account broadcasting rules: if a certain
      // dimension is 1, then the corresponding index does not matter (i.e.
      // the effective index to use is 0)
      offs = offs * shape[k] + ((shape[k] == 1) ? 0 : i[k]);
    }
    return data[offs];
  };

  // Perform the specified 'operation' element-wise between in1 and in2 (on
  // the host) and compare with what is returned by the device.
  // We need to nest a variable number ('n') of loops to scan through all
  // 'n' dimensions. We use a recursive function the will recur once for each
  // nested loop ('n' times).
  // The vector 'i[]' contains the indices into 'in1', 'in2, 'out'
  std::vector<unsigned> i(n);
  // Cannot use 'auto' for the type, because it's a recursive function.
  std::function<void(unsigned)> loopOn = [&](unsigned k) {
    // Run the k-th nested loop
    for (i[k] = 0; i[k] < shapeOut[k]; i[k]++) {
      if (k == n - 1) {
        // This is the "innermost loop"; we need to compute:
        // expected[ i[0], i[1],... ] =
        //                in1[ i[0], i[1],... ] *OP*  in2[ i[0], i[1],... ]
        // and compare with the actual value from the device

        float actual = get(outHost.data(), shapeOut, i); // from device

        float val1 = get(in1Host.data(), shape1Ext, i);
        float val2 = get(in2Host.data(), shape2Ext, i);

        float expected;
        if (operation == expr::BinaryOpType::ADD) {
          expected = val1 + val2;
        } else if (operation == expr::BinaryOpType::MULTIPLY) {
          expected = val1 * val2;
        } else if (operation == expr::BinaryOpType::SUBTRACT) {
          expected = val1 - val2;
        } else {
          throw std::logic_error("Unrecognised operation type!");
        }
        if (!equalValues(expected, actual)) {
          std::cerr << "out[" << i[0];
          for (unsigned j = 1; j < n; j++)
            std::cerr << "," << i[j];
          std::cerr << "] : expected:" << expected << ";  actual:" << actual
                    << "\n";
          errCount++;
        }
      } else {
        loopOn(k + 1); // recur to go down to next nested loop
      }
    }
  };
  loopOn(0);

  // if (maxDelta>0) {
  //   std::cout << "max delta: " << maxDelta << "\n";
  // }
  return errCount == 0;
}

//*************************************************************************
/// Do a binary operation, where the first operand ('in1') is a tensor with
/// shape 'shape1', and the second operand ('in2') is a tensor with shape
/// 'shape2'. The shape that the output will have has already been computed
/// using broadcasting rules ('shapeOut').
///
/// \param deviceType            The device used.
/// \param dataType              The data type used (float, half).
/// \param shape1                Shape for first operand, as is.
/// \param shape1Ext             Shape for first operand, rank-extended.
/// \param map1                  Indicates where to map first operand.
/// \param shape2                Shape for second operand, as is.
/// \param shape2Ext             Shape for second operand, rank-extended.
/// \param map2                  Indicates where to map second operand.
/// \param outShape              Shape of result (computed from shape1, shape2).
/// \param tiles                 How many tiles to allocate.
/// \param mapLinearly           If yes, we map linearly the two operands on
///                              all tiles.
/// \param operation             Operation performed on device.
/// \param inPlace               Is the operation to be done in place?
/// \param doReport              Print poplar report.
/// \param doPrintTensors        Print the tensors (for verification).
/// \param ignoreData            Do not verify results.
/// \param enableOptimisations   Enable broadcasted vector op optimisations.
static bool doBinaryOpTest(
    const DeviceType &deviceType, const Type &dataType,
    const std::vector<size_t> &shape1, const std::vector<size_t> &shape1Ext,
    const std::vector<MappingDesc> &map1, const std::vector<size_t> &shape2,
    const std::vector<size_t> &shape2Ext, const std::vector<MappingDesc> &map2,
    const std::vector<size_t> &shapeOut, const unsigned tiles,
    const bool mapLinearly, const expr::BinaryOpType operation,
    const bool inPlace, const bool doReport, const bool doPrintTensors,
    const bool ignoreData, bool enableOptimisations) {

  auto nElems1 = std::accumulate(shape1.begin(), shape1.end(), std::size_t(1),
                                 std::multiplies<std::size_t>());

  auto nElems2 = std::accumulate(shape2.begin(), shape2.end(), std::size_t(1),
                                 std::multiplies<std::size_t>());

  auto nElemsOut =
      std::accumulate(shapeOut.begin(), shapeOut.end(), std::size_t(1),
                      std::multiplies<std::size_t>());

  // Allocate and initialise host buffers with some values. For compatibility
  // with the half case, we make sure we never set a value greater than the
  // greatest half value.
  std::vector<float> in1Host(nElems1);
  std::vector<float> in2Host(nElems2);
  for (unsigned i = 0; i < in1Host.size(); i++) {
    in1Host[i] = static_cast<float>((i + 1) % 65000);
  }
  for (unsigned i = 0; i < in2Host.size(); i++) {
    in2Host[i] = static_cast<float>((i + 32000) % 65000);
  }

  // Create Graph object, target and device
  auto device = createTestDevice(deviceType, 1, tiles);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  auto in1 = graph.addVariable(dataType, shape1, "in1");
  auto in2 = graph.addVariable(dataType, shape2, "in2");

  // Map operands on tiles. First it is mapped linearly on all tiles,
  // then the mappings specified by --mapX are applied.
  // Note that each mapping can/will override the previous. This makes easier
  // to obtain arbitrary mappings.
  auto mapTensor = [&](const Tensor &t,
                       const std::vector<MappingDesc> &mapping) {
    if (mapLinearly || (mapping.size() == 0)) {
      mapTensorLinearly(graph, t);
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
  };
  mapTensor(in1, map1);
  mapTensor(in2, map2);

  OptionFlags opOpts{{"enableVectorBroadcastOptimisations",
                      (enableOptimisations ? "true" : "false")}};

  // Make a program sequence to run the operation
  Sequence prog;
  Tensor out;
  if (inPlace) {
    mapInPlace(graph, operation, in1, in2, prog, "", opOpts);
    out = in1;
  } else {
    out = map(graph, operation, in1, in2, prog, "", opOpts);
  }

  // Create host 'transfer' buffers with the right size for the device type
  // (half,float)
  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  std::unique_ptr<char[]> in1HostRaw;
  std::unique_ptr<char[]> in2HostRaw;
  std::unique_ptr<char[]> outHostRaw;
  char *outHostRawPtr = nullptr;
  if (!ignoreData) {
    in1HostRaw = allocateHostMemoryForTensor(in1, "in1", graph, uploadProg,
                                             downloadProg, tmap);
    in2HostRaw = allocateHostMemoryForTensor(in2, "in2", graph, uploadProg,
                                             downloadProg, tmap);
    if (!inPlace) {
      outHostRaw = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                               downloadProg, tmap);
      outHostRawPtr = outHostRaw.get();
    } else {
      outHostRawPtr = in1HostRaw.get();
    }
  }

  if (doPrintTensors) {
    prog.add(PrintTensor("in1", in1));
    prog.add(PrintTensor("in2", in2));
    if (!inPlace)
      prog.add(PrintTensor("out", out));
  }

  // Run sequences
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), options);
  attachStreams(engine, tmap);

  // Copy and convert the data from the initialised buffers to the transfer
  // buffers (still on host)
  if (!ignoreData) {
    copy(target, in1Host.data(), in1Host.size(), dataType, in1HostRaw.get());
    copy(target, in2Host.data(), in2Host.size(), dataType, in2HostRaw.get());
  }

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run();

    if (doReport) {
      OptionFlags opt;
      opt.set("showExecutionSteps", "true");
      engine.printProfileSummary(std::cout, opt);
    }
  });

  // Check the result
  if (ignoreData) {
    std::cout << "Result not checked for correctness\n";
  } else {
    // Get the result out of the device
    std::vector<float> outHost(nElemsOut);
    copy(target, dataType, outHostRawPtr, outHost.data(), outHost.size());

    return verifyResult(deviceType, dataType, in1Host, shape1Ext, in2Host,
                        shape2Ext, outHost, shapeOut, operation);
  }
  return true;
}

//*************************************************************************
int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type dataType;

  std::string operation;
  bool doReport = false;
  bool doPrintTensors = false;
  bool inPlace = false;
  unsigned tiles = 0;
  bool ignoreData = false;
  bool enableOptimisations = true;
  ShapeOption<size_t> shape1;
  std::vector<MappingDesc> map1;
  ShapeOption<size_t> shape2;
  std::vector<MappingDesc> map2;

  po::options_description desc("Perform a binary operation between two tensors "
                               "having any specified shape, each mapped in any "
                               "desired way among tiles.\nOptions are:");

  // clang-format off
  desc.add_options()
    ("help", "Print help")
    ("report",
     po::value<bool>(&doReport)->default_value(doReport),
     "Provide a poplar report")
    ("options-file",
     po::value<std::string>(),
     "A file containing options, with the same syntax as the command line; "
     "can be also specified with '@options_file_name'")
    ("print",
     po::value<bool>(&doPrintTensors)->default_value(doPrintTensors),
     "Print the tensors")
    ("ignore-data",
     po::value<bool>(&ignoreData)->default_value(ignoreData),
     "Do not check correctness of result, useful for benchmarking without "
     "overhead of upload/download of tensors and slow host-side computation")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("data-type",
     po::value<Type>(&dataType)->required(),
     "Data Type (half, float)")
    ("in-place",
     po::value<bool>(&inPlace)->default_value(inPlace),
     "Do the specified operation in place")
    ("tiles",
     po::value<unsigned>(&tiles)->default_value(tiles),
     "Number of tiles to use for linearly mapping the operands. If "
     "unspecified, or 0, do not map lineraly the operands (use only the "
     "explicit mapping specified by --map1, --map2)")
    ("shape1",
     po::value<ShapeOption<size_t>>(&shape1)->multitoken(),
     "Shape for first operand, curly bracket delimited:  {d1,d2,...}.")
    ("map1",
     po::value<std::vector<MappingDesc>>(&map1)->multitoken(),
     "Tile mapping for first operand; a sequence of one or more: "
     "T:{d1,d2,...} ... , where T is the tile number and {d1,d2,...} is the "
     "slice mapped on T. If not specified, the operand is mapped linearly on "
     "the allocated tiles.")
    ("shape2",
     po::value<ShapeOption<size_t>>(&shape2)->multitoken(),
     "Shape for second operand; see --shape1")
    ("map2",
     po::value<std::vector<MappingDesc>>(&map2)->multitoken(),
     "Tile mapping for second operand; see --map1.")
    ("operation",
     po::value<std::string>(&operation)->required(),
     "Operation to perform: ADD,  MULTIPLY (MUL),  SUBTRACT (SUB)\n")
    ("enable-optimisations",
     po::value<bool>(&enableOptimisations)->default_value(enableOptimisations),
     "Enable broadcast operation optimisations")
    ;
  // clang-format on
  po::variables_map vm;
  try {
    // Additional command line parser to interpret an argument '@filename' as a
    // option "config-file" with the value "filename"
    auto at_option_parser = [](std::string const &s) {
      if ('@' == s[0])
        return std::make_pair(std::string("options-file"), s.substr(1));
      else
        return std::pair<std::string, std::string>();
    };

    po::store(po::command_line_parser(argc, argv)
                  .options(desc)
                  .extra_parser(at_option_parser)
                  .run(),
              vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
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
    return 1;
  }
  expr::BinaryOpType binOp;

  // Operations
  if (operation == "ADD") {
    binOp = expr::BinaryOpType::ADD;
  } else if ((operation == "MULTIPLY") || (operation == "MUL")) {
    binOp = expr::BinaryOpType::MULTIPLY;
  } else if ((operation == "SUBTRACT") || (operation == "SUB")) {
    binOp = expr::BinaryOpType::SUBTRACT;
  } else if (operation == "") {
    std::cerr << "Error: Operation not specified\n";
    return 1;
  } else {
    std::cerr << "Error: Operation <" << operation << "> not recognised\n";
    return 1;
  }

  // Find the shape of the output, applying the broadcasting rules
  // First, get the 'extended to the left' operand shapes; for instance, if
  // the two operands have shapes {9,8,7,6} and {7,1}, the second is 'extended'
  // to {1,1,7,1}
  unsigned n1 = shape1.val.size();
  unsigned n2 = shape2.val.size();
  unsigned n = std::max(n1, n2);
  const std::vector<size_t> &shape1Ext = extendShape(shape1.val, n);
  const std::vector<size_t> &shape2Ext = extendShape(shape2.val, n);

  std::vector<size_t> shapeOut(n);
  for (unsigned i = 0; i < n; i++) {
    size_t d1 = shape1Ext[i];
    size_t d2 = shape2Ext[i];

    // If the dimensions are different, one of them must be '1'
    if ((d1 != d2) && (d1 != 1) && (d2 != 1)) {
      std::cerr << "Error: shapes incompatible for broadcasting\n";
      return 1;
    }
    shapeOut[i] = std::max(d1, d2);
  }
  if (inPlace && (shapeOut != shape1.val)) {
    std::cerr << "Error: cannot specify '--in-place:true' if shape of output "
                 "is not the same as shape of first operand\n";
    return 1;
  }

  bool mapLinearly = tiles > 0;

  if (tiles == 0) {
    // Find the highest tile number in the tile mapping for the two operands
    for (auto m : map1) {
      tiles = std::max(tiles, m.tile);
    }
    for (auto m : map2) {
      tiles = std::max(tiles, m.tile);
    }
    tiles++;
  }

  return !doBinaryOpTest(deviceType, dataType, shape1.val, shape1Ext, map1,
                         shape2.val, shape2Ext, map2, shapeOut, tiles,
                         mapLinearly, binOp, inPlace, doReport, doPrintTensors,
                         ignoreData, enableOptimisations);
}

// Utility function to read a MappingDesc from a stream
std::istream &operator>>(std::istream &in, MappingDesc &md) {
  char c;
  in >> md.tile;
  in >> c;
  if (c != ':') {
    throw std::runtime_error("Invalid shape; expected ':'after tile number'");
  }
  ShapeOption<size_t> slice;
  in >> slice;
  md.slice = slice.val;
  return in;
}

// Utility function to write a MappingDesc to a stream
std::ostream &operator<<(std::ostream &os, const MappingDesc &md) {
  os << md.tile << ":{";
  for (auto x : md.slice)
    os << x << ",";
  return os << "}";
}
