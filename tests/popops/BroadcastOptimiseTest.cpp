// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE BroadcastOptimiseTest
// Test for the broadcastOp vertex operations.
// Used to verify aspects of implementation that
// aren't simply to correctness of arithmetic on a single item. Also
// for benchmarking.
// Eg - different length vectors for Supervisor vertices or other
// vectorised implementations, where data quantity is important.
//
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include "../lib/popops/ExprOpUtil.hpp"
#include "popops/ElementWise.hpp"
#include <poplibs_test/Util.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/program_options.hpp>

#include <exception>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popops;
using namespace poplibs_support;

const poplar::OptionFlags options{{"debug.instrumentCompute", "true"}};

//*************************************************
// Do a broadcast operation, where the first operand ('in') is a tensor with
// shape 'dims', and the second operand ('in2') is a tensor with shape
// {X,1,..,1} where 'X' is the 'dim'-th dimension of 'in'.
//
// For instance, if 'in' has shape {9,8,7,6}, 'dim' can be 0, 1, 2 or 3:
//
//    dim==0  -->  'in2' has shape {9,1,1,1}
//    dim==1  -->  'in2' has shape   {8,1,1}
//    dim==2  -->  'in2' has shape     {7,1}
//    dim==3  -->  'in2' has shape       {6}
//
// 'shuffleShape' indicates a shuffle of dimensions for 'in'. Note that 'in'
// is first created with the shuffled dimensions and then is back-shuffled to
// the original dimensions.
//
// If 'sliceMap' is empty, 'in' will be mapped linearly on the available tiles.
// If 'sliceMap' is NOT empty, 'in' will be mapped all on tile 0, except for
// the slice define by 'sliceMap' which will be mapped on tile 1.
bool doBroadcastVectorOptimiseTest(
    const DeviceType &deviceType, const unsigned tiles, const Type &dataType,
    const std::vector<unsigned> &dims,
    const std::vector<unsigned> &shuffleShape,
    const std::vector<unsigned> &sliceMap1,
    const std::vector<unsigned> &sliceMap2, const unsigned dim,
    expr::BinaryOpType operation, const bool inPlace,
    const std::function<double(double, double)> &hostFn, const bool doReport,
    const bool doPrintTensors, const bool ignoreData,
    bool enableOptimisations) {
  if (dim >= dims.size()) {
    std::cerr << "Dim out of range: " << dim << " >= " << dims.size() << "\n";
    return false;
  }

  auto nElems = std::accumulate(dims.begin(), dims.end(), std::size_t(1),
                                std::multiplies<std::size_t>());

  std::vector<double> inHost(nElems);
  std::vector<double> in2Host(dims[dim]);

  for (unsigned i = 0; i < nElems; i++) {
    inHost[i] = static_cast<double>(i) + 1;
  }

  for (unsigned i = 0; i < in2Host.size(); i++) {
    in2Host[i] = static_cast<double>(i) + 1;
  }

  // Create Graph object, target and device
  auto device = createTestDevice(deviceType, 1, tiles);
  Target target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  // Create and map the tensor to produce a layout based on the *shuffled*
  // dimensions (will be 'back-shuffled' later)
  std::vector<unsigned long> constructDims(dims.size());
  for (unsigned i = 0; i < dims.size(); i++)
    constructDims[i] = dims[shuffleShape[i]];
  Tensor in = graph.addVariable(dataType, constructDims, "Input Data");

  // If no 'sliceX' options specified, map linearly on all tiles.
  // Otherwise map everything on tile 0, except each 'sliceX' which will be
  // mapped on tile X.
  if ((sliceMap1.size() == 0) && (sliceMap2.size() == 0)) {
    mapTensorLinearly(graph, in);
  } else {
    graph.setTileMapping(in, 0);
    auto mapOneSlice = [&](const std::vector<unsigned> &slice, unsigned tile) {
      unsigned n = slice.size();
      if (n > 0) {
        std::vector<size_t> begins;
        std::vector<size_t> ends;
        for (auto i : slice) {
          begins.push_back(i);
          ends.push_back(i + 1);
        }
        graph.setTileMapping(in.slice(begins, ends), tile);
      }
    };
    mapOneSlice(sliceMap1, 1);
    mapOneSlice(sliceMap2, 2);
  }

  // Create operand 2 to element-wise ops such that we have a vector
  // broadcast.
  std::vector<std::size_t> in2Dims(dims.size() - dim, 1);
  in2Dims[0] = in2Host.size();
  auto in2 = graph.addVariable(dataType, in2Dims, "vector");
  mapTensorLinearly(graph, in2);

  // 'back-shuffle' to specified dimensions
  std::vector<unsigned> invShuffleShape(dims.size());
  for (unsigned i = 0; i < dims.size(); i++)
    invShuffleShape[shuffleShape[i]] = i;
  in = in.dimShuffle(invShuffleShape);

  // Find the size of the total data in the innermost dimension - those after
  // the specified dimension
  auto innerDimsSize =
      std::accumulate(&dims[dim + 1], &dims[dims.size()], std::size_t(1),
                      std::multiplies<std::size_t>());

  OptionFlags opOpts{{"enableVectorBroadcastOptimisations",
                      (enableOptimisations ? "true" : "false")}};

  // Make a program sequence to run the operation
  Sequence prog;
  Tensor out;
  if (inPlace) {
    switch (operation) {
    case expr::BinaryOpType::ADD:
      addInPlace(graph, in, in2, prog, "", opOpts);
      break;
    case expr::BinaryOpType::MULTIPLY:
      mulInPlace(graph, in, in2, prog, "", opOpts);
      break;
    case expr::BinaryOpType::SUBTRACT:
      subInPlace(graph, in, in2, prog, "", opOpts);
      break;
    default:
      throw std::logic_error("Unrecognised operation type!");
    }
    out = in;
  } else {
    switch (operation) {
    case expr::BinaryOpType::ADD:
      out = add(graph, in, in2, prog, "", opOpts);
      break;
    case expr::BinaryOpType::MULTIPLY:
      out = mul(graph, in, in2, prog, "", opOpts);
      break;
    case expr::BinaryOpType::SUBTRACT:
      out = sub(graph, in, in2, prog, "", opOpts);
      break;
    default:
      throw std::logic_error("Unrecognised operation type!");
    }
  }

  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  std::unique_ptr<char[]> inHostRaw;
  std::unique_ptr<char[]> in2HostRaw;
  std::unique_ptr<char[]> outHostRaw;
  char *outHostRawPtr = nullptr;
  if (!ignoreData) {
    inHostRaw = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                            downloadProg, tmap);
    in2HostRaw = allocateHostMemoryForTensor(in2, "in2", graph, uploadProg,
                                             downloadProg, tmap);
    if (!inPlace) {
      outHostRaw = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                               downloadProg, tmap);
      outHostRawPtr = outHostRaw.get();
    } else {
      outHostRawPtr = inHostRaw.get();
    }
  }

  if (doPrintTensors) {
    prog.add(PrintTensor("in", in));
    prog.add(PrintTensor("in2", in2));
    if (!inPlace)
      prog.add(PrintTensor("out", out));
  }

  // Run sequences and compare host and IPU result
  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, options);
  attachStreams(engine, tmap);

  if (!ignoreData) {
    copy(target, inHost.data(), inHost.size(), dataType, inHostRaw.get());
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
  if (!ignoreData) {
    std::vector<double> outHost(nElems);
    copy(target, dataType, outHostRawPtr, outHost.data(), outHost.size());

    std::vector<double> outModel(nElems);
    for (unsigned i = 0; i < nElems; i++)
      outModel[i] = 0;

    for (std::size_t i = 0; i < nElems; ++i) {
      auto in2Index = (i / innerDimsSize) % in2Host.size();
      outModel[i] = hostFn(inHost[i], in2Host[in2Index]);
    }

    return checkIsClose("StdTest", outHost.data(), {outHost.size()},
                        outModel.data(), outModel.size(), 0.05, 0.05);
  }

  return true;
}

//******************************************************************************
int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  Type dataType;

  std::string operation;
  bool doReport = false;
  bool doPrintTensors = false;
  bool inPlace = true;
  bool ignoreData = false;
  bool enableOptimisations = true;
  unsigned dim = 1;
  unsigned tiles = 0;
  ShapeOption<unsigned> dims;
  ShapeOption<unsigned> sliceMap1;
  ShapeOption<unsigned> sliceMap2;
  ShapeOption<unsigned> shuffleShape;
  po::options_description desc("Options");

  // clang-format off
  desc.add_options()
    ("help", "Print help")
    ("report",
     po::value<bool>(&doReport)->default_value(doReport),
     "Provide a poplar report")
    ("print",
     po::value<bool>(&doPrintTensors)->default_value(doPrintTensors),
     "Print the tensors")
    ("ignoreData",
     po::value<bool>(&ignoreData)->default_value(ignoreData),
     "Ignore values of data, useful for benchmarking without "
     "overhead of upload/download of tensors")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("data-type",
     po::value<Type>(&dataType)->required(),
     "Data Type")
     ("dim",
     po::value<unsigned>(&dim)->default_value(dim),
     "Index (into 'dims') of first dimension of second operand")
    ("in-place",
     po::value<bool>(&inPlace)->default_value(inPlace),
     "Do the specified operation in place")
    ("tiles",
     po::value<unsigned>(&tiles)->default_value(tiles),
     "Number of tiles to use")
    ("dims",
     po::value<ShapeOption<unsigned>>(&dims)->multitoken(),
     "Dimensions for first operand")
    ("slice1",
     po::value<ShapeOption<unsigned>>(&sliceMap1)->multitoken(),
     "Slice of first operand that will be allocated to tile 1")
    ("slice2",
     po::value<ShapeOption<unsigned>>(&sliceMap2)->multitoken(),
     "Slice of first operand that will be allocated to tile 2")
    ("dim-shuffle",
      po::value<ShapeOption<unsigned>>(&shuffleShape)->multitoken(),
      "Shape to shuffle the tensor to before mapping")
    ("operation",
     po::value<std::string>(&operation)->required(),
     "Allowed operations: ADD MULTIPLY SUBTRACT\n")
    ("enable-optimisations",
     po::value<bool>(&enableOptimisations)->default_value(enableOptimisations),
     "Enable broadcasted vector op optimisations")
    ;
  // clang-format on
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  expr::BinaryOpType broadcastOperation;
  std::function<double(double, double)> broadcastHostFn;

  // Operations
  if (operation == "ADD") {
    broadcastOperation = expr::BinaryOpType::ADD;
    broadcastHostFn = [](double x, double y) -> double { return x + y; };
  } else if ((operation == "MULTIPLY") || (operation == "MUL")) {
    broadcastOperation = expr::BinaryOpType::MULTIPLY;
    broadcastHostFn = [](double x, double y) -> double { return x * y; };
  } else if ((operation == "SUBTRACT") || (operation == "SUB")) {
    broadcastOperation = expr::BinaryOpType::SUBTRACT;
    broadcastHostFn = [](double x, double y) -> double { return x - y; };
  } else {
    std::cerr << " Error: Operation " << operation << " not recognised\n";
    return 1;
  }

  // If 'shuffleShape' was not specified, just specify a 'null' shuffle.
  std::vector<unsigned> shuffle = shuffleShape.val;
  if (shuffle.size() == 0) {
    for (unsigned i = 0; i < dims.val.size(); i++)
      shuffle.push_back(i);
  }

  // If the 'tiles' option was not specified, we set 1, 2 or 3 tiles
  // depending if the 'slice1' and 'slice2' options where specified or not.
  if (tiles == 0) {
    if (sliceMap2.val.size() != 0)
      tiles = 3;
    else if (sliceMap1.val.size() != 0)
      tiles = 2;
    else
      tiles = 1;
  }

  return !doBroadcastVectorOptimiseTest(
      deviceType, tiles, dataType, dims.val, shuffle, sliceMap1.val,
      sliceMap2.val, dim, broadcastOperation, inPlace, broadcastHostFn,
      doReport, doPrintTensors, ignoreData, enableOptimisations);
}
