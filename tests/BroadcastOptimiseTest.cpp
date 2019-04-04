// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
// Test for the broadcastOp vertex operations.
// Used to verify aspects of implementation that
// aren't simply to correctness of arithmetic on a single item. Also
// for benchmarking.
// Eg - different length vectors for Supervisor vertices or other
// vectorised implementations, where data quantity is important.
//
#include <TestDevice.hpp>
#include <poplar/Engine.hpp>
#include <popops/Zero.hpp>

#include "poputil/VertexTemplates.hpp"

#include <poputil/TileMapping.hpp>
#include <popops/codelets.hpp>
#include "popops/ElementWise.hpp"
#include "../lib/popops/ExprOpUtil.hpp"
#include <poplibs_test/Util.hpp>

#include <boost/program_options.hpp>

#include <exception>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_test::util;
using namespace popops;

const poplar::OptionFlags options {
  {"debug.executionProfile", "compute_sets"}
};

//*************************************************
bool doBroadcastVectorOptimiseTest(const DeviceType &deviceType,
              const unsigned tiles,
              const Type &dataType,
              const std::vector<unsigned> &dims,
              const std::vector<unsigned> &shuffleShape,
              const unsigned dim,
              expr::BroadcastOpType operation, const bool inPlace,
              const std::function<double(double, double)> &hostFn,
              const bool doReport,
              const bool ignoreData,
              bool enableOptimisations) {
  if(dim >= dims.size()) {
    std::cerr<<"Dim out of range: " << dim << " >= "<< dims.size() <<"\n";
    return false;
  }

  auto nElems = std::accumulate(dims.begin(), dims.end(),
                                std::size_t(1),
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

  // Create and map the tensor to produce a layout based on the shuffled
  // dimensions.  Then shuffle it to match the specified tensor shape
  std::vector<unsigned long> constructDims(dims.size());
  for(unsigned i = 0; i < dims.size(); i++)
    constructDims[i] = dims[shuffleShape[i]];
  Tensor in = graph.addVariable(dataType, constructDims, "Input Data");
  mapTensorLinearly(graph, in);

  // Create operand 2 to element-wise ops such that we have a vector
  // broadcast.
  std::vector<std::size_t> in2Dims(dims.size() - dim, 1);
  in2Dims[0] = in2Host.size();
  auto in2 = graph.addVariable(dataType, in2Dims, "vector");
  mapTensorLinearly(graph, in2);

  std::vector<unsigned> invShuffleShape(dims.size());
  for(unsigned i = 0; i < dims.size(); i++)
    invShuffleShape[shuffleShape[i]] = i;
  in = in.dimShuffle(invShuffleShape);


  // Find the size of the total data in the innermost dimension - those after
  // the specified dimension
  auto innerDimsSize = std::accumulate(&dims[dim + 1],
                                       &dims[dims.size()],
                                       std::size_t(1),
                                       std::multiplies<std::size_t>());

  OptionFlags opOpts{
    {"enableVectorBroadcastOptimisations",
      (enableOptimisations ? "true" : "false")}
  };

  // Make a program sequence to run the operation
  Sequence prog;
  Tensor out;
  if(inPlace) {
    switch (operation) {
      case expr::BroadcastOpType::ADD:
        addInPlace(graph, in, in2, prog, "", opOpts);
        break;
      case expr::BroadcastOpType::MULTIPLY:
        mulInPlace(graph, in, in2, prog, "", opOpts);
        break;
      case expr::BroadcastOpType::SUBTRACT:
        subInPlace(graph, in, in2, prog, "", opOpts);
        break;
      default:
        throw std::logic_error("Unrecognised operation type!");
    }
    out = in;
  }
  else {
    switch (operation) {
      case expr::BroadcastOpType::ADD:
        out = add(graph, in, in2, prog, "", opOpts);
        break;
      case expr::BroadcastOpType::MULTIPLY:
        out = mul(graph, in, in2, prog, "", opOpts);
        break;
      case expr::BroadcastOpType::SUBTRACT:
        out = sub(graph, in, in2, prog, "", opOpts);
        break;
      default:
        throw std::logic_error("Unrecognised operation type!");
    }
  }

  std::vector<std::pair<std::string, char *>> tmap;
  Sequence uploadProg, downloadProg;
  std::unique_ptr<char []> inHostRaw;
  std::unique_ptr<char []> in2HostRaw;
  std::unique_ptr<char []> outHostRaw;
  char *outHostRawPtr = nullptr;
  if (!ignoreData) {
    inHostRaw = allocateHostMemoryForTensor(in, "in", graph,
                                            uploadProg, downloadProg, tmap);
    in2HostRaw = allocateHostMemoryForTensor(in2, "in2", graph,
                                             uploadProg, downloadProg, tmap);
    if (!inPlace) {
      outHostRaw = allocateHostMemoryForTensor(out, "out", graph,
                                               uploadProg, downloadProg, tmap);
      outHostRawPtr = outHostRaw.get();
    } else {
      outHostRawPtr = inHostRaw.get();
    }
  }

  // Run each sequence and compare host and IPU result
  Engine engine(graph, Sequence(uploadProg, prog, downloadProg), options);
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
    for(unsigned i = 0;i < nElems ;i++)
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
  bool inPlace = true;
  bool ignoreData = false;
  bool enableOptimisations = true;
  unsigned dim = 1;
  unsigned tiles = 1;
  ShapeOption<unsigned> dims;
  ShapeOption<unsigned> shuffleShape ;
  po::options_description desc("Options");

  desc.add_options()
    ("help", "Print help")
     ("report",
     po::value<bool>(&doReport)->default_value(doReport),
     "Provide a poplar report")
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
     "Dimension")
    ("in-place",
     po::value<bool>(&inPlace)->default_value(inPlace),
     "in place operation")
    ("tiles",
     po::value<unsigned>(&tiles)->default_value(tiles),
     "Number of tiles to use")
    ("dims",
     po::value<ShapeOption<unsigned>>(&dims)->multitoken(),
     "Dimensions for scaledAdd")
     ("dim-shuffle",
      po::value<ShapeOption<unsigned>>(&shuffleShape)->multitoken(),
      "Shape to shuffle the tensor to before mapping")
    ("operation",
     po::value<std::string>(&operation)->required(),
     "Allowed operations: ADD MULTIPLY SUBTRACT VARIANCE_TO_INV_STD_DEV"
     " INV_STD_DEV_TO_VARIANCE\n")
    ("enable-optimisations",
     po::value<bool>(&enableOptimisations)->default_value(enableOptimisations),
     "Enable broadcasted vector op optimisations")
    ;
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
  expr::BroadcastOpType broadcastOperation;
  std::function<double(double, double)> broadcastHostFn;

  // Operations
  if(operation == "ADD") {
    broadcastOperation = expr::BroadcastOpType::ADD;
    broadcastHostFn = [](double x, double y) -> double {
          return x + y;};
  }
  else if(operation == "MULTIPLY") {
    broadcastOperation = expr::BroadcastOpType::MULTIPLY;
    broadcastHostFn = [](double x, double y) -> double {
          return x * y;};
  }
  else if(operation == "SUBTRACT") {
    broadcastOperation = expr::BroadcastOpType::SUBTRACT;
    broadcastHostFn = [](double x, double y) -> double {
          return x - y;};
  }
  else {
    std::cerr<< " Error: Operation " << operation << " not recognised\n";
    return 1;
  }

  return !doBroadcastVectorOptimiseTest(deviceType, tiles, dataType, dims.val,
                  shuffleShape.val, dim,
                  broadcastOperation, inPlace,
                  broadcastHostFn, doReport, ignoreData,
                  enableOptimisations);
}
