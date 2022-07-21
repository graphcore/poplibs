// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE RedcutionTests
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <functional>
#include <iostream>
#include <limits>
#include <poplar/Engine.hpp>
#include <poplibs_support/LogArithmetic.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/Reduce.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

// Tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;
using namespace poplibs_support;
using namespace poplibs_test;
using namespace poplibs_test::util;
using namespace poplar_test;

const OptionFlags options;

// Initialise value for a given type of computation
static double initValue(popops::Operation operation) {
  double val = 0.0;
  switch (operation) {
  case popops::Operation::ADD:
  case popops::Operation::SQUARE_ADD:
    val = 0.0;
    break;
  case popops::Operation::LOG_ADD:
    val = log::probabilityZero;
    break;
  case popops::Operation::MUL:
    val = 1.0;
    break;
  case popops::Operation::MIN:
    val = std::numeric_limits<double>::max();
    break;
  case popops::Operation::MAX:
    val = std::numeric_limits<double>::lowest();
    break;
  case popops::Operation::LOGICAL_AND:
    val = 1.0;
    break;
  case popops::Operation::LOGICAL_OR:
    val = 0.0;
    break;
  }
  return val;
}

// Perform a binary operation
static double doComputation(double x, double y, popops::Operation comp) {
  double res = y;
  switch (comp) {
  case popops::Operation::ADD:
    res += x;
    break;
  case popops::Operation::SQUARE_ADD:
    res += x * x;
    break;
  case popops::Operation::LOG_ADD:
    res = log::add<double>(res, x);
    break;
  case popops::Operation::MUL:
    res *= x;
    break;
  case popops::Operation::MIN:
    res = std::min(res, x);
    break;
  case popops::Operation::MAX:
    res = std::max(res, x);
    break;
  case popops::Operation::LOGICAL_AND:
    res = res && x;
    break;
  case popops::Operation::LOGICAL_OR:
    res = res || x;
    break;
  }
  return res;
}

// Reduce across given dimensions. The general case where there is no
// restriction on outDims is not coded yet (i.e. dimensions to reduce
// must be less than the number of dimensions to reduce
static void reduceTensor(boost::multi_array_ref<double, 3> in,
                         boost::multi_array_ref<double, 1> out,
                         const std::vector<std::size_t> &outDims,
                         popops::Operation operation) {

  if (outDims.size() == 3) {
    for (auto i2 = 0U; i2 != in.shape()[2]; ++i2) {
      for (auto i1 = 0U; i1 != in.shape()[1]; ++i1) {
        for (auto i0 = 0U; i0 != in.shape()[0]; ++i0) {
          out[i0 * in.shape()[1] * in.shape()[2] + i1 * in.shape()[2] + i2] =
              in[i0][i1][i2];
        }
      }
    }
  } else if (outDims.size() == 2) {
    for (auto i2 = 0U; i2 != in.shape()[2]; ++i2) {
      for (auto i1 = 0U; i1 != in.shape()[1]; ++i1) {
        auto res = initValue(operation);
        for (auto i0 = 0U; i0 != in.shape()[0]; ++i0) {
          res = doComputation(in[i0][i1][i2], res, operation);
        }
        out[i1 * in.shape()[2] + i2] = res;
      }
    }
  } else if (outDims.size() == 1) {
    for (auto i2 = 0U; i2 != in.shape()[2]; ++i2) {
      auto res = initValue(operation);
      for (auto i1 = 0U; i1 != in.shape()[1]; ++i1) {
        for (auto i0 = 0U; i0 != in.shape()[0]; ++i0) {
          res = doComputation(in[i0][i1][i2], res, operation);
        }
      }
      out[i2] = res;
    }
  } else if (outDims.size() == 0) {
    double res = initValue(operation);
    for (auto i2 = 0U; i2 != in.shape()[2]; ++i2) {
      for (auto i1 = 0U; i1 != in.shape()[1]; ++i1) {
        for (auto i0 = 0U; i0 != in.shape()[0]; ++i0) {
          res = doComputation(in[i0][i1][i2], res, operation);
        }
      }
    }
    out[0] = res;
  }
}

// Reduce across 1st dimension
static void reduceTensor(boost::multi_array_ref<double, 2> in,
                         boost::multi_array_ref<double, 1> out) {
  assert(in.shape()[1] == out.shape()[0]);
  std::size_t rows = in.shape()[0];
  std::size_t cols = in.shape()[1];
  for (auto c = 0U; c != cols; ++c) {
    double sum = 0;
    for (auto r = 0U; r != rows; ++r) {
      sum += in[r][c];
    }
    out[c] = sum;
  }
}

static bool reduceAddTest(const DeviceType &deviceType,
                          const std::vector<std::size_t> &dims,
                          const Type &partialsType, const Type &outType,
                          float k, bool update, bool scale, unsigned int many) {
  auto device = createTestDevice(deviceType, 1, 64);
  auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  assert(!(scale && update));
  assert(!(scale && many != 0));
  assert(dims.size() == 2);

  auto in = graph.addVariable(partialsType, {dims}, "in");
  poputil::mapTensorLinearly(graph, in);

  auto prev = graph.addVariable(outType, {dims[1]}, "prev");
  poputil::mapTensorLinearly(graph, prev);

  auto prog = Sequence();
  std::vector<Tensor> outs;

  if (scale) {
    auto rate = graph.addVariable(FLOAT, {});
    graph.setTileMapping(rate, 0);
    graph.setInitialValue(rate, k);
    outs.push_back(popops::reduce(graph, in, outType, {0},
                                  {popops::Operation::ADD, false, rate}, prog));
  } else if (many) {
    auto rate = graph.addConstant(FLOAT, {}, k);
    graph.setTileMapping(rate, 0);
    std::vector<SingleReduceOp> reductions;
    reductions.reserve(many);
    for (size_t i = 0; i < many; ++i) {
      reductions.push_back(SingleReduceOp{
          in, {0}, {popops::Operation::ADD, update, rate}, outType});
    }
    if (update) {
      outs.reserve(many);
      for (size_t i = 0; i < many; ++i) {
        outs.push_back(graph.clone(prev));
        prog.add(Copy(prev, outs.back()));
      }
    }
    popops::reduceMany(graph, reductions, outs, prog);
  } else if (update) {
    auto rate = graph.addVariable(FLOAT, {});
    graph.setTileMapping(rate, 0);
    graph.setInitialValue(rate, k);
    outs.push_back(graph.clone(prev));
    prog.add(Copy(prev, outs.back()));
    popops::reduceWithOutput(graph, in, outs[0], {0},
                             {popops::Operation::ADD, true, rate}, prog);
  } else {
    outs.push_back(
        popops::reduce(graph, in, outType, {0}, popops::Operation::ADD, prog));
  }

  std::vector<std::string> outNames;
  outNames.reserve(outs.size());
  for (size_t i = 0; i < outs.size(); ++i)
    outNames.push_back("out_" + std::to_string(i));

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, HostMemory>> tmap;
  auto rawHostPrev = allocateHostMemoryForTensor(
      prev, "prev", graph, uploadProg, downloadProg, tmap);
  auto rawHostIn = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                               downloadProg, tmap);
  std::vector<std::unique_ptr<char[]>> rawHostOuts;
  rawHostOuts.reserve(outs.size());
  for (size_t i = 0; i < outs.size(); ++i)
    rawHostOuts.push_back(allocateHostMemoryForTensor(
        outs[i], outNames[i], graph, uploadProg, downloadProg, tmap));

  boost::multi_array<double, 1> hostPrev(boost::extents[dims[1]]);
  boost::multi_array<double, 2> hostIn(boost::extents[dims[0]][dims[1]]);
  std::vector<boost::multi_array<double, 1>> hostOuts;
  hostOuts.reserve(outs.size());
  for (size_t i = 0; i < outs.size(); ++i)
    hostOuts.emplace_back(boost::extents[dims[1]]);

  std::mt19937 randomEngine;
  for (size_t i = 0; i < hostOuts.size(); ++i)
    std::fill(hostOuts[i].data(),
              hostOuts[i].data() + hostOuts[i].num_elements(), 0);
  writeRandomValues(target, partialsType, hostPrev, -1.0, +5.0, randomEngine);
  writeRandomValues(target, partialsType, hostIn, 1.5, 1.6, randomEngine);

  for (size_t i = 0; i < rawHostOuts.size(); ++i)
    copy(target, hostOuts[i], outType, rawHostOuts[i].get());
  copy(target, hostPrev, outType, rawHostPrev.get());
  copy(target, hostIn, partialsType, rawHostIn.get());

  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, options);
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);
    engine.run(0); // Run.
  });

  for (size_t i = 0; i < rawHostOuts.size(); ++i)
    copy(target, outType, rawHostOuts[i].get(), hostOuts[i]);

  boost::multi_array<double, 1> modelReduced(boost::extents[dims[1]]);

  reduceTensor(hostIn, modelReduced);

  boost::multi_array<double, 1> modelOut(boost::extents[dims[1]]);

  double kp = 0, kn = 1.0;
  if (update) {
    kp = 1.0;
    kn = k;
  } else if (scale || many)
    kn = k;

  for (auto c = 0U; c != dims[1]; ++c) {
    modelOut[c] = kp * hostPrev[c] + kn * modelReduced[c];
  }

  const double absoluteTolerance =
      outType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;
  const double relativeTolerance =
      outType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;

  bool allMatchModel = true;
  for (size_t i = 0; i < rawHostOuts.size(); ++i)
    allMatchModel =
        allMatchModel && checkIsClose(outNames[i], hostOuts[i], modelOut,
                                      relativeTolerance, absoluteTolerance);
  return allMatchModel;
}

static bool reduceOpsTest(const DeviceType &deviceType,
                          const std::vector<std::size_t> &dims,
                          const std::vector<std::size_t> &redVect,
                          const Type &outType, popops::Operation operation) {
  auto device = createTestDevice(deviceType, 1, 64);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  assert(dims.size() == 3);
  assert(redVect.size() <= 3);

  auto in = graph.addVariable(outType, {dims}, "in");
  poputil::mapTensorLinearly(graph, in);

  auto prog = Sequence();
  Tensor out = popops::reduce(graph, in, redVect, operation, prog);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, HostMemory>> tmap;
  auto rawHostIn = allocateHostMemoryForTensor(in, "in", graph, uploadProg,
                                               downloadProg, tmap);
  auto rawHostOut = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                                downloadProg, tmap);

  // check reduction dimensions: restricted set allowed
#ifndef NDEBUG
  for (const auto i : redVect) {
    assert(i < redVect.size());
  }
#endif

  // find output dims
  std::vector<std::size_t> outDims;
  std::size_t numOutElements = 1ULL;
  for (std::size_t i = 0; i != in.rank(); ++i) {
    if (std::find(redVect.begin(), redVect.end(), i) == redVect.end()) {
      numOutElements *= in.dim(i);
      outDims.push_back(i);
    }
  }

  boost::multi_array<double, 3> hostIn(
      boost::extents[dims[0]][dims[1]][dims[2]]);

  // keep flattened outputs
  boost::multi_array<double, 1> hostOut(boost::extents[numOutElements]);

  std::mt19937 randomEngine;
  std::fill(hostOut.data(), hostOut.data() + hostOut.num_elements(), 0);
  writeRandomValues(target, outType, hostIn, -2., 2., randomEngine);

  copy(target, hostOut, outType, rawHostOut.get());
  copy(target, hostIn, outType, rawHostIn.get());

  Engine engine(graph, Sequence{uploadProg, prog, downloadProg}, options);
  device.bind([&](const Device &d) {
    engine.load(d);
    attachStreams(engine, tmap);

    engine.run(0); // Run.
  });

  copy(target, outType, rawHostOut.get(), hostOut);

  boost::multi_array<double, 1> modelReduced(boost::extents[numOutElements]);

  reduceTensor(hostIn, modelReduced, outDims, operation);

  const double absoluteTolerance =
      outType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;
  const double relativeTolerance =
      outType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;

  auto matchesModel = checkIsClose("out", hostOut, modelReduced,
                                   relativeTolerance, absoluteTolerance);
  return matchesModel;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType;
  float k;
  Type outType;
  Type partialsType;
  ShapeOption<std::size_t> dims;
  ShapeOption<std::size_t> redVect;
  bool update;
  bool scale;
  unsigned many;
  Operation operation;
  std::string test;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Print help")
    ("device-type",
     po::value<DeviceType>(&deviceType)->required(),
     "Device Type")
    ("k",
     po::value<float>(&k)->default_value(NAN),
     "k")
    ("out-type",
     po::value<Type>(&outType)->required(),
     "Output Type")
    ("partials-type",
     po::value<Type>(&partialsType)->default_value(FLOAT),
     "Partials Type")
    ("dims",
     po::value<ShapeOption<std::size_t>>(&dims)->required(),
     "Dimensions")
    ("red-vect",
     po::value<ShapeOption<std::size_t>>(&redVect)
      ->default_value(ShapeOption<std::size_t>()),
     "Reduction vector")
    ("update",
     po::value<bool>(&update)->default_value(false),
     "Update")
    ("scale",
     po::value<bool>(&scale)->default_value(false),
     "Scale")
    ("many",
     po::value<decltype(many)>(&many)->default_value(0),
     "Number of reductions to do in parallel")
    ("operation",
     po::value<Operation>(&operation)->default_value(Operation::ADD),
     "The operation to perform (ADD, SQUARE_ADD, LOG_ADD, MUL, MIN, MAX,"
                                 " LOGICAL_AND or LOGICAL_OR)."
                                 " Log add used for log scaled probabilities")
    ("test",
     po::value<std::string>(&test)->required(),
     "Test: Add | Ops");
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

  if (test == "Add") {
    if (vm["k"].defaulted() || vm["partials-type"].defaulted() ||
        vm["update"].defaulted() || vm["scale"].defaulted()) {
      std::cerr << "k, partials-type, update and scale options are required"
                << "for Reduction Add test." << std::endl;
      return 1;
    }

    auto matchesModel = reduceAddTest(deviceType, dims.val, partialsType,
                                      outType, k, update, scale, many);
    return matchesModel ? 0 : 1;
  } else if (test == "Ops") {
    if (vm["red-vect"].defaulted() || vm["operation"].defaulted()) {
      std::cerr << "red-vect and operation options are required for"
                << "Reduction Ops test." << std::endl;
      return 1;
    }

    auto matchesModel =
        reduceOpsTest(deviceType, dims.val, redVect.val, outType, operation);
    return matchesModel ? 0 : 1;
  } else {
    std::cerr << "Unknown test '" << test << "'";
    return 1;
  }
}
