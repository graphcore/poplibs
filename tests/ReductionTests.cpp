#define BOOST_TEST_MODULE ReductionTests

#include <boost/test/unit_test.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <popops/codelets.hpp>
#include <popops/Reduce.hpp>
#include <poplibs_test/Util.hpp>
#include <iostream>
#include <functional>
#include <limits>
#include <boost/multi_array.hpp>

// Tolerances used in tests
#define FLOAT_REL_TOL  0.01
#define HALF_REL_TOL   0.1
#define FLOAT_ABS_TOL  1e-6
#define HALF_ABS_TOL   1e-5

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;
using namespace poplibs_test::util;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

// Initialise value for a given type of computation
static double initValue(popops::Operation operation) {
  double val = 0.0;
  switch (operation) {
  case popops::Operation::ADD:
  case popops::Operation::SQUARE_ADD:
    val = 0.0;
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

static bool reduceAddTest(const std::vector<std::size_t> &dims,
                          const Type &partialsType,
                          const Type &outType,
                          float k,
                          bool update,
                          bool scale) {
  IPUModel ipuModel;
  ipuModel.tilesPerIPU = 64;
  auto device = ipuModel.createDevice();
  const auto &target = device.getTarget();
  Graph graph(device);
  popops::addCodelets(graph);

  assert(!(scale && update));
  assert(dims.size() == 2);

  auto in = graph.addVariable(partialsType, {dims}, "in");
  poputil::mapTensorLinearly(graph, in);

  auto prev = graph.addVariable(outType, {dims[1]}, "prev");
  poputil::mapTensorLinearly(graph, prev);

  auto prog = Sequence();
  Tensor out;

  if (scale) {
    out = popops::reduce(graph, in, outType, {0},
                         {popops::Operation::ADD, k}, prog);
  } else if (update) {
    out = graph.clone(prev);
    prog.add(Copy(prev, out));
    popops::reduceWithOutput(graph, in, out, {0},
                             {popops::Operation::ADD, k, true}, prog);
  } else {
    out = popops::reduce(graph, in, outType, {0},
                         popops::Operation::ADD, prog);
  }

  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostPrev =
          allocateHostMemoryForTensor(prev, "prev", graph, tmap);
  auto rawHostIn =
          allocateHostMemoryForTensor(in, "in", graph, tmap);
  auto rawHostOut =
          allocateHostMemoryForTensor(out, "out", graph, tmap);

  boost::multi_array<double, 1>
      hostPrev(boost::extents[dims[1]]);
  boost::multi_array<double, 2>
      hostIn(boost::extents[dims[0]][dims[1]]);
  boost::multi_array<double, 1>
      hostOut(boost::extents[dims[1]]);

  std::mt19937 randomEngine;
  std::fill(hostOut.data(), hostOut.data() + hostOut.num_elements(), 0);
  writeRandomValues(target, partialsType, hostPrev, -1.0, +5.0, randomEngine);
  writeRandomValues(target, partialsType, hostIn, 1.5, 1.6, randomEngine);

  copy(target, hostOut, outType, rawHostOut.get());
  copy(target, hostPrev, outType, rawHostPrev.get());
  copy(target, hostIn, partialsType, rawHostIn.get());

  Engine engine(device, graph, prog);

  upload(engine, tmap);
  engine.run(0); // Run.
  download(engine, tmap);

  copy(target, outType, rawHostOut.get(), hostOut);

  boost::multi_array<double, 1>
      modelReduced(boost::extents[dims[1]]);

  reduceTensor(hostIn, modelReduced);

  boost::multi_array<double, 1>
      modelOut(boost::extents[dims[1]]);

  double kp = 0, kn = 1.0;
  if (scale) {
    kn = k;
  } else if (update) {
    kp = 1.0; kn = k;
  }

  for (auto c = 0U; c != dims[1]; ++c) {
    modelOut[c] = kp * hostPrev[c] + kn * modelReduced[c];
  }

  const double absoluteTolerance = outType == FLOAT ? FLOAT_ABS_TOL :
                                                      HALF_ABS_TOL;
  const double relativeTolerance = outType == FLOAT ? FLOAT_REL_TOL :
                                                      HALF_REL_TOL;

  auto matchesModel = checkIsClose("out", hostOut, modelOut,
                                   relativeTolerance, absoluteTolerance);
  return matchesModel;
}

static bool reduceOpsTest(const std::vector<std::size_t> &dims,
                          const std::vector<std::size_t> &redVect,
                          const Type &outType,
                          popops::Operation operation) {
  IPUModel ipuModel;
  ipuModel.tilesPerIPU = 64;
  auto device = ipuModel.createDevice();
  const auto &target = device.getTarget();
  Graph graph(device);
  popops::addCodelets(graph);

  assert(dims.size() == 3);
  assert(redVect.size() <= 3);

  auto in = graph.addVariable(outType, {dims}, "in");
  poputil::mapTensorLinearly(graph, in);

  auto prog = Sequence();
  Tensor out =  popops::reduce(graph, in, redVect, operation, prog);

  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostIn =
          allocateHostMemoryForTensor(in, "in", graph, tmap);
  auto rawHostOut =
          allocateHostMemoryForTensor(out, "out", graph, tmap);

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
    if (std::find(redVect.begin(), redVect.end(), i)
        == redVect.end()) {
      numOutElements *= in.dim(i);
      outDims.push_back(i);
    }
  }

  boost::multi_array<double, 3>
      hostIn(boost::extents[dims[0]][dims[1]][dims[2]]);

  // keep flattened outputs
  boost::multi_array<double, 1> hostOut(boost::extents[numOutElements]);

  std::mt19937 randomEngine;
  std::fill(hostOut.data(), hostOut.data() + hostOut.num_elements(), 0);
  writeRandomValues(target, outType, hostIn, -2, 2, randomEngine);

  if (outType == BOOL) {
    for (auto it = hostIn.data(); it != hostIn.data() + hostIn.num_elements();
          ++it) {
      *it = *it <= 0 ? 0 : 1;
    }
  } else if (outType == INT) {
    for (auto it = hostIn.data(); it != hostIn.data() + hostIn.num_elements();
         ++it) {
      *it = std::floor(*it);
    }
  }

  copy(target, hostOut, outType, rawHostOut.get());
  copy(target, hostIn, outType, rawHostIn.get());

  Engine engine(device, graph, prog);

  upload(engine, tmap);
  engine.run(0); // Run.
  download(engine, tmap);

  copy(target, outType, rawHostOut.get(), hostOut);

  boost::multi_array<double, 1>
      modelReduced(boost::extents[numOutElements]);

  reduceTensor(hostIn, modelReduced, outDims, operation);

  const double absoluteTolerance = outType == FLOAT ? FLOAT_ABS_TOL :
                                                      HALF_ABS_TOL;
  const double relativeTolerance = outType == FLOAT ? FLOAT_REL_TOL :
                                                      HALF_REL_TOL;

  auto matchesModel = checkIsClose("out", hostOut, modelReduced,
                                   relativeTolerance, absoluteTolerance);
  return matchesModel;
}

BOOST_AUTO_TEST_CASE(Reduce_100x100_float_float_noupdate) {
  auto matchesModel = reduceAddTest({100, 100}, FLOAT, FLOAT,
                                     1.0, false, false);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_10x200_half_half) {
  auto matchesModel = reduceAddTest({10, 200}, HALF, HALF,
                                     2.0, false, false);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_31x201_scale_half_half) {
  auto matchesModel = reduceAddTest({31, 201}, HALF, HALF,
                                     3.0, false, true);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_31x201_scale_float_half) {
  auto matchesModel = reduceAddTest({31, 201}, FLOAT, HALF,
                                    -1.5, false, true);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_1x201_scale_float_half) {
  auto matchesModel = reduceAddTest({1, 201}, FLOAT, HALF,
                                    -1.5, false, true);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_1x201_scale_half_half) {
  auto matchesModel = reduceAddTest({1, 201}, HALF, HALF,
                                    -1.5, false, true);
  BOOST_TEST(matchesModel == true);
}


BOOST_AUTO_TEST_CASE(Reduce_31x201_update_float_float) {
  auto matchesModel = reduceAddTest({31, 101}, FLOAT, FLOAT,
                                    -1.5, true, false);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_31x201_update_half_half) {
  auto matchesModel = reduceAddTest({31, 201}, HALF, HALF,
                                    2.0, true, false);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_31x201_update_float_half) {
  auto matchesModel = reduceAddTest({31, 201}, FLOAT, HALF,
                                    -1.5, true, false);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Add_float) {
  auto matchesModel = reduceOpsTest({10, 20, 30}, {0}, FLOAT,
                                    popops::Operation::ADD);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Add_half) {
  auto matchesModel = reduceOpsTest({10, 20, 30}, {0}, HALF,
                                    popops::Operation::ADD);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Add_int) {
  auto matchesModel = reduceOpsTest({10, 20, 30}, {0}, INT,
                                    popops::Operation::ADD);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_SquareAdd_float) {
  auto matchesModel = reduceOpsTest({10, 20, 30}, {0}, FLOAT,
                                    popops::Operation::SQUARE_ADD);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_SquareAdd_half) {
  auto matchesModel = reduceOpsTest({10, 20, 30}, {0}, HALF,
                                    popops::Operation::SQUARE_ADD);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_SquareAdd_int) {
  auto matchesModel = reduceOpsTest({10, 20, 30}, {0}, INT,
                                    popops::Operation::SQUARE_ADD);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Mul_float) {
  auto matchesModel = reduceOpsTest({33, 22, 11}, {0}, FLOAT,
                                    popops::Operation::MUL);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Mul_half) {
  auto matchesModel = reduceOpsTest({33, 22, 11}, {0}, FLOAT,
                                    popops::Operation::MUL);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Mul_int) {
  auto matchesModel = reduceOpsTest({33, 22, 11}, {0}, FLOAT,
                                    popops::Operation::MUL);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Max_float) {
  auto matchesModel = reduceOpsTest({20, 30, 40}, {0, 1}, HALF,
                                    popops::Operation::MAX);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Max_half) {
  auto matchesModel = reduceOpsTest({20, 30, 10}, {0, 1}, HALF,
  popops::Operation::MAX);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Max_int) {
  auto matchesModel = reduceOpsTest({20, 30, 10}, {0, 1}, HALF,
                                    popops::Operation::MAX);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Min_float) {
  auto matchesModel = reduceOpsTest({20, 30, 10}, {0, 1}, FLOAT,
                                    popops::Operation::MIN);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Min_half) {
  auto matchesModel = reduceOpsTest({20, 30, 10}, {0, 1}, FLOAT,
                                    popops::Operation::MIN);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Min_int) {
  auto matchesModel = reduceOpsTest({20, 30, 10}, {0, 1}, FLOAT,
                                    popops::Operation::MIN);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_And_bool) {
  auto matchesModel = reduceOpsTest({20, 30, 10}, {0, 1}, BOOL,
                                    popops::Operation::LOGICAL_AND);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Or_bool) {
  auto matchesModel = reduceOpsTest({20, 30, 10}, {0, 1}, BOOL,
                                    popops::Operation::LOGICAL_OR);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_All_ADD_float) {
  auto matchesModel = reduceOpsTest({20, 30, 11}, {1, 0, 2}, FLOAT,
                                    popops::Operation::ADD);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_None_ADD_float) {
  auto matchesModel = reduceOpsTest({20, 30, 11}, {}, FLOAT,
                                    popops::Operation::ADD);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_Skip_ADD_float) {
  auto matchesModel = reduceOpsTest({1, 1, 11}, {0, 1}, FLOAT,
                                    popops::Operation::ADD);
  BOOST_TEST(matchesModel == true);
}
