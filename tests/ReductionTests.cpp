#define BOOST_TEST_MODULE ReductionTests

#include <boost/test/unit_test.hpp>
#include <popstd/TileMapping.hpp>
#include <poplar/Engine.hpp>
#include <poplar/HalfFloat.hpp>
#include <popstd/codelets.hpp>
#include <popreduce/codelets.hpp>
#include <popreduce/Reduce.hpp>
#include <poplib_test/Util.hpp>
#include <iostream>
#include <functional>
#include <boost/multi_array.hpp>

// Tolerances used in tests
#define FLOAT_REL_TOL  0.01
#define HALF_REL_TOL   0.1
#define FLOAT_ABS_TOL  1e-6
#define HALF_ABS_TOL   1e-5

using namespace poplar;
using namespace poplar::program;
using namespace popstd;
using namespace popreduce;
using namespace poplib_test::util;

namespace utf = boost::unit_test;
namespace fpc = boost::test_tools::fpc;

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

static bool reduceTest(const std::vector<unsigned> dims,
                       const std::string &partialsTypeStr,
                       const std::string &outTypeStr,
                       float k,
                       bool update,
                       bool scale) {
  Graph graph(createIPUModelDevice());
  popstd::addCodelets(graph);
  popreduce::addCodelets(graph);

  assert(!(scale && update));
  assert(dims.size() == 2);

  auto in = graph.addTensor(partialsTypeStr, {dims[0], dims[1]}, "in");
  popstd::mapTensorLinearly(graph, in);

  auto prev = graph.addTensor(outTypeStr, {dims[1]}, "prev");
  popstd::mapTensorLinearly(graph, prev);

  auto prog = Sequence();
  Tensor out;

  if (scale) {
    out = popreduce::reduceScale(graph, k, in, outTypeStr, prog);
  } else if (update) {
    out = graph.clone(prev);
    prog.add(Copy(prev, out));
    popreduce::reduceAcc(graph, out, k, in, prog);
  } else {
    out = popreduce::reduce(graph, in, prog);
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
  writeRandomValues(hostPrev, -1.0, +5.0, randomEngine);
  writeRandomValues(hostIn, 1.5, 1.6, randomEngine);

  copy(hostOut, outTypeStr, rawHostOut.get());
  copy(hostPrev, outTypeStr, rawHostPrev.get());
  copy(hostIn, partialsTypeStr, rawHostIn.get());

  Engine engine(graph, prog);

  upload(engine, tmap);
  engine.run(0); // Run.
  download(engine, tmap);

  copy(outTypeStr, rawHostOut.get(), hostOut);

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

  const double absoluteTolerance = outTypeStr == "float" ? FLOAT_ABS_TOL :
                                                           HALF_ABS_TOL;
  const double relativeTolerance = outTypeStr == "float" ? FLOAT_REL_TOL :
                                                           HALF_REL_TOL;

  auto matchesModel = checkIsClose("out", hostOut, modelOut,
                                   relativeTolerance, absoluteTolerance);
  return matchesModel;
}


BOOST_AUTO_TEST_CASE(Reduce_100x100_float_float_noupdate) {
  auto matchesModel = reduceTest({100, 100}, "float", "float",
                                 1.0, false, false);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_10x200_half_half) {
  auto matchesModel = reduceTest({10, 200}, "half", "half",
                                 2.0, false, false);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_31x201_scale_half_half) {
  auto matchesModel = reduceTest({31, 201}, "half", "half",
                                 3.0, false, true);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_31x201_scale_float_half) {
  auto matchesModel = reduceTest({31, 201}, "float", "half",
                                 -1.5, false, true);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_1x201_scale_float_half) {
  auto matchesModel = reduceTest({1, 201}, "float", "half",
                                 -1.5, false, true);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_1x201_scale_half_half) {
  auto matchesModel = reduceTest({1, 201}, "half", "half",
                                 -1.5, false, true);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_31x201_update_float_float) {
  auto matchesModel = reduceTest({31, 201}, "float", "float",
                                 -1.5, true, false);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_31x201_update_half_half) {
  auto matchesModel = reduceTest({31, 201}, "half", "half",
                                 2.0, true, false);
  BOOST_TEST(matchesModel == true);
}

BOOST_AUTO_TEST_CASE(Reduce_31x201_update_float_half) {
  auto matchesModel = reduceTest({31, 201}, "float", "half",
                                 -1.5, true, false);
  BOOST_TEST(matchesModel == true);
}
