// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE MultiArray
#include <boost/test/unit_test.hpp>

#include "poplibs_support/MultiArray.hpp"
#include "poplibs_test/Util.hpp"

#include <poplar/Target.hpp>
#include <poplar_test/Util.hpp>

#include <boost/multi_array.hpp>

#include <random>

using namespace poplibs_support;
using namespace poplar_test;
using namespace poplibs_test::util;

BOOST_AUTO_TEST_CASE(CompareWithBoostMultiArray) {
  const std::array<std::size_t, 5> dims{1, 2, 3, 4, 5};

  MultiArray<double> uut{dims[0], dims[1], dims[2], dims[3], dims[4]};
  boost::multi_array<double, 5> model{
      boost::extents[dims[0]][dims[1]][dims[2]][dims[3]][dims[4]]};
  BOOST_CHECK(model.storage_order() == boost::c_storage_order());

  BOOST_CHECK(uut.size() == model.size());
  BOOST_CHECK(uut.numElements() == model.num_elements());
  BOOST_CHECK(uut.numDimensions() == model.num_dimensions());

  const auto uutShape = uut.shape();
  BOOST_CHECK(uutShape.size() == uut.numDimensions());
  BOOST_CHECK(uutShape.size() == dims.size());
  BOOST_CHECK(uutShape[0] == dims[0]);
  BOOST_CHECK(uutShape[1] == dims[1]);
  BOOST_CHECK(uutShape[2] == dims[2]);
  BOOST_CHECK(uutShape[3] == dims[3]);
  BOOST_CHECK(uutShape[4] == dims[4]);

  const auto modelShape = model.shape();
  BOOST_CHECK(std::equal(std::begin(uutShape), std::end(uutShape), modelShape));

  std::fill_n(uut.data(), uut.numElements(), 42.7);
  for (unsigned i = 0; i < dims[0]; ++i) {
    for (unsigned j = 0; j < dims[1]; ++j) {
      for (unsigned k = 0; k < dims[2]; ++k) {
        for (unsigned l = 0; l < dims[3]; ++l) {
          for (unsigned m = 0; m < dims[4]; ++m) {
            BOOST_CHECK(uut[i][j][k][l][m] == 42.7);

            MultiArrayShape indices{i, j, k, l, m};
            BOOST_CHECK(uut[indices] == 42.7);
          }
        }
      }
    }
  }

  std::mt19937 gen;
  std::uniform_real_distribution<> dist(-100.0, 100.0);
  std::generate_n(model.data(), model.num_elements(),
                  [&] { return dist(gen); });

  for (unsigned i = 0; i < dims[0]; ++i) {
    for (unsigned j = 0; j < dims[1]; ++j) {
      for (unsigned k = 0; k < dims[2]; ++k) {
        for (unsigned l = 0; l < dims[3]; ++l) {
          for (unsigned m = 0; m < dims[4]; ++m) {
            uut[i][j][k][l][m] = model[i][j][k][l][m];
            BOOST_CHECK(uut[i][j][k][l][m] == model[i][j][k][l][m]);
          }
        }
      }
    }
  }

  BOOST_CHECK(
      std::equal(uut.data(), uut.data() + uut.numElements(), model.data()));

  MultiArray<double> uutClone{dims[0], dims[1], dims[2], dims[3], dims[4]};
  for (unsigned i = 0; i < dims[0]; ++i) {
    for (unsigned j = 0; j < dims[1]; ++j) {
      for (unsigned k = 0; k < dims[2]; ++k) {
        for (unsigned l = 0; l < dims[3]; ++l) {
          for (unsigned m = 0; m < dims[4]; ++m) {
            MultiArrayShape indices{i, j, k, l, m};
            uutClone[indices] = uut[i][j][k][l][m];
            BOOST_CHECK(uutClone[i][j][k][l][m] == uut[indices]);
          }
        }
      }
    }
  }
  BOOST_CHECK(checkIsClose("uut", uut, uutClone, 0.1));

  const auto target = poplar::Target::createIPUTarget(1, "ipu1");
  const auto size = uut.numElements() * sizeof(float);

  std::unique_ptr<char[]> uutBuffer(new char[size]);
  copy(target, uut, poplar::FLOAT, uutBuffer.get());

  std::unique_ptr<char[]> modelBuffer(new char[size]);
  copy(target, model, poplar::FLOAT, modelBuffer.get());

  BOOST_CHECK(
      std::equal(uutBuffer.get(), uutBuffer.get() + size, modelBuffer.get()));

  MultiArray<double> uutAfter{dims[0], dims[1], dims[2], dims[3], dims[4]};
  boost::multi_array<double, 5> modelAfter{
      boost::extents[dims[0]][dims[1]][dims[2]][dims[3]][dims[4]]};

  copy(target, poplar::FLOAT, uutBuffer.get(), uutAfter);
  copy(target, poplar::FLOAT, modelBuffer.get(), modelAfter);
  BOOST_CHECK(std::equal(uutAfter.data(),
                         uutAfter.data() + uutAfter.numElements(),
                         modelAfter.data()));
}

BOOST_AUTO_TEST_CASE(TestForEachIndex) {
  MultiArrayShape shape{4, 3, 2};

  std::vector<std::array<std::size_t, 3>> result;
  forEachIndex(shape, [&](MultiArrayShapeRange indices) {
    BOOST_CHECK(indices.size() == 3);
    result.push_back({indices[0], indices[1], indices[2]});
  });

  std::sort(std::begin(result), std::end(result));

  std::vector<std::array<std::size_t, 3>> expected{
      {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {0, 2, 0}, {0, 2, 1},
      {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}, {1, 2, 0}, {1, 2, 1},
      {2, 0, 0}, {2, 0, 1}, {2, 1, 0}, {2, 1, 1}, {2, 2, 0}, {2, 2, 1},
      {3, 0, 0}, {3, 0, 1}, {3, 1, 0}, {3, 1, 1}, {3, 2, 0}, {3, 2, 1}};
  BOOST_CHECK(result == expected);
}
