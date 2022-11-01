// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "../lib/popsparse/SparsePartitionerImpl.hpp"
#include "../lib/popsparse/SparseStorageInternal.hpp"
#include "poplar/Target.hpp"
#include "poplar/Type.hpp"
#include "poplibs_support/logging.hpp"
#include "poplibs_test/SparseMatrix.hpp"
#include "poplibs_test/Util.hpp"
#include "poplibs_test/exceptions.hpp"
#include "popsparse/MatMulParams.hpp"
#include "popsparse/PlanningCache.hpp"
#include "popsparse/SparsePartitioner.hpp"
#include "poputil/exceptions.hpp"
#include <poplibs_support/TestDevice.hpp>

#include <gccs/Algorithm.hpp>

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <random>

using namespace poplibs_support;
using namespace poplibs_test::util;
using namespace poplar_test;
using namespace popsparse::static_;
using namespace popsparse;

enum class SparseRepresentation { CSC, CSR, COO };

const char *asString(const SparseRepresentation &sp) {
  switch (sp) {
  case SparseRepresentation::CSC:
    return "csc";
  case SparseRepresentation::CSR:
    return "csr";
  case SparseRepresentation::COO:
    return "coo";
  }
  POPLIB_UNREACHABLE();
}

std::ostream &operator<<(std::ostream &os, const SparseRepresentation &sp) {
  return os << asString(sp);
}

std::istream &operator>>(std::istream &in, SparseRepresentation &sp) {
  std::string token;
  in >> token;
  if (token == "coo")
    sp = SparseRepresentation::COO;
  else if (token == "csr")
    sp = SparseRepresentation::CSR;
  else if (token == "csc")
    sp = SparseRepresentation::CSC;
  else
    throw poplibs_test::poplibs_test_error(
        "Unsupported sparse representation <" + token + ">");
  return in;
}

// Build CSR matrix
static popsparse::CSRMatrix<float>
buildCSRMatrix(const std::vector<size_t> &dimensions, double sparsityFactor,
               const std::array<std::size_t, 2> &blockDimensions = {1, 1}) {

  assert(dimensions[0] % blockDimensions[0] == 0 &&
         dimensions[1] % blockDimensions[1] == 0);

  using EType = float;
  std::mt19937 randomEngine;
  auto [nzValues, columnIndices, rowIndices] =
      poplibs_test::sparse::buildCSRMatrix<EType, std::size_t>(
          randomEngine, dimensions, {blockDimensions[0], blockDimensions[1]},
          sparsityFactor, {0, 0}, {0, 0}, 0, false);
  logging::popsparse::debug("NZ Values {} : {}", nzValues.size(), nzValues);
  logging::popsparse::debug("Columns Indices {} : {}", columnIndices.size(),
                            columnIndices);
  logging::popsparse::debug("Row Indices {} : {}", rowIndices.size(),
                            rowIndices);

  return popsparse::CSRMatrix<EType>(
      dimensions[0], dimensions[1], std::move(nzValues),
      std::move(columnIndices), std::move(rowIndices), blockDimensions);
}

static bool compareCSRMatrices(CSRMatrix<float> &csr1, CSRMatrix<float> &csr2) {
  canonicalizeCSR(csr1);
  canonicalizeCSR(csr2);
  if (csr1.numRows != csr2.numRows) {
    std::cerr << "\nmismatch in rows";
    return false;
  }
  if (csr1.numColumns != csr2.numColumns) {
    std::cerr << "\nmismatch in columns";
    return false;
  }
  if (csr1.nzValues != csr2.nzValues) {
    std::cerr << "\nmismatch in NZ values";
    return false;
  }
  if (csr1.rowIndices != csr2.rowIndices) {
    std::cerr << "\nmismatch in row indices";
    return false;
  }
  if (csr1.columnIndices != csr2.columnIndices) {
    std::cerr << "\nmismatch in column indices";
    return false;
  }
  return true;
}

static bool validatePartition(const poplar::Target &target,
                              const poplar::Type &dataType, std::size_t m,
                              std::size_t k, std::size_t n,
                              std::size_t blockLength, double sparsityFactor,
                              SparseRepresentation sp) {
  // create partitioner
  PlanningCache cache;
  auto params = MatMulParams::createForSparseDense(1, m, k, n);
  auto partitioner =
      Partitioner<float>(params, dataType, target, {}, &cache, "p-sparse");

  auto csrMatrix =
      buildCSRMatrix({m, k, n}, sparsityFactor, {blockLength, blockLength});

  switch (sp) {
  case SparseRepresentation::CSR: {
    auto dataImpl = partitioner.createSparsityDataImpl(csrMatrix);
    auto csrAfter = partitioner.sparsityDataImplToCSRMatrix(dataImpl);
    return compareCSRMatrices(csrMatrix, csrAfter);
  } break;
  case SparseRepresentation::CSC: {
    auto dataImpl =
        partitioner.createSparsityDataImpl(csrToCSC(m, k, csrMatrix));
    auto cscAfter = partitioner.sparsityDataImplToCSCMatrix(dataImpl);
    auto csrAfter = cscToCSR(m, k, cscAfter);
    return compareCSRMatrices(csrMatrix, csrAfter);
  } break;
  case SparseRepresentation::COO: {
    std::vector<std::size_t> rowIndices;
    for (unsigned r = 0; r != m / blockLength; ++r) {
      std::fill_n(std::back_inserter(rowIndices),
                  (csrMatrix.rowIndices[r + 1] - csrMatrix.rowIndices[r]) /
                      (blockLength * blockLength),
                  r * blockLength);
    }
    auto cooMatrix =
        COOMatrix<float>(m, k, csrMatrix.nzValues, csrMatrix.columnIndices,
                         rowIndices, {blockLength, blockLength});

    auto dataImpl = partitioner.createSparsityDataImpl(cooMatrix);
    auto cooAfter = partitioner.sparsityDataImplToCOOMatrix(dataImpl);
    auto csrAfter = cooToCSR(m, k, cooAfter);
    return compareCSRMatrices(csrMatrix, csrAfter);
  } break;
  }
  return false;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  double sparsityLevel = 0.1;
  DeviceType deviceType = DeviceType::IpuModel2;

  boost::optional<unsigned> tilesPerIPU;
  ShapeOption<std::size_t> matShape;
  std::size_t blockLength = 1;
  SparseRepresentation representation;
  poplar::Type dataType = poplar::HALF;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("matmul-shape",
     po::value<ShapeOption<std::size_t>>(&matShape)->required(),
     "Triplet representing matmul {Rows, Columns, Batches}")
    ("sparsity-level",
     po::value<double>(&sparsityLevel)->default_value(sparsityLevel),
     "Level of sparsity")
    ("data-type",
     po::value<poplar::Type>(&dataType)->default_value(dataType),
     "Type of input")
    ("block-length",
     po::value<std::size_t>(&blockLength)->default_value(blockLength),
     "Block length to use")
    ("tiles-per-ipu", po::value(&tilesPerIPU), "Number of tiles per IPU")
    ("sparse-representation", 
     po::value<SparseRepresentation>(&representation)->required(),
     "Sparse representation to use (one of: csr, csc, coo")
  ;
  // clang-format on
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  if (matShape.val.size() != 3) {
    throw poputil::poplibs_error("shape of matrix must be 3-dimensional");
  }

  const auto numIPUs = 1;
  auto device = tilesPerIPU
                    ? createTestDevice(deviceType, numIPUs, *tilesPerIPU, true)
                    : createTestDeviceFullSize(deviceType, numIPUs, true);
  const auto &target = device.getTarget();

  if (matShape[0] % blockLength != 0 || matShape[1] % blockLength != 0) {
    throw poputil::poplibs_error(
        "sparse matrix dimensions (" + std::to_string(matShape[0]) + "," +
        std::to_string(matShape[1]) + ") are not divisible by block lengyh (" +
        std::to_string(blockLength) + "," + std::to_string(blockLength));
  }

  auto success = validatePartition(target, dataType, matShape.val[0],
                                   matShape.val[1], matShape.val[2],
                                   blockLength, sparsityLevel, representation);
  std::cerr << "\n Test success ? " << success << "\n";
  return !success;
}
