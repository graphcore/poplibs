// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

#define DEVELOP_BLOCKSPARSE

#include "poplibs_support/logging.hpp"
#include "poplibs_test/Util.hpp"
#include <poplar/IPUModel.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/codelets.hpp>
#include <popsparse/experimental/BlockSparse.hpp>
#include <popsparse/experimental/BlockSparseMatMul.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include "poplibs_test/Pass.hpp"
#include "poplibs_test/Util.hpp"

using namespace poplar;
using namespace poplin;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace popsparse::experimental;
using namespace poplibs_support;
using namespace poplibs_test;

const OptionFlags defaultEngineOptions{{"debug.allowOutOfMemory", "true"}};

bool ReadMatrixMask(std::string &fileName, int &nonZeroBlock, int &row,
                    int &col, std::vector<unsigned char> &mask,
                    bool readTransposed) {
  std::ifstream cin(fileName);
  if (!cin.is_open()) {
    std::cerr << " Can not open matrix sparsity mask file " << fileName
              << std::endl;
    return false;
  }

  cin >> row >> col;
  if (row <= 0 || col <= 0) {
    std::cerr << " Incorrect matrix sparsity mask file format " << fileName
              << std::endl;
    return false;
  }
  mask.resize(row * col);
  unsigned char *data = mask.data();

  if (!data) {
    return false;
  }

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      int c;
      cin >> c;
      if (!readTransposed) {
        data[i * col + j] = c ? 1 : 0;
      } else {
        data[j * row + i] = c ? 1 : 0;
      }
    }
  }

  nonZeroBlock = 0;
  for (int i = 0; i < row * col; i++) {
    if (data[i] == 1) {
      nonZeroBlock++;
    }
  }
  return true;
}

void populateMatrixData(const std::array<int, 3> &dim,
                        const std::array<int, 3> &blockSize,
                        boost::multi_array<float, 2> &lhsMatrix,
                        boost::multi_array<float, 2> &rhsMatrix,
                        const unsigned char *sparsity, bool rhsNeedTranspose) {

  std::mt19937 randomEngine;
  randomEngine.seed(102302);
  for (int i = 0; i < dim[0]; i++) {
    for (int j = 0; j < dim[1]; j++) {
      lhsMatrix[i][j] = static_cast<float>(randomEngine()) / INT_MAX;
    }
  }

  int blockRow;
  int blockCol;
  int nRowBlock;
  int nColBlock;
  if (!rhsNeedTranspose) {
    blockRow = blockSize[1];
    blockCol = blockSize[2];
    nRowBlock = dim[1] / blockSize[1];
    nColBlock = dim[2] / blockSize[2];
  } else {
    blockRow = blockSize[2];
    blockCol = blockSize[1];
    nRowBlock = dim[2] / blockSize[2];
    nColBlock = dim[1] / blockSize[1];
  }

  for (int i = 0; i < nRowBlock; i++) {
    for (int j = 0; j < nColBlock; j++) {
      int blockRowStart = i * blockRow;
      int blockColStart = j * blockCol;
      int blockRowEnd = blockRowStart + blockRow;
      int blockColEnd = blockColStart + blockCol;

      if (!sparsity || sparsity[i * nColBlock + j] == 1) {
        for (int r = blockRowStart; r < blockRowEnd; r++) {
          for (int c = blockColStart; c < blockColEnd; c++) {
            rhsMatrix[r][c] = static_cast<float>(randomEngine()) /
                              static_cast<float>(INT_MAX);
          }
        }
      }
    }
  }
}

void populateMatrixData(const std::array<int, 3> &dim,
                        const std::array<int, 3> &blockSize, int numGroups,
                        boost::multi_array<float, 3> &lhsMatrix,
                        boost::multi_array<float, 3> &rhsMatrix,
                        const unsigned char *sparsity, bool rhsNeedTranspose) {

  std::mt19937 randomEngine;
  randomEngine.seed(102302);
  for (int g = 0; g < numGroups; ++g) {
    for (int r = 0; r < dim[0]; r++) {
      for (int c = 0; c < dim[1]; c++) {
        lhsMatrix[g][r][c] = static_cast<float>(randomEngine()) / INT_MAX;
      }
    }
  }

  int blockRow;
  int blockCol;
  int nRowBlock;
  int nColBlock;
  if (!rhsNeedTranspose) {
    blockRow = blockSize[1];
    blockCol = blockSize[2];
    nRowBlock = dim[1] / blockSize[1];
    nColBlock = dim[2] / blockSize[2];
  } else {
    blockRow = blockSize[2];
    blockCol = blockSize[1];
    nRowBlock = dim[2] / blockSize[2];
    nColBlock = dim[1] / blockSize[1];
  }

  std::size_t sparsityDenseSize = nRowBlock * nColBlock;
  const unsigned char *itemSparsity = sparsity;
  for (int g = 0; g < numGroups; ++g) {
    for (int i = 0; i < nRowBlock; i++) {
      for (int j = 0; j < nColBlock; j++) {
        int blockRowStart = i * blockRow;
        int blockColStart = j * blockCol;
        int blockRowEnd = blockRowStart + blockRow;
        int blockColEnd = blockColStart + blockCol;
        if (!itemSparsity || itemSparsity[i * nColBlock + j] == 1) {
          for (int r = blockRowStart; r < blockRowEnd; r++) {
            for (int c = blockColStart; c < blockColEnd; c++) {
              rhsMatrix[g][r][c] = static_cast<float>(randomEngine()) /
                                   static_cast<float>(INT_MAX);
            }
          }
        }
      }
    }
    if (sparsity) {
      itemSparsity += sparsityDenseSize;
    }
  }
}

void populateMatrixData(const std::array<int, 2> &dim,
                        const std::array<int, 2> &blockSize,
                        boost::multi_array<float, 2> &matrix,
                        const unsigned char *sparsity) {
  std::mt19937 randomEngine;
  randomEngine.seed(102302);

  int blockRow = blockSize[0];
  int blockCol = blockSize[1];
  int nRowBlock = dim[0] / blockSize[0];
  int nColBlock = dim[1] / blockSize[1];

  for (int i = 0; i < nRowBlock; i++) {
    for (int j = 0; j < nColBlock; j++) {
      int blockRowStart = i * blockRow;
      int blockColStart = j * blockCol;
      int blockRowEnd = blockRowStart + blockRow;
      int blockColEnd = blockColStart + blockCol;

      float val = 0.0f;
      if (!sparsity || sparsity[i * nColBlock + j] == 1) {
        for (int r = blockRowStart; r < blockRowEnd; r++) {
          for (int c = blockColStart; c < blockColEnd; c++) {
            val = static_cast<float>(randomEngine()) /
                  static_cast<float>(INT_MAX);
            matrix[r][c] = val;
          }
        }
      }
    }
  }
}

void populateMatrixData(const std::array<int, 3> &dim,
                        boost::multi_array<float, 2> &lhsMatrix,
                        boost::multi_array<float, 2> &rhsMatrix) {

  std::mt19937 randomEngine;
  randomEngine.seed(102302);

  for (int i = 0; i < dim[0]; i++) {
    for (int j = 0; j < dim[1]; j++) {
      lhsMatrix[i][j] = static_cast<float>(randomEngine()) / INT_MAX;
    }
  }
  for (int i = 0; i < dim[1]; i++) {
    for (int j = 0; j < dim[2]; j++) {
      rhsMatrix[i][j] = static_cast<float>(randomEngine()) / INT_MAX;
    }
  }
}

void populateMatrixData(const std::array<int, 2> &dim,
                        boost::multi_array<float, 2> &matrix) {

  std::mt19937 randomEngine;
  randomEngine.seed(102302);

  for (int i = 0; i < dim[0]; i++) {
    for (int j = 0; j < dim[1]; j++) {
      matrix[i][j] = static_cast<float>(randomEngine()) / INT_MAX;
    }
  }
}

void printMatrix(std::string msg, float *matrix, int row, int col) {
#ifdef DEBUG
  std::cout << msg << std::endl;

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      std::cout << matrix[i * col + j] << " ";
    }
    std::cout << std::endl;
  }
#endif
}

// TODO: Move to some kind of utils
void getSparseMatrixBlocks(int rows, int cols, int rowsInBlock, int colsInBlock,
                           unsigned char *sparsity, unsigned nonZeroBlock,
                           const boost::multi_array<float, 2> &denseMat,
                           boost::multi_array<float, 1> &sparseMat) {
  const int blockRows = rows / rowsInBlock;
  const int blockCols = cols / colsInBlock;

  unsigned sparseSize = nonZeroBlock * rowsInBlock * colsInBlock;
  sparseMat.resize(boost::extents[sparseSize]);

  unsigned index = 0;
  for (int br = 0, idxDense = 0; br < blockRows; br++) {
    for (int bc = 0; bc < blockCols; ++bc, ++idxDense) {
      if (sparsity[idxDense] == 1) {
        int rowStart = br * rowsInBlock;
        int colStart = bc * colsInBlock;
        int rowEnd = rowStart + rowsInBlock;
        int colEnd = colStart + colsInBlock;
        for (int r = rowStart; r < rowEnd; r++) {
          for (int c = colStart; c < colEnd; c++) {
            assert(index < sparseSize);
            sparseMat[index++] = denseMat[r][c];
          }
        }
      }
    }
  }
  assert(index == sparseSize);
}

void getSparseMatrixBlocks(int rows, int cols, int rowsInBlock, int colsInBlock,
                           int numGroups, unsigned char *sparsity,
                           unsigned nonZeroBlock,
                           const boost::multi_array<float, 3> &denseMat,
                           boost::multi_array<float, 1> &sparseMat) {
  const int blockRows = rows / rowsInBlock;
  const int blockCols = cols / colsInBlock;

  unsigned sparseSize = nonZeroBlock * rowsInBlock * colsInBlock;
  sparseMat.resize(boost::extents[sparseSize]);

  std::size_t sparsityDenseSize = blockRows * blockCols;
  const unsigned char *itemSparsity = sparsity;
  unsigned index = 0;
  for (int g = 0; g < numGroups; ++g) {
    int blockCount = 0;
    for (int br = 0; br < blockRows; br++) {
      for (int bc = 0; bc < blockCols; bc++) {
        if (itemSparsity[br * blockCols + bc] == 1) {
          int rowStart = br * rowsInBlock;
          int colStart = bc * colsInBlock;
          int rowEnd = rowStart + rowsInBlock;
          int colEnd = colStart + colsInBlock;
          for (int r = rowStart; r < rowEnd; r++) {
            for (int c = colStart; c < colEnd; c++) {
              assert(index < sparseSize);
              sparseMat[index++] = denseMat[g][r][c];
            }
          }
          blockCount++;
        }
      }
    }
    itemSparsity += sparsityDenseSize;
  }
  assert(index == sparseSize);
}

BSMatMulParams createBsMatMul(const std::array<int, 3> &dim,
                              const std::array<int, 3> &blockSize,
                              const std::vector<unsigned char> &sparsity,
                              poplar::Type dataType, bool isResSparse,
                              bool rhsNeedTranspose, SubBlockMask mask,
                              unsigned int numGroups) {
  if (isResSparse) {
    return BSMatMulParams(dim, blockSize, sparsity, dataType, dataType,
                          dataType, mask, numGroups);
  } else
    return BSMatMulParams(dim, blockSize, sparsity, rhsNeedTranspose, dataType,
                          dataType, dataType, numGroups);
}

void savePoplarReport(poplar::Engine &engine, std::string &dir) {
  // Graph Report
  poplar::ProfileValue graphProfile = engine.getGraphProfile();
  std::ofstream graphReport;
  graphReport.open(dir + "/graph.json");
  poplar::serializeToJSON(graphReport, graphProfile);
  graphReport.close();

  // Execution Report
  poplar::ProfileValue execProfile = engine.getExecutionProfile();
  std::ofstream execReport;
  execReport.open(dir + "/execution.json");
  poplar::serializeToJSON(execReport, execProfile);
  execReport.close();
}

enum class OperandsType {
  null,
  d,   // 1 operand: dense
  s,   // 1 operand: sparse
  ddd, // 3 operands: dense, dense, dense
  dsd, // 3 operands: dense, sparse, dense
  dds, // 3 operands: dense, dense, sparse
};

enum Scenario {
  ddd = 0x0,
  dsd = 0x1,
  dds = 0x2,
  smd = 0x3,
  sms = 0x4,
  fwdMask = 0x7,
  bwd = 0x8,
  wu = 0x10,
  all = bwd | wu,
  dsdBwd = dsd | bwd,
  dsdWu = dsd | wu,
  dsdAll = dsdBwd | dsdWu,
  ddsBwd = dds | bwd,
  ddsWu = dds | wu,
  ddsAll = ddsBwd | ddsWu,
  smdBwd = smd | bwd,
  smsBwd = sms | bwd,
};

int main(int argc, char **argv) {
  // Default options
  DeviceType deviceType = DeviceType::Hw;
  std::string sparsityFileName = "";
  std::string profileDir = ".";
  std::string subBlockMask = "None";
  std::string scenarioName = "dsd";
  std::string partitionMethod = "strip";
  int lhsBlockRowSize0 = 8;
  int lhsBlockColSize0 = 8;
  int rhsBlockSize0 = 8;
  int lhsRows0 = 0;
  int lhsCols0 = 0;
  int rhsCols0 = 0;
  int lhsBlockCols0 = 0;
  int rhsNeedTranspose0 = 0;
  boost::optional<unsigned> tilesPerIPU;
  Type dataType = FLOAT;
  float memoryCycleRatio = 1;
  int runs = 1;
  int nPass = 1;
  int numReps = 1;
  int numGroups = 1;

  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()
      // help
      ("help", "Produce help message")
      // scenario
      ("scenario",
       po::value<std::string>(&scenarioName)->default_value(scenarioName),
       "Scenario: "
       "ddd|dsd|dds|dsd-bwd|dsd-wu|dsd-all|dds-bwd|dds-wu|dds-"
       "all|dds-sm|dds-sm-all|sm|smd|sms-bwd|smd-bwd\n"
       "ddd = dense x dense = dense\n"
       "dsd = dense x sparse = dense\n"
       "dds = dense x dense = sparse\n"
       "-bwd = also compute the gradient of the left matrix\n"
       "-wu = also compute the gradient of the right matrix\n"
       "-all = also compute the gradient of left and right matrices\n"
       "-sm = also compute softmax of the result matrix\n"
       "-all = also compute softmax gradient and then use it as an outer "
       "gradient and compute the gradient of left and right matrices\n"
       "sms = compute softmax only on a sparse matrix (LHS block dimensions "
       "used)\n"
       "smd = compute dense softmax on a dense matrix (LHS dimensions used)\n"
       "sms-bwd = compute softmax gradient only on a sparse matrix (LHS block "
       "dimensions used)\n"
       "smd-bwd = compute dense softmax gradient on a dense matrix (LHS "
       "dimensions used)\n")
      // compile-only
      ("compile-only", "Stop after compilation; don't run the program")
      // device-type
      ("device-type",
       po::value<DeviceType>(&deviceType)->default_value(deviceType),
       deviceTypeHelp)
      // tiles-per-ipu
      ("tiles-per-ipu", po::value(&tilesPerIPU), "Number of tiles per IPU")
      // data-type
      ("data-type", po::value<Type>(&dataType)->default_value(dataType),
       "matmul data type")
      // profile
      ("profile", "Output profiling report")
      // profile-execution
      ("profile-execution", "Output execution steps in the profiling report")
      // profile-vars
      ("profile-vars", "Output variables storage in the profiling report")
      // profile-dir
      ("profile-dir", po::value<std::string>(&profileDir),
       "The directory to output profiling report")
      // lhs-rows
      ("lhs-rows", po::value<int>(&lhsRows0),
       "The number of rows for the LHS matrix (for ddd, dsd-, smd- scenarios)")
      // lhs-cols
      ("lhs-cols", po::value<int>(&lhsCols0),
       "The number of cols for the LHS matrix (for ddd, smd- scenarios)")
      // rhs-cols
      ("rhs-cols", po::value<int>(&rhsCols0),
       "The number of cols for the RHS matrix (for ddd scenario)")
      // lhs-block-cols
      ("lhs-block-cols", po::value<int>(&lhsBlockCols0),
       "The number of blocks columns for the LHS matrix (for dds- scenarios)")
      // lhs-block-row
      ("lhs-block-row", po::value<int>(&lhsBlockRowSize0),
       "The block size for the row of rhe LHS matrix (for dsd-, dds, sms- "
       "scenarios)")
      // lhs-block-col
      ("lhs-block-col", po::value<int>(&lhsBlockColSize0),
       "The block size for the column of the LHS matrix (for dsd-, dds, sms- "
       "scenarios)")
      // rhs-block
      ("rhs-block", po::value<int>(&rhsBlockSize0),
       "The block col for the right matrix that does not need be transposed or "
       "the block row for the right matrix that needs be transposed (for dsd-, "
       "dds- scenarios)")
      // rhs-need-transpose
      ("rhs-need-transpose", po::value<int>(&rhsNeedTranspose0),
       "the right matrix need be transposed (for dsd- scenarios)")
      // sparsity-matrix
      ("sparsity-matrix",
       po::value<std::string>(&sparsityFileName)
           ->default_value(sparsityFileName),
       "The file name for the sparsity mask (for dsd-, dds-, sms- scenarios)")
      // memory-cycle-ratio
      ("memory-cycle-ratio", po::value<float>(&memoryCycleRatio),
       "the ratio between memory weight and cycle weight")
      // number-of-pass
      ("number-of-pass", po::value<int>(&nPass)->default_value(nPass),
       "number of pass")
      // sub-block-mask
      ("sub-block-mask",
       po::value<std::string>(&subBlockMask)->default_value(subBlockMask),
       "the mask inside a block: None, ZeroUpperTriangle, "
       "ZeroLowerTriangle")
      // partition-method
      ("partition-method",
       po::value<std::string>(&partitionMethod)->default_value(partitionMethod),
       "The method to generate the computation graph: block, strip")
      // runs
      ("runs", po::value<int>(&runs), "Number of calls to Engine::run")
      // number-or-reps
      ("number-of-reps", po::value<int>(&numReps), "Number of repetitions")
      // number-or-groups
      ("number-of-groups", po::value<int>(&numGroups), "Number of groups")
      // check-result
      ("check-result", "check if the result is correct");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help") || !vm.count("sparsity-matrix")) {
      std::cout << desc << std::endl;
      return -1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "Command line parsing error: " << e.what() << std::endl;
    return -1;
  }

  Scenario scenario = Scenario::ddd;
  // Up to 3 steps
  unsigned numSteps = 1;
  std::array<OperandsType, 3> operandsType = {
      OperandsType::null, OperandsType::null, OperandsType::null};
  std::array<Pass, 3> runType = {Pass::FWD, Pass::FWD, Pass::FWD};
  bool computeMatmul = true;
  bool computeSoftmax = false;
  if (scenarioName == "ddd") {
    scenario = Scenario::ddd;
    operandsType[0] = OperandsType::ddd;
    runType[0] = Pass::FWD;
  } else if (scenarioName == "dsd") {
    scenario = Scenario::dsd;
    operandsType[0] = OperandsType::dsd;
    runType[0] = Pass::FWD;
  } else if (scenarioName == "dds") {
    scenario = Scenario::dds;
    operandsType[0] = OperandsType::dds;
    runType[0] = Pass::FWD;
  } else if (scenarioName == "dds-sm") {
    scenario = Scenario::dds;
    operandsType[0] = OperandsType::dds;
    runType[0] = Pass::FWD;
    computeSoftmax = true;
  } else if (scenarioName == "smd") {
    scenario = Scenario::smd;
    operandsType[0] = OperandsType::d;
    runType[0] = Pass::FWD;
    computeMatmul = false;
    computeSoftmax = true;
  } else if (scenarioName == "sms") {
    scenario = Scenario::sms;
    operandsType[0] = OperandsType::s;
    runType[0] = Pass::FWD;
    computeMatmul = false;
    computeSoftmax = true;
  } else if (scenarioName == "smd-bwd") {
    scenario = Scenario::smdBwd;
    operandsType[0] = OperandsType::d;
    runType[0] = Pass::FWD;
    computeMatmul = false;
    computeSoftmax = true;
  } else if (scenarioName == "sms-bwd") {
    scenario = Scenario::smsBwd;
    operandsType[0] = OperandsType::s;
    runType[0] = Pass::FWD;
    computeMatmul = false;
    computeSoftmax = true;
  } else if (scenarioName == "dsd-bwd") {
    numSteps = 2;
    scenario = Scenario::dsdBwd;
    operandsType[0] = OperandsType::dsd;
    operandsType[1] = OperandsType::dsd;
    runType[0] = Pass::FWD;
    runType[1] = Pass::BWD;
  } else if (scenarioName == "dsd-wu") {
    numSteps = 2;
    scenario = Scenario::dsdWu;
    operandsType[0] = OperandsType::dsd;
    operandsType[1] = OperandsType::dds;
    runType[0] = Pass::FWD;
    runType[1] = Pass::WU;
  } else if (scenarioName == "dsd-all") {
    numSteps = 3;
    scenario = Scenario::dsdAll;
    operandsType[0] = OperandsType::dsd;
    operandsType[1] = OperandsType::dsd;
    operandsType[2] = OperandsType::dds;
    runType[0] = Pass::FWD;
    runType[1] = Pass::BWD;
    runType[2] = Pass::WU;
  } else if (scenarioName == "dds-bwd") {
    numSteps = 2;
    scenario = Scenario::ddsBwd;
    operandsType[0] = OperandsType::dds;
    operandsType[1] = OperandsType::dsd;
    runType[0] = Pass::FWD;
    runType[1] = Pass::BWD;
  } else if (scenarioName == "dds-wu") {
    numSteps = 2;
    scenario = Scenario::ddsWu;
    operandsType[0] = OperandsType::dds;
    operandsType[1] = OperandsType::dsd;
    runType[0] = Pass::FWD;
    runType[1] = Pass::WU;
  } else if (scenarioName == "dds-all") {
    numSteps = 3;
    scenario = Scenario::ddsAll;
    operandsType[0] = OperandsType::dds;
    operandsType[1] = OperandsType::dsd;
    operandsType[2] = OperandsType::dsd;
    runType[0] = Pass::FWD;
    runType[1] = Pass::BWD;
    runType[2] = Pass::WU;
  } else if (scenarioName == "dds-sm-all") {
    numSteps = 3;
    scenario = Scenario::ddsAll;
    operandsType[0] = OperandsType::dds;
    operandsType[1] = OperandsType::dsd;
    operandsType[2] = OperandsType::dsd;
    runType[0] = Pass::FWD;
    runType[1] = Pass::BWD;
    runType[2] = Pass::WU;
    computeSoftmax = true;
  } else {
    std::cerr << "--scenario: Unrecognized scenario: " << scenarioName
              << std::endl;
    return -1;
  }
  std::cout << scenarioName << " scenario" << std::endl;
  if (numGroups > 1) {
    std::cout << "Number of groups: " << numGroups << std::endl;
  }

  std::map<std::string, std::pair<int *, std::vector<unsigned int>>>
      optionsUsage = {
          {"lhs-rows",
           {&lhsRows0, {0, 1, Scenario::ddd, Scenario::dsd, Scenario::smd}}},
          {"lhs-cols", {&lhsCols0, {0, 1, Scenario::ddd, Scenario::smd}}},
          {"rhs-cols", {&rhsCols0, {0, 1, Scenario::ddd}}},
          {"lhs-block-cols", {&lhsBlockCols0, {0, 1, Scenario::dds}}},
          {"lhs-block-row",
           {&lhsBlockRowSize0,
            {0, 1, Scenario::dsd, Scenario::dds, Scenario::sms}}},
          {"lhs-block-col",
           {&lhsBlockColSize0,
            {0, 1, Scenario::dsd, Scenario::dds, Scenario::sms}}},
          {"rhs-block", {&rhsBlockSize0, {0, 1, Scenario::dsd, Scenario::dds}}},
          {"rhs-need-transpose", {&rhsNeedTranspose0, {1, 0, Scenario::dsd}}},
          {"sparsity-matrix",
           {nullptr, {0, 0, Scenario::dsd, Scenario::dds, Scenario::sms}}},
          {"sub-block-mask",
           {nullptr, {1, 0, Scenario::dsd, Scenario::dds, Scenario::sms}}},
          {"num-groups", {&numGroups, {1, 1, Scenario::dsd, Scenario::dds}}},
      };

  for (auto iter = optionsUsage.begin(); iter != optionsUsage.end(); ++iter) {
    const std::string option = iter->first;
    const std::vector<unsigned int> &usedIn = iter->second.second;
    int *optionValue = iter->second.first;
    // First 2 elements of usedIn has special meanings:
    unsigned int hasDefault = usedIn[0];
    int minValue = static_cast<int>(usedIn[1]);
    bool used = false;
    for (unsigned int i = 2; i < usedIn.size(); ++i) {
      if ((scenario & Scenario::fwdMask) == usedIn[i]) {
        used = true;
        break;
      }
    }
    if (vm.count(option) > 0) {
      if (!used) {
        logging::popsparse::warn(
            "Option {} is not used in scenario {}. Ignored.", option.c_str(),
            scenarioName.c_str());
      } else {
        if (optionValue && *optionValue < minValue) {
          std::cerr << "Option " << option << " minimum value is "
                    << *optionValue << std::endl;
          return -1;
        }
      }
    } else {
      if (used && !hasDefault) {
        std::cerr << "Option " << option << " is missing." << std::endl;
        return -1;
      }
    }
  }
  bool compileIPUCode = false;
  if (isIpuModel(deviceType))
    compileIPUCode = true;
  const unsigned numIPUs = 1;
  auto device =
      tilesPerIPU
          ? createTestDevice(deviceType, numIPUs, *tilesPerIPU, compileIPUCode)
          : createTestDeviceFullSize(deviceType, numIPUs, compileIPUCode);

  bool checkResult = vm.count("check-result") > 0;

  if (checkResult && (((scenario & Scenario::bwd) == Scenario::bwd) ||
                      ((scenario & Scenario::wu) == Scenario::wu))) {
    std::cerr << "Checking results in not supported in scenario "
              << scenarioName << std::endl;
    return -1;
  }

  if (checkResult && numReps > 1) {
    std::cerr << "Checking results in not supported for many repetitions."
              << std::endl;
    return -1;
  }

  if (((scenario & Scenario::fwdMask) != Scenario::dsd) &&
      ((scenario & Scenario::fwdMask) != Scenario::dds)) {
    numGroups = 1;
  }

  bool hasSparsity = ((scenario & Scenario::fwdMask) == Scenario::dsd) ||
                     ((scenario & Scenario::fwdMask) == Scenario::dds) ||
                     ((scenario & Scenario::fwdMask) == Scenario::sms);

  SubBlockMask mask = SubBlockMask::None;
  if (hasSparsity) {
    if (subBlockMask == "ZeroUpperTriangle") {
      mask = SubBlockMask::ZeroUpperTriangle;
    } else if (subBlockMask == "ZeroLowerTriangle") {
      mask = SubBlockMask::ZeroLowerTriangle;
    } else if (subBlockMask != "None") {
      std::cerr << "Unrecognized sub-block mask parameter: " << subBlockMask
                << std::endl;
      return -1;
    }
  }

  bool doProfiling = vm.count("profile") > 0;
  bool doProfilingExecution = doProfiling && vm.count("profile-execution") > 0;
  bool doProfilingVars = doProfiling && vm.count("profile-vars") > 0;

  int nonZeroBlock = 0;
  int sparsityRows, sparsityCols;
  std::vector<unsigned char> sparsity;
  std::size_t sparsityDenseSize = 0;
  if (hasSparsity) {
    bool readTransposed =
        ((scenario & Scenario::fwdMask) == Scenario::dsd) && rhsNeedTranspose0;
    if (!ReadMatrixMask(sparsityFileName, nonZeroBlock, sparsityRows,
                        sparsityCols, sparsity, readTransposed)) {
      return -1;
    }

    if (numGroups == 1) {
      std::cout << "non zero block: " << nonZeroBlock << std::endl;
      std::cout << "sparsity: "
                << (1.0 - nonZeroBlock /
                              static_cast<float>(sparsityRows * sparsityCols))
                << std::endl;
    } else {
      if (sparsityRows % numGroups != 0) {
        std::cerr << "error: Number of rows " << sparsityRows
                  << " in concatenated group sparsity mask "
                     "is not divisible by the number of groups "
                  << numGroups << std::endl;
        return -1;
      }
      std::cout << "Average non zero block: "
                << static_cast<float>(nonZeroBlock) / numGroups << std::endl;
      std::cout << "Average sparsity: "
                << (1.0 - nonZeroBlock /
                              static_cast<float>(sparsityRows * sparsityCols))
                << std::endl;
      sparsityRows /= numGroups;
    }
    sparsityDenseSize = sparsityRows * sparsityCols;
  }

  int lhsBlockRows0 = 0;
  int rhsBlockCols0 = 0;
  bool isRhsMatrixSparse0 = false;
  bool isResMatrixSparse0 = false;

  int resBlockRows0 = 0;
  int resBlockCols0 = 0;
  int resBlockRowSize0 = 0;
  int resBlockColSize0 = 0;
  int resRows0;
  int resCols0;

  if (computeMatmul) {
    if (hasSparsity) {
      isRhsMatrixSparse0 = (operandsType[0] == OperandsType::dsd);
      isResMatrixSparse0 = (operandsType[0] == OperandsType::dds);

      if (isRhsMatrixSparse0) {
        if (lhsRows0 % lhsBlockRowSize0 != 0) {
          std::cerr << "error: Number of LHS rows " << lhsRows0
                    << " is not divisible "
                       "by the number of rows in a block for the LHS matrix "
                    << lhsBlockRowSize0 << std::endl;
          return -1;
        }
        lhsBlockRows0 = lhsRows0 / lhsBlockRowSize0;
        lhsBlockCols0 = sparsityRows;
      } else if (isResMatrixSparse0) {
        lhsBlockRows0 = sparsityRows;
        lhsRows0 = lhsBlockRows0 * lhsBlockRowSize0;
        rhsNeedTranspose0 = false;
      } else {
        assert(0);
      }

      rhsBlockCols0 = sparsityCols;
      lhsCols0 = lhsBlockCols0 * lhsBlockColSize0;

      resBlockRows0 = lhsBlockRows0;
      resBlockCols0 = rhsBlockCols0;
      resBlockRowSize0 = lhsBlockRowSize0;
      resBlockColSize0 = rhsBlockSize0;
      resRows0 = resBlockRows0 * resBlockRowSize0;
      resCols0 = resBlockCols0 * resBlockColSize0;
    } else {
      resRows0 = lhsRows0;
      resCols0 = rhsCols0;
    }
  } else if (computeSoftmax) {
    if (hasSparsity) {
      resBlockRows0 = sparsityRows;
      resBlockCols0 = sparsityCols;
      resBlockRowSize0 = lhsBlockRowSize0;
      resBlockColSize0 = lhsBlockColSize0;
      resRows0 = resBlockRows0 * resBlockRowSize0;
      resCols0 = resBlockCols0 * resBlockColSize0;
      isResMatrixSparse0 = true;
    } else {
      resRows0 = lhsRows0;
      resCols0 = lhsCols0;
    }
  } else {
    assert(0);
  }

  const std::string debugPrefix = "bs_matmul";
  Sequence matSeq, uploadProg, downloadProg;

  std::vector<std::unique_ptr<char[]>> lhsRawHost(numSteps);
  std::vector<std::unique_ptr<char[]>> rhsRawHost(numSteps);
  std::vector<std::unique_ptr<char[]>> inputRawHost(numSteps);
  std::unique_ptr<char[]> outputRawHost;
  std::vector<std::pair<std::string, char *>> tmap;

  const auto &target = device.getTarget();
  Graph graph(target);

  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  double flops = 0.0;

  boost::multi_array<float, 2> lhsHost;
  boost::multi_array<float, 2> rhsHost;
  boost::multi_array<float, 2> inputHost;

  boost::multi_array<float, 3> lhsHost3D;
  boost::multi_array<float, 3> rhsHost3D;
  boost::multi_array<float, 3> inputHost3D;

  poplar::Tensor aTensor;
  poplar::Tensor bTensor;
  poplar::Tensor dCTensor;
  poplar::Tensor dBTensor;

  poplar::OptionFlags options = {
      {"memoryCycleRatio", std::to_string(memoryCycleRatio)},
      {"numberOfPass", std::to_string(nPass)},
      {"partitionMethod", partitionMethod},
  };

  for (int nRep = 0; nRep < numReps; ++nRep) {
    for (unsigned nStep = 0; nStep < numSteps; ++nStep) {
      std::string stepSuffix =
          numSteps == 1 ? "" : "(" + std::to_string(nStep) + ")";
      if (computeMatmul) {
        int lhsBlockCols = lhsBlockCols0;
        int rhsBlockCols = rhsBlockCols0;
        int lhsRows = lhsRows0;
        int lhsCols = lhsCols0;
        int rhsCols = rhsCols0;
        int lhsBlockRowSize = lhsBlockRowSize0;
        int lhsBlockColSize = lhsBlockColSize0;
        int rhsBlockSize = rhsBlockSize0;
        bool rhsNeedTranspose = rhsNeedTranspose0;

        if ((scenario & Scenario::fwdMask) == Scenario::dsd) {
          if (runType[nStep] == Pass::BWD) {
            lhsBlockCols = rhsBlockCols0;
            rhsBlockCols = lhsBlockCols0;
            lhsRows = lhsRows0;
            lhsBlockColSize = rhsBlockSize0;
            rhsBlockSize = lhsBlockColSize0;
            rhsNeedTranspose = !rhsNeedTranspose;
          } else if (runType[nStep] == Pass::WU) {
            lhsBlockCols = lhsBlockRows0;
            rhsBlockCols = rhsBlockCols0;
            lhsRows = lhsCols0;
            lhsBlockRowSize = lhsBlockColSize0;
            lhsBlockColSize = lhsBlockRowSize0;
            rhsBlockSize = rhsBlockSize0;
            rhsNeedTranspose = false;
          }
        } else if ((scenario & Scenario::fwdMask) == Scenario::dds) {
          if (runType[nStep] == Pass::BWD) {
            lhsBlockCols = rhsBlockCols0;
            rhsBlockCols = lhsBlockRows0;
            lhsRows = lhsCols0;
            lhsBlockColSize = rhsBlockSize0;
            lhsBlockRowSize = lhsBlockColSize0;
            rhsBlockSize = lhsBlockRowSize0;
          } else if (runType[nStep] == Pass::WU) {
            lhsBlockCols = lhsBlockRows0;
            rhsBlockCols = rhsBlockCols0;
            lhsRows = lhsCols0;
            lhsBlockColSize = lhsBlockRowSize0;
            lhsBlockRowSize = lhsBlockColSize0;
            rhsBlockSize = rhsBlockSize0;
            rhsNeedTranspose = true;
          }
        }
        if (hasSparsity) {
          lhsCols = lhsBlockCols * lhsBlockColSize;
          rhsCols = rhsBlockCols * rhsBlockSize;
        }
        int lhsBlockRows = lhsRows / lhsBlockRowSize;

        int resBlockRows = lhsBlockRows;
        int resBlockCols = rhsBlockCols;
        int resBlockRowSize = lhsBlockRowSize;
        int resBlockColSize = rhsBlockSize;
        int resRows = resBlockRows * resBlockRowSize;
        int resCols = resBlockCols * resBlockColSize;

        bool isResMatrixSparse = (operandsType[nStep] == OperandsType::dds ||
                                  operandsType[nStep] == OperandsType::s);
        bool isRhsMatrixSparse = (operandsType[nStep] == OperandsType::dsd);

        std::array<int, 3> dim;
        dim[0] = lhsRows;
        dim[1] = lhsCols;
        dim[2] = rhsCols;

        std::array<int, 3> blockSize = {lhsBlockRowSize, lhsBlockColSize,
                                        rhsBlockSize};

        std::cout << "Step " << nStep << std::endl;
        std::cout << "dim: " << dim[0] << " x " << dim[1] << " x " << dim[2]
                  << std::endl;
        if (operandsType[nStep] != OperandsType::ddd) {
          std::cout << "block size: " << blockSize[0] << " x " << blockSize[1]
                    << " x " << blockSize[2] << std::endl;
        }
        if (rhsNeedTranspose) {
          logging::popsparse::info("Right matrix must be transposed");
        }
        logging::popsparse::info("LHS rows: {} ", lhsRows);

        float flopsOp;
        if (operandsType[nStep] == OperandsType::ddd) {
          flopsOp = 2.0 * dim[0] * dim[1] * dim[2];
        } else {
          if (!isResMatrixSparse) {
            flopsOp = 2.0 * nonZeroBlock * dim[0] * blockSize[1] * blockSize[2];
          } else {
            flopsOp = 2.0 * nonZeroBlock * dim[1] * blockSize[0] * blockSize[2];
          }
        }
        logging::popsparse::info("flops op: {:.1f} ", flopsOp);
        flops += flopsOp;

        if (numGroups == 1) {
          lhsHost.resize(boost::extents[dim[0]][dim[1]]);
          if (!rhsNeedTranspose) {
            rhsHost.resize(boost::extents[dim[1]][dim[2]]);
          } else {
            rhsHost.resize(boost::extents[dim[2]][dim[1]]);
          }
          if (operandsType[nStep] == OperandsType::ddd) {
            populateMatrixData(dim, lhsHost, rhsHost);
          } else {
            populateMatrixData(dim, blockSize, lhsHost, rhsHost,
                               (!isRhsMatrixSparse ? nullptr : sparsity.data()),
                               rhsNeedTranspose);
          }
        } else {
          lhsHost3D.resize(boost::extents[numGroups][dim[0]][dim[1]]);
          if (!rhsNeedTranspose) {
            rhsHost3D.resize(boost::extents[numGroups][dim[1]][dim[2]]);
          } else {
            rhsHost3D.resize(boost::extents[numGroups][dim[2]][dim[1]]);
          }
          populateMatrixData(dim, blockSize, numGroups, lhsHost3D, rhsHost3D,
                             (!isRhsMatrixSparse ? nullptr : sparsity.data()),
                             rhsNeedTranspose);
        }

        if (operandsType[nStep] == OperandsType::ddd) {
          poplin::matmul::PlanningCache cache;
          OptionFlags mmOpt;

          // LHS matrix
          poplar::Tensor lhsTensor = graph.addVariable(
              dataType, {(unsigned long)dim[0], (unsigned long)dim[1]},
              debugPrefix + "/matrix_lhs");
          poputil::mapTensorLinearly(graph, lhsTensor);

          if (numReps == 1) { // Copying data from host is not supported for
                              // multi-rep tests yet
            lhsRawHost[nStep] = allocateHostMemoryForTensor(
                lhsTensor, debugPrefix + "/matrix_lhs" + stepSuffix, graph,
                uploadProg, downloadProg, tmap);
            assert(lhsHost.num_elements() == lhsTensor.numElements());
            copy(target, lhsHost, dataType, lhsRawHost[nStep].get());
          }

          // RHS matrix
          poplar::Tensor rhsTensor = graph.addVariable(
              dataType, {(unsigned long)dim[1], (unsigned long)dim[2]},
              debugPrefix + "/matrix_rhs");
          poputil::mapTensorLinearly(graph, rhsTensor);

          if (numReps == 1) {
            rhsRawHost[nStep] = allocateHostMemoryForTensor(
                rhsTensor, debugPrefix + "/matrix_rhs" + stepSuffix, graph,
                uploadProg, downloadProg, tmap);
            assert(rhsHost.num_elements() == rhsTensor.numElements());
            copy(target, rhsHost, dataType, rhsRawHost[nStep].get());
          }

          // output matrix
          poplar::Tensor outTensor =
              matMul(graph, lhsTensor, rhsTensor, matSeq,
                     debugPrefix + "/lhsxrhs", mmOpt, &cache);
          if (checkResult == 1) {
            outputRawHost = poplibs_test::util::allocateHostMemoryForTensor(
                outTensor, debugPrefix + "/matrix_output" + stepSuffix, graph,
                uploadProg, downloadProg, tmap);
          }
        } else {
          BSMatMulParams bsMatMulObj = createBsMatMul(
              dim, blockSize, sparsity, dataType, isResMatrixSparse,
              rhsNeedTranspose, mask, numGroups);

          // LHS dense matrix
          poplar::Tensor lhsTensor;
          poplar::Tensor rhsTensor;
          if (runType[nStep] == Pass::FWD) {
            lhsTensor = createBSMatMulInputLHS(
                graph, bsMatMulObj, debugPrefix + "/matrix_lhs", options);
            rhsTensor = createBSMatMulInputRHS(
                graph, bsMatMulObj, debugPrefix + "/matrix_rhs", options);
            if (((scenario & Scenario::bwd) == Scenario::bwd) ||
                ((scenario & Scenario::wu) == Scenario::wu)) {
              // Save A tensor
              aTensor = lhsTensor;
              // Save B tensor
              bTensor = rhsTensor;
            }
          } else if (runType[nStep] == Pass::BWD) {
            if ((scenario & Scenario::fwdMask) == Scenario::dsd) {
              lhsTensor = createBSMatMulInputLHS(
                  graph, bsMatMulObj, debugPrefix + "/matrix_lhs", options);
              rhsTensor = bTensor;
              if ((scenario & Scenario::wu) == Scenario::wu) {
                // Save dC tensor
                dCTensor = lhsTensor;
              }
            } else if ((scenario & Scenario::fwdMask) == Scenario::dds) {
              lhsTensor = bTensor;
              rhsTensor = createBSMatMulInputRHS(
                  graph, bsMatMulObj, debugPrefix + "/matrix_rhs", options);
              if ((scenario & Scenario::wu) == Scenario::wu) {
                // Save dB tensor
                dBTensor = rhsTensor;
              }
            } else {
              assert(0);
            }
          } else if (runType[nStep] == Pass::WU) {
            if ((scenario & Scenario::fwdMask) == Scenario::dsd) {
              assert(aTensor.valid());
              lhsTensor = aTensor.transpose();
              if ((scenario & Scenario::bwd) == Scenario::bwd) {
                // If tensor was saved
                assert(dCTensor.valid());
                rhsTensor = dCTensor;
              } else {
                rhsTensor = createBSMatMulInputRHS(
                    graph, bsMatMulObj, debugPrefix + "/matrix_rhs", options);
              }
            } else if ((scenario & Scenario::fwdMask) == Scenario::dds) {
              assert(aTensor.valid());
              lhsTensor = aTensor.transpose();
              if ((scenario & Scenario::bwd) == Scenario::bwd) {
                // If tensor was saved
                assert(dBTensor.valid());
                rhsTensor = dBTensor;
              } else {
                rhsTensor = createBSMatMulInputRHS(
                    graph, bsMatMulObj, debugPrefix + "/matrix_rhs", options);
              }
            } else {
              assert(0);
            }
          } else {
            assert(0);
          }

          // sparse matmul
          poplar::Tensor outTensor;
          try {
            outTensor =
                bsMatMul(graph, bsMatMulObj, matSeq, lhsTensor, rhsTensor,
                         options, debugPrefix + "/bs-matmul");
          } catch (const poputil::poplibs_error &e) {
            std::cerr << "bsMatMul() failed with the message: " << e.what()
                      << std::endl;
            return -1;
          }
          if (isResMatrixSparse) {
            assert((int)outTensor.dim(0) == nonZeroBlock);
          }
          if ((scenario & Scenario::fwdMask) == Scenario::dds) {
            if (runType[nStep] == Pass::BWD) {
              outTensor = outTensor.transpose();
            }
          }
          if (runType[nStep] == Pass::BWD) {
            assert(outTensor.shape() == aTensor.shape());
          } else if (runType[nStep] == Pass::WU) {
            assert(outTensor.shape() == bTensor.shape());
          }
          if (runType[nStep] == Pass::FWD && computeSoftmax) {
            std::array<int, 2> dimSm;
            std::array<int, 2> blockSizeSm;
            dimSm[0] = resRows;
            dimSm[1] = resCols;
            blockSizeSm[0] = resBlockRowSize;
            blockSizeSm[1] = resBlockColSize;

            Tensor softmaxTensor =
                bsSoftmax(graph, outTensor, dimSm, blockSizeSm, sparsity, mask,
                          matSeq, debugPrefix + "/bs-softmax");
            outTensor = softmaxTensor;
            if (((scenario & Scenario::bwd) == Scenario::bwd) ||
                ((scenario & Scenario::wu) == Scenario::wu)) {
              // Create helper BSMatMulParams in order to generate a matrix in
              // block-sparse format w/out extra matmul overhead
              std::array<int, 3> dimHelper;
              std::array<int, 3> blockSizeHelper;
              dimHelper[0] = blockSize[0]; // Can be anything
              dimHelper[1] = resRows;
              dimHelper[2] = resCols;
              blockSizeHelper[0] = blockSize[0]; // Can be anything
              blockSizeHelper[1] = resBlockRowSize;
              blockSizeHelper[2] = resBlockColSize;

              BSMatMulParams bsMatMulObjHelper =
                  createBsMatMul(dimHelper, blockSizeHelper, sparsity, dataType,
                                 false, false, mask, numGroups);
              Tensor outerGrad =
                  createBSMatMulInputRHS(graph, bsMatMulObjHelper,
                                         debugPrefix + "/outer_grad", options);
              assert(outerGrad.shape() == softmaxTensor.shape());
              poputil::mapTensorLinearly(graph, outerGrad);
              Tensor softmaxGradTensor = bsSoftmaxGrad(
                  graph, softmaxTensor, outerGrad, dimSm, blockSizeSm, sparsity,
                  matSeq, debugPrefix + "/bs-softmax-grad");
            }
          }

          if (numReps == 1) {
            lhsRawHost[nStep] = allocateHostMemoryForTensor(
                lhsTensor, debugPrefix + "/matrix_lhs" + stepSuffix, graph,
                uploadProg, downloadProg, tmap);
            if (numGroups == 1) {
              assert(lhsHost.num_elements() == lhsTensor.numElements());
              copy(target, lhsHost, dataType, lhsRawHost[nStep].get());
            } else {
              assert(lhsHost3D.num_elements() == lhsTensor.numElements());
              copy(target, lhsHost3D, dataType, lhsRawHost[nStep].get());
            }
          }

          if (isRhsMatrixSparse) {
            // RHS sparse matrix
            boost::multi_array<float, 1> rhsBlocksHost;
            if (numGroups == 1) {
              if (!rhsNeedTranspose) {
                getSparseMatrixBlocks(dim[1], dim[2], blockSize[1],
                                      blockSize[2], sparsity.data(),
                                      nonZeroBlock, rhsHost, rhsBlocksHost);
              } else {
                getSparseMatrixBlocks(dim[2], dim[1], blockSize[2],
                                      blockSize[1], sparsity.data(),
                                      nonZeroBlock, rhsHost, rhsBlocksHost);
              }
            } else {
              if (!rhsNeedTranspose) {
                getSparseMatrixBlocks(dim[1], dim[2], blockSize[1],
                                      blockSize[2], numGroups, sparsity.data(),
                                      nonZeroBlock, rhsHost3D, rhsBlocksHost);
              } else {
                getSparseMatrixBlocks(dim[2], dim[1], blockSize[2],
                                      blockSize[1], numGroups, sparsity.data(),
                                      nonZeroBlock, rhsHost3D, rhsBlocksHost);
              }
            }
            if (numReps == 1) {
              rhsRawHost[nStep] = allocateHostMemoryForTensor(
                  rhsTensor, debugPrefix + "/matrix_rhs" + stepSuffix, graph,
                  uploadProg, downloadProg, tmap);
              assert(rhsBlocksHost.num_elements() == rhsTensor.numElements());
              copy(target, rhsBlocksHost, dataType, rhsRawHost[nStep].get());
            }
          } else {
            // RHS dense matrix
            if (numReps == 1) {
              rhsRawHost[nStep] = allocateHostMemoryForTensor(
                  rhsTensor, debugPrefix + "/matrix_rhs" + stepSuffix, graph,
                  uploadProg, downloadProg, tmap);
              if (numGroups == 1) {
                assert(rhsHost.num_elements() == rhsTensor.numElements());
                copy(target, rhsHost, dataType, rhsRawHost[nStep].get());
              } else {
                assert(rhsHost3D.num_elements() == rhsTensor.numElements());
                copy(target, rhsHost3D, dataType, rhsRawHost[nStep].get());
              }
            }
          }
          if (checkResult == 1) {
            outputRawHost = poplibs_test::util::allocateHostMemoryForTensor(
                outTensor, debugPrefix + "/matrix_output" + stepSuffix, graph,
                uploadProg, downloadProg, tmap);
          }
        }
      } else if (computeSoftmax) {
        std::array<int, 2> dim = {resRows0, resCols0};

        std::cout << "dim: " << dim[0] << " x " << dim[1] << std::endl;

        if (hasSparsity) {
          std::array<int, 2> blockSize = {resBlockRowSize0, resBlockColSize0};

          // Create helper BSMatMulParams in order to generate a matrix in
          // block-sparse format w/out extra matmul overhead
          std::array<int, 3> dimHelper;
          std::array<int, 3> blockSizeHelper;
          dimHelper[0] = blockSize[0]; // Can be anything
          dimHelper[1] = resRows0;
          dimHelper[2] = resCols0;
          blockSizeHelper[0] = blockSize[0]; // Can be anything
          blockSizeHelper[1] = resBlockRowSize0;
          blockSizeHelper[2] = resBlockColSize0;

          BSMatMulParams bsMatMulObjHelper =
              createBsMatMul(dimHelper, blockSizeHelper, sparsity, dataType,
                             false, false, mask, numGroups);

          Tensor inputTensor = createBSMatMulInputRHS(
              graph, bsMatMulObjHelper, debugPrefix + "/input", options);
          poputil::mapTensorLinearly(graph, inputTensor);

          Tensor outerGradTensor;
          if (scenario == Scenario::smsBwd) {
            outerGradTensor = createBSMatMulInputRHS(
                graph, bsMatMulObjHelper, debugPrefix + "/outer_grad", options);
            assert(outerGradTensor.shape() == inputTensor.shape());
            poputil::mapTensorLinearly(graph, outerGradTensor);
          }

          poplar::Tensor outTensor;
          if (scenario == Scenario::sms) {
            outTensor = bsSoftmax(graph, inputTensor, dim, blockSize, sparsity,
                                  mask, matSeq, debugPrefix + "/bs-softmax");
          } else if (scenario == Scenario::smsBwd) {
            outTensor = bsSoftmaxGrad(graph, inputTensor, outerGradTensor, dim,
                                      blockSize, sparsity, matSeq,
                                      debugPrefix + "/bs-softmax-grad");
          } else {
            assert(0);
          }

          inputHost.resize(boost::extents[dim[0]][dim[1]]);
          populateMatrixData(dim, blockSize, inputHost, sparsity.data());

          boost::multi_array<float, 1> inputBlocksHost;
          getSparseMatrixBlocks(dim[0], dim[1], blockSize[0], blockSize[1],
                                sparsity.data(), nonZeroBlock, inputHost,
                                inputBlocksHost);
          if (numReps == 1) {
            inputRawHost[nStep] = allocateHostMemoryForTensor(
                inputTensor, debugPrefix + "/input" + stepSuffix, graph,
                uploadProg, downloadProg, tmap);
            assert(inputBlocksHost.num_elements() == inputTensor.numElements());
            copy(target, inputBlocksHost, dataType, inputRawHost[nStep].get());
          }
          if (checkResult == 1) {
            outputRawHost = poplibs_test::util::allocateHostMemoryForTensor(
                outTensor, debugPrefix + "/output" + stepSuffix, graph,
                uploadProg, downloadProg, tmap);
          }
        } else {
          Tensor inputTensor = graph.addVariable(
              dataType, {(unsigned long)dim[0], (unsigned long)dim[1]},
              debugPrefix + "/input");
          poputil::mapTensorLinearly(graph, inputTensor);

          Tensor outerGradTensor;
          if (scenario == Scenario::smdBwd) {
            outerGradTensor = graph.addVariable(
                dataType, {(unsigned long)dim[0], (unsigned long)dim[1]},
                debugPrefix + "/outer_grad");
            assert(outerGradTensor.shape() == inputTensor.shape());
            poputil::mapTensorLinearly(graph, outerGradTensor);
          }

          poplar::Tensor outTensor;
          if (scenario == Scenario::smd) {
            float nonLinearityScaling;
            outTensor = popnn::nonLinearity(
                graph, popnn::NonLinearityType::SOFTMAX_STABLE, inputTensor,
                nonLinearityScaling, matSeq, debugPrefix + "/softmax");
          } else if (scenario == Scenario::smdBwd) {
            outTensor = popnn::nonLinearityInputGradient(
                graph, popnn::NonLinearityType::SOFTMAX_STABLE, inputTensor,
                outerGradTensor, matSeq, debugPrefix + "/softmax-grad");
          } else {
            assert(0);
          }

          inputHost.resize(boost::extents[dim[0]][dim[1]]);
          populateMatrixData(dim, inputHost);

          if (numReps == 1) {
            inputRawHost[nStep] = allocateHostMemoryForTensor(
                inputTensor, debugPrefix + "/input" + stepSuffix, graph,
                uploadProg, downloadProg, tmap);
            assert(inputHost.num_elements() == inputTensor.numElements());
            copy(target, inputHost, dataType, inputRawHost[nStep].get());
          }
          if (checkResult == 1) {
            outputRawHost = poplibs_test::util::allocateHostMemoryForTensor(
                outTensor, debugPrefix + "/output" + stepSuffix, graph,
                uploadProg, downloadProg, tmap);
          }
        }
      } else {
        assert(0);
      }
    }
  }

  auto engineOptions = defaultEngineOptions;
  if (doProfiling) {
    engineOptions.set("debug.instrumentCompute", "true");
    if (doProfilingExecution) {
      engineOptions.set("debug.instrumentExternalExchange", "true");
    }
    if (doProfilingVars) {
      engineOptions.set("debug.loweredVarDumpFile", profileDir + "/vars.capnp");
    }
  }
  engineOptions.set("exchange.multicastPolicy", "balanced");

  Sequence allSequence;
  // if run many times, we are doing benchmark, so ignore the host data copy
  if (checkResult == 1) {
    allSequence.add(uploadProg);
    allSequence.add(matSeq);
    allSequence.add(downloadProg);
  } else {
    allSequence.add(uploadProg);
    allSequence.add(Repeat(runs, matSeq));
    allSequence.add(downloadProg);
  }

  Engine engine(graph, allSequence, engineOptions);

  if (vm.count("compile-only"))
    std::exit(0);

  std::cout << "Start " << runs << " runs of " << numReps << " rep(s)..."
            << std::endl;
  poplibs_test::util::attachStreams(engine, tmap);
  device.bind([&](const Device &d) {
    engine.load(d);
    auto start = std::chrono::system_clock::now();
    engine.run(0);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedSeconds = end - start;
    std::cout << "elapsed time: " << std::fixed << std::setprecision(6)
              << elapsedSeconds.count() << std::endl;

    double avgTime = elapsedSeconds.count() / runs;
    std::cout << "average kernel run time: " << std::fixed
              << std::setprecision(1) << avgTime * 1000000 << " mcs"
              << std::endl;
    std::cout << "flop: " << std::fixed << std::setprecision(1) << flops
              << std::endl;
    std::cout << "GFLOPS: " << std::fixed << std::setprecision(6)
              << flops / 1000000000.0 / avgTime << std::endl;
  });

  if (deviceType != DeviceType::Cpu && doProfiling) {
    engine.printProfileSummary(
        std::cout,
        OptionFlags{
            {"showExecutionSteps", doProfilingExecution ? "true" : "false"},
            {"showVarStorage", doProfilingVars ? "true" : "false"}});
    if (vm.count("profile-dir")) {
      savePoplarReport(engine, profileDir);
    }
  }

  if (!checkResult) {
    return 0;
  }

  std::cout << "Checking the result..." << std::endl;
  boost::multi_array<float, 4> denseMatC(
      boost::extents[1][1][resRows0][resCols0]);
  boost::multi_array<float, 5> denseMatC3D(
      boost::extents[numGroups][1][1][resRows0][resCols0]);

  std::fill_n(denseMatC.data(), denseMatC.num_elements(), 0.0f);
  if (numGroups == 1) {
    if (!isResMatrixSparse0) {
      copy(target, dataType, outputRawHost.get(), denseMatC);
    } else {
      denseMatC.reshape(std::vector<int>(
          {resBlockRows0, resBlockCols0, resBlockRowSize0, resBlockColSize0}));
      boost::multi_array<float, 3> denseMatCTmp(
          boost::extents[nonZeroBlock][resBlockRowSize0][resBlockColSize0]);
      copy(target, dataType, outputRawHost.get(), denseMatCTmp);
      int blockCount = 0;
      for (int br = 0; br < resBlockRows0; ++br) {
        for (int bc = 0; bc < resBlockCols0; ++bc) {
          if (sparsity[br * resBlockCols0 + bc] > 0) {
            denseMatC[br][bc] = denseMatCTmp[blockCount++];
          }
        }
      }
    }
  } else {
    if (!isResMatrixSparse0) {
      copy(target, dataType, outputRawHost.get(), denseMatC3D);
    } else {
      denseMatC3D.reshape(std::vector<int>(
          {static_cast<int>(numGroups), resBlockRows0, resBlockCols0,
           resBlockRowSize0, resBlockColSize0}));
      const unsigned char *itemSparsity = sparsity.data();
      boost::multi_array<float, 3> denseMatCTmp(
          boost::extents[nonZeroBlock][resBlockRowSize0][resBlockColSize0]);
      copy(target, dataType, outputRawHost.get(), denseMatCTmp);
      int blockCount = 0;
      for (int g = 0; g < numGroups; ++g) {
        for (int br = 0; br < resBlockRows0; ++br) {
          for (int bc = 0; bc < resBlockCols0; ++bc) {
            if (itemSparsity[br * resBlockCols0 + bc] > 0) {
              denseMatC3D[g][br][bc] = denseMatCTmp[blockCount++];
            }
          }
        }
        itemSparsity += sparsityDenseSize;
      }
    }
  }

  std::vector<std::vector<float>> hostMatC(resRows0,
                                           std::vector<float>(resCols0, 0.0f));

  std::vector<std::vector<std::vector<float>>> hostMatC3D(
      numGroups, std::vector<std::vector<float>>(
                     resRows0, std::vector<float>(resCols0, 0.0f)));

  if (computeMatmul) {
    if (!isResMatrixSparse0) {
      for (int r = 0; r < resRows0; r++) {
        for (int c = 0; c < resCols0; c++) {
          if (numGroups == 1) {
            float sum = 0.0f;
            for (int cmul = 0; cmul < lhsCols0; cmul++) {
              sum +=
                  lhsHost[r][cmul] *
                  (rhsNeedTranspose0 ? (rhsHost[c][cmul]) : (rhsHost[cmul][c]));
            }
            hostMatC[r][c] = sum;
          } else {
            for (int g = 0; g < numGroups; ++g) {
              float sum = 0.0f;
              for (int cmul = 0; cmul < lhsCols0; cmul++) {
                sum += lhsHost3D[g][r][cmul] * (rhsNeedTranspose0
                                                    ? (rhsHost3D[g][c][cmul])
                                                    : (rhsHost3D[g][cmul][c]));
              }
              hostMatC3D[g][r][c] = sum;
            }
          }
        }
      }
    } else {
      for (int r = 0; r < resRows0; r++) {
        for (int c = 0; c < resCols0; c++) {
          if (numGroups == 1) {
            float sum = 0.0f;
            if (!((mask == SubBlockMask::ZeroUpperTriangle && r < c) ||
                  (mask == SubBlockMask::ZeroLowerTriangle && r > c))) {
              int br = r / lhsBlockRowSize0;
              int bc = c / rhsBlockSize0;
              if (sparsity[br * rhsBlockCols0 + bc] != 0) {
                for (int cmul = 0; cmul < lhsCols0; cmul++) {
                  sum += lhsHost[r][cmul] * (rhsNeedTranspose0
                                                 ? (rhsHost[c][cmul])
                                                 : (rhsHost[cmul][c]));
                }
              }
            }
            hostMatC[r][c] = sum;
          } else {
            const unsigned char *itemSparsity = sparsity.data();
            for (int g = 0; g < numGroups; ++g) {
              float sum = 0.0f;
              if (!((mask == SubBlockMask::ZeroUpperTriangle && r < c) ||
                    (mask == SubBlockMask::ZeroLowerTriangle && r > c))) {
                int br = r / lhsBlockRowSize0;
                int bc = c / rhsBlockSize0;
                if (itemSparsity[br * rhsBlockCols0 + bc] != 0) {
                  for (int cmul = 0; cmul < lhsCols0; cmul++) {
                    sum += lhsHost3D[g][r][cmul] *
                           (rhsNeedTranspose0 ? (rhsHost3D[g][c][cmul])
                                              : (rhsHost3D[g][cmul][c]));
                  }
                }
              }
              hostMatC3D[g][r][c] = sum;
              itemSparsity += sparsityDenseSize;
            }
          }
        }
      }
    }
  } else {
    for (int r = 0; r < resRows0; r++) {
      for (int c = 0; c < resCols0; c++) {
        hostMatC[r][c] = inputHost[r][c];
      }
    }
  }

  if (computeSoftmax) {
    if (hasSparsity) {
      std::vector<float> maxs(resRows0, -3.4028235e+38);
      for (int r = 0; r < resRows0; r++) {
        for (int c = 0; c < resCols0; c++) {
          if (!((mask == SubBlockMask::ZeroUpperTriangle && r < c) ||
                (mask == SubBlockMask::ZeroLowerTriangle && r > c))) {
            int br = r / resBlockRowSize0;
            int bc = c / resBlockColSize0;
            if (sparsity[br * resBlockCols0 + bc] != 0) {
              maxs[r] = std::max(maxs[r], hostMatC[r][c]);
            }
          }
        }
      }
      for (int r = 0; r < resRows0; r++) {
        std::vector<double> expVals(resCols0, 0.0f);
        double expSum = 0.0f;
        for (int c = 0; c < resCols0; c++) {
          if (!((mask == SubBlockMask::ZeroUpperTriangle && r < c) ||
                (mask == SubBlockMask::ZeroLowerTriangle && r > c))) {
            int br = r / resBlockRowSize0;
            int bc = c / resBlockColSize0;
            if (sparsity[br * resBlockCols0 + bc] != 0) {
              double expV =
                  std::exp(static_cast<double>(hostMatC[r][c] - maxs[r]));
              expVals[c] = expV;
              expSum += expV;
            }
          }
        }
        for (int c = 0; c < resCols0; c++) {
          hostMatC[r][c] = 0.0f;
          if (!((mask == SubBlockMask::ZeroUpperTriangle && r < c) ||
                (mask == SubBlockMask::ZeroLowerTriangle && r > c))) {
            int br = r / resBlockRowSize0;
            int bc = c / resBlockColSize0;
            if (sparsity[br * resBlockCols0 + bc] != 0) {
              hostMatC[r][c] = expVals[c] / expSum;
            }
          }
        }
      }
    } else {
      std::vector<float> maxs(resRows0, -3.4028235e+38);
      for (int r = 0; r < resRows0; r++) {
        for (int c = 0; c < resCols0; c++) {
          maxs[r] = std::max(maxs[r], hostMatC[r][c]);
        }
      }
      for (int r = 0; r < resRows0; r++) {
        std::vector<double> expVals(resCols0, 0.0f);
        double expSum = 0.0f;
        for (int c = 0; c < resCols0; c++) {
          double expV = std::exp(static_cast<double>(hostMatC[r][c] - maxs[r]));
          expVals[c] = expV;
          expSum += expV;
        }
        for (int c = 0; c < resCols0; c++) {
          hostMatC[r][c] = expVals[c] / expSum;
        }
      }
    }
  }

  const float threshold = dataType == poplar::FLOAT ? 1.0e-4 : 1.0e-2;
  float maxErr = 0.0f;
  int errRow = 0;
  int errCol = 0;
  float valTruth = 0.0f;
  float valTest = 0.0f;
  bool success = true;
  bool foundNan = false;
  for (int g = 0; g < numGroups; ++g && !foundNan) {
    for (int r = 0; r < resRows0; r++ && !foundNan) {
      for (int c = 0; c < resCols0; c++ && !foundNan) {
        float err;
        float curValTruth, curValTest;
        if (numGroups == 1) {
          curValTruth = hostMatC[r][c];
        } else {
          curValTruth = hostMatC3D[g][r][c];
        }
        if (!isResMatrixSparse0) {
          if (numGroups == 1) {
            curValTest = denseMatC[0][0][r][c];
          } else {
            curValTest = denseMatC3D[g][0][0][r][c];
          }
          err = fabs(curValTest - curValTruth);
        } else {
          int br = r / resBlockRowSize0;
          int bc = c / resBlockColSize0;
          int rb = r % resBlockRowSize0;
          int cb = c % resBlockColSize0;
          if (numGroups == 1) {
            curValTest = denseMatC[br][bc][rb][cb];
          } else {
            curValTest = denseMatC3D[g][br][bc][rb][cb];
          }
          err = fabs(curValTest - curValTruth);
        }
        if (std::isnan(curValTest)) {
          foundNan = true;
          success = false;
          logging::popsparse::debug("[{}][{}] True: {}, test = {}", r, c,
                                    curValTruth, curValTest);
        } else if (!checkIsClose(curValTruth, curValTest, threshold)) {
          success = false;
          logging::popsparse::debug("[{}][{}] True: {}, test = {}", r, c,
                                    curValTruth, curValTest);
          if (err > maxErr) {
            valTruth = curValTruth;
            valTest = curValTest;
            errRow = r;
            errCol = c;
            maxErr = err;
          }
        }
      }
    }
  }

  if (success) {
    std::cout << "Result check succeeded." << std::endl;
  } else {
    if (foundNan) {
      std::cout << "Result check failed." << std::endl
                << "Found NaN in the output" << std::endl;
    } else {
      std::cout << "Result check failed." << std::endl
                << " Maximum error = " << std::fixed << std::setprecision(5)
                << maxErr << std::endl
                << " test[" << errRow << "][" << errCol << "] = " << std::fixed
                << std::setprecision(5) << valTest << std::endl
                << " true[" << errRow << "][" << errCol << "] = " << std::fixed
                << std::setprecision(5) << valTruth << std::endl;
    }
  }

  return (success ? 0 : -1);
}
