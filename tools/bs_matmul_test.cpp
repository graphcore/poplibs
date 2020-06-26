// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <boost/program_options.hpp>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

#define DEVELOP_BLOCKSPARSE

#include "TestDevice.hpp"
#include "poplibs_support/logging.hpp"
#include "poplibs_test/Util.hpp"
#include <poplar/IPUModel.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <popsparse/experimental/BlockSparseMatMul.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include "poplibs_test/Util.hpp"

using namespace poplar;
using namespace poplin;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace popsparse::experimental;
namespace logging = poplibs_support::logging;

const OptionFlags defaultEngineOptions{{"debug.allowOutOfMemory", "true"}};

bool ReadMatrixMask(std::string &fileName, int &nonZeroBlock, int &row,
                    int &col, std::vector<unsigned char> &mask) {
  std::ifstream cin(fileName);
  if (!cin.is_open()) {
    std::cout << " Can not open matrix sparsity mask file " << fileName << "\n";
    return false;
  }

  cin >> row >> col;
  mask.resize(row * col);
  unsigned char *data = mask.data();

  if (!data)
    return false;

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      int c;
      cin >> c;
      if (c)
        data[i * col + j] = 1;
      else
        data[i * col + j] = 0;
    }
  }

  nonZeroBlock = 0;
  for (int i = 0; i < row * col; i++)
    if (data[i] == 1)
      nonZeroBlock++;
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

void printMatrix(std::string msg, float *matrix, int row, int col) {
#ifdef DEBUG
  std::cout << msg << "\n";

  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      std::cout << matrix[i * col + j] << " ";
    }
    std::cout << "\n";
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

int main(int argc, char **argv) {
  // Default options
  DeviceType deviceType = DeviceType::Hw;
  std::string sparsityFileName = "sparsity.txt";
  std::string profileDir = ".";
  std::string subBlockMask = "None";
  std::string partitionMethod = "strip";
  int lhsBlockRowSize = 36;
  int lhsBlockColSize = 8;
  int rhsBlockSize = 8;
  int batchSize = 72;
  int lhsBlockCols = 0;
  IPUModel ipuModel;
  Type dataType = FLOAT;
  float memoryCycleRatio = 1;
  int isRhsMatrixSparse = 1;
  int isResMatrixSparse = 0;
  unsigned runs = 1;
  int rhsNeedTranspose = 0;
  int checkResult = 1;
  int numGroups = 1;

  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()
      // help
      ("help", "Produce help message")
      // device-type
      ("device-type",
       po::value<DeviceType>(&deviceType)->default_value(deviceType),
       "Device type: Cpu | Sim | Sim2 | Hw | IpuModel | IpuModel2")
      // profile
      ("profile", "Output profiling report")
      // profile-execution
      ("profile-execution", "Output execution steps in the profiling report")
      // profile-vars
      ("profile-vars", "Output variables storage in the profiling report")
      // profile-dir
      ("profile-dir",
       po::value<std::string>(&profileDir)->default_value(profileDir),
       "The directory to output profiling report")
      // dense-matmul
      ("dense-matmul", "Use dense matrix multiply")
      // tiles-per-ipu
      ("tiles-per-ipu",
       po::value<unsigned>(&ipuModel.tilesPerIPU)
           ->default_value(ipuModel.tilesPerIPU),
       "Number of tiles per IPU")("ipus",
                                  po::value<unsigned>(&ipuModel.numIPUs)
                                      ->default_value(ipuModel.numIPUs),
                                  "Number of IPUs")
      // data-type
      ("data-type", po::value<Type>(&dataType)->default_value(dataType),
       "matmul data type")
      // batch
      ("batch", po::value<int>(&batchSize)->default_value(batchSize),
       "The batch size for the LHS matrix")
      // lhs-block-cols
      ("lhs-block-cols",
       po::value<int>(&lhsBlockCols)->default_value(lhsBlockCols),
       "The number of blocks columns for the LHS matrix")
      // lhs-block-row
      ("lhs-block-row",
       po::value<int>(&lhsBlockRowSize)->default_value(lhsBlockRowSize),
       "The block size for the row of rhe LHS matrix")
      // lhs-block-col
      ("lhs-block-col",
       po::value<int>(&lhsBlockColSize)->default_value(lhsBlockColSize),
       "The block size for the column of the LHS matrix")
      // rhs-block
      ("rhs-block", po::value<int>(&rhsBlockSize)->default_value(rhsBlockSize),
       "The block col for the right matrix that does not need be transposed or "
       "the block row for the right matrix that needs be transposed")
      // is-rhs-matrix-sparse
      ("is-rhs-matrix-sparse",
       po::value<int>(&isRhsMatrixSparse)->default_value(isRhsMatrixSparse),
       "RHS matrix is sparse or dense")
      // rhs-need-transpose
      ("rhs-need-transpose",
       po::value<int>(&rhsNeedTranspose)->default_value(rhsNeedTranspose),
       "the right matrix need be transposed")
      // is-res-matrix-sparse
      ("is-res-matrix-sparse",
       po::value<int>(&isResMatrixSparse)->default_value(isResMatrixSparse),
       "Result matrix is sparse or dense")
      // sparsity-matrix
      ("sparsity-matrix",
       po::value<std::string>(&sparsityFileName)
           ->default_value(sparsityFileName),
       "The file name for the sparsity mask")
      // memory-cycle-ratio
      ("memory-cycle-ratio",
       po::value<float>(&memoryCycleRatio)->default_value(memoryCycleRatio),
       "the ratio between memory weight and cycle weight")
      // partition-method
      ("partition-method",
       po::value<std::string>(&partitionMethod)->default_value(partitionMethod),
       "The method to generate the computation graph: block, block-naive")
      // runs
      ("runs", po::value<unsigned>(&runs)->default_value(runs),
       "Number of calls to Engine::run")
      // number-or-groups
      ("number-of-groups", po::value<int>(&numGroups), "Number of groups")
      // check-result
      ("check-result", po::value<int>(&checkResult)->default_value(checkResult),
       "check if the ressult is correct")
      // Sub-block mask
      ("sub-block-mask",
       po::value<std::string>(&subBlockMask)->default_value(subBlockMask),
       "the mask inside a block: None, ZeroUpperTriangle, "
       "ZeroLowerTriangle");

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help") || !vm.count("sparsity-matrix")) {
      std::cout << desc << "\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  if (isRhsMatrixSparse && isResMatrixSparse) {
    std::cerr << "error: If the resulting matrix is sparse, both LHS and RHS "
                 "matrices must be dense.\n";
    return 1;
  } else if (!isRhsMatrixSparse && !isResMatrixSparse) {
    std::cerr << "error: All matrices are dense. Nothing to test.\n";
    return 1;
  }
  if (batchSize % lhsBlockRowSize != 0) {
    std::cerr << "error: The batch size " << batchSize
              << " is not divisible by the number of rows in a block for the "
                 "LHS matrix "
              << lhsBlockRowSize << ".\n";
    return 1;
  }
  if (numGroups > 1) {
    printf("Number of groups: %d\n", numGroups);
  }

  bool compileIPUCode = false;
  if (isIpuModel(deviceType))
    compileIPUCode = true;
  auto device = createTestDevice(deviceType, ipuModel.numIPUs,
                                 ipuModel.tilesPerIPU, compileIPUCode);

  const auto &target = device.getTarget();
  Graph graph(target);

  std::array<int, 3> dim;

  SubBlockMask mask = SubBlockMask::None;
  if (subBlockMask == "ZeroUpperTriangle") {
    mask = SubBlockMask::ZeroUpperTriangle;
  } else if (subBlockMask == "ZeroLowerTriangle") {
    mask = SubBlockMask::ZeroLowerTriangle;
  } else if (subBlockMask != "None") {
    logging::err("Unrecognized sub-block mask parameter: {}.",
                 subBlockMask.c_str());
    return -1;
  }
  int nonZeroBlock = 0;
  int sparsityRows, sparsityCols;
  std::vector<unsigned char> sparsity;
  std::size_t sparsityDenseSize = 0;
  if (!ReadMatrixMask(sparsityFileName, nonZeroBlock, sparsityRows,
                      sparsityCols, sparsity))
    return -1;
  if (numGroups == 1) {
    printf("non zero block: %d\n", nonZeroBlock);
    printf("sparsity: %f\n",
           1.0 -
               nonZeroBlock / static_cast<float>(sparsityRows * sparsityCols));
  } else {
    if (sparsityRows % numGroups != 0) {
      logging::err(
          "error: Number of rows {} in concatenated group sparsity mask "
          "is not divisible "
          "by the number of groups {}.",
          sparsityRows, numGroups);
      return -1;
    }
    printf("Average non zero block: %f\n",
           static_cast<float>(nonZeroBlock) / numGroups);
    printf("Average sparsity: %f\n",
           1.0 -
               nonZeroBlock / static_cast<float>(sparsityRows * sparsityCols));
    sparsityRows /= numGroups;
  }
  sparsityDenseSize = sparsityRows * sparsityCols;

  bool doProfiling = vm.count("profile") > 0;
  bool doProfilingExecution = doProfiling && vm.count("profile-execution") > 0;
  bool doProfilingVars = doProfiling && vm.count("profile-vars") > 0;

  const int lhsRows = batchSize;
  const int rhsBlockCols = sparsityCols;
  const int lhsBlockRows = lhsRows / lhsBlockRowSize;

  if (isResMatrixSparse) {
    if (lhsBlockRows != sparsityRows) {
      std::cerr << "error: LHS number of block rows " << lhsBlockRows
                << " does not match the number of block rows in output matrix "
                << sparsityRows << ".\n";
      return 1;
    }
  } else {
    if (lhsBlockCols != sparsityRows) {
      std::cerr << "error: LHS number of block columns " << lhsBlockCols
                << " does not match the number of block rows in RHS matrix "
                << sparsityRows << ".\n";
      return 1;
    }
  }

  const int resBlockRows = lhsBlockRows;
  const int resBlockCols = rhsBlockCols;
  const int resBlockRowSize = lhsBlockRowSize;
  const int resBlockColSize = rhsBlockSize;

  dim[0] = lhsRows;
  dim[1] = lhsBlockCols * lhsBlockColSize;
  dim[2] = rhsBlockCols * rhsBlockSize;

  std::array<int, 3> blockSize = {lhsBlockRowSize, lhsBlockColSize,
                                  rhsBlockSize};

  BSMatMulParams bsMatMulObj =
      createBsMatMul(dim, blockSize, sparsity, dataType, isResMatrixSparse,
                     (bool)rhsNeedTranspose, mask, numGroups);

  double flops;
  if (vm.count("dense-matmul") != 0) {
    flops = dim[0] * dim[1] * dim[2] * 2.0;
  } else {
    if (!isResMatrixSparse) {
      flops = nonZeroBlock * dim[0] * blockSize[1] * blockSize[2] * 2.0;
    } else {
      flops = nonZeroBlock * dim[1] * blockSize[0] * blockSize[2] * 2.0;
    }
  }

  boost::multi_array<float, 2> lhsHost;
  boost::multi_array<float, 2> rhsHost;
  boost::multi_array<float, 3> lhsHost3D;
  boost::multi_array<float, 3> rhsHost3D;
  if (numGroups == 1) {
    lhsHost.resize(boost::extents[dim[0]][dim[1]]);
    if (!rhsNeedTranspose) {
      rhsHost.resize(boost::extents[dim[1]][dim[2]]);
    } else {
      rhsHost.resize(boost::extents[dim[2]][dim[1]]);
    }
    populateMatrixData(dim, blockSize, lhsHost, rhsHost,
                       (!isRhsMatrixSparse ? nullptr : sparsity.data()),
                       rhsNeedTranspose);
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

  const std::string debugPrefix = "bs_matmul";
  Sequence matSeq, uploadProg, downloadProg;

  std::unique_ptr<char[]> lhsRawHost, rhsRawHost, outputRawHost;
  std::vector<std::unique_ptr<char[]>> rhsBlocksRawHost, outputBlocksRawHost;

  std::vector<std::pair<std::string, char *>> tmap;

  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  if (vm.count("dense-matmul") != 0) {
    poplin::matmul::PlanningCache cache;
    OptionFlags mmOpt;

    // LHS matrix
    poplar::Tensor lhsTensor = graph.addVariable(
        dataType, {(unsigned long)dim[0], (unsigned long)dim[1]},
        debugPrefix + "matrix_lhs");
    poputil::mapTensorLinearly(graph, lhsTensor);
    lhsRawHost = allocateHostMemoryForTensor(lhsTensor, "matrix_lhs", graph,
                                             uploadProg, downloadProg, tmap);
    copy(target, lhsHost, dataType, lhsRawHost.get());

    // RHS matrix
    poplar::Tensor rhsTensor = graph.addVariable(
        dataType, {(unsigned long)dim[1], (unsigned long)dim[2]},
        debugPrefix + "matrix_rhs");
    poputil::mapTensorLinearly(graph, rhsTensor);
    rhsRawHost = allocateHostMemoryForTensor(rhsTensor, "matrix_rhs", graph,
                                             uploadProg, downloadProg, tmap);
    copy(target, rhsHost, dataType, rhsRawHost.get());

    // output matrix
    poplar::Tensor outTensor = matMul(graph, lhsTensor, rhsTensor, matSeq,
                                      debugPrefix + "lhsxrhs", mmOpt, &cache);

    outputRawHost = poplibs_test::util::allocateHostMemoryForTensor(
        outTensor, "matrix_output", graph, uploadProg, downloadProg, tmap);
  } else {
    // LHS dense matrix
    poplar::Tensor lhsTensor =
        createBSMatMulInputLHS(graph, bsMatMulObj, "matrix_lhs");
    poplar::Tensor rhsTensor =
        createBSMatMulInputRHS(graph, bsMatMulObj, "matrix_rhs");

    poplar::OptionFlags options = {
        {"memoryCycleRatio", std::to_string(memoryCycleRatio)},
        {"partitionMethod", partitionMethod},
    };

    // sparse matmul
    poplar::Tensor outTensor;
    try {
      outTensor = bsMatMul(graph, bsMatMulObj, matSeq, lhsTensor, rhsTensor,
                           options, "bs-matmul");
    } catch (const poputil::poplibs_error &e) {
      std::cout << "bsMatMul() failed" << std::endl;
      return -1;
    }
    if (isResMatrixSparse) {
      assert((int)outTensor.dim(0) == nonZeroBlock);
    }

    if (checkResult) {
      lhsRawHost = allocateHostMemoryForTensor(lhsTensor, "matrix_lhs", graph,
                                               uploadProg, downloadProg, tmap);
      if (numGroups == 1) {
        copy(target, lhsHost, dataType, lhsRawHost.get());
      } else {
        copy(target, lhsHost3D, dataType, lhsRawHost.get());
      }

      if (isRhsMatrixSparse) {
        // RHS sparse matrix
        boost::multi_array<float, 1> rhsBlocksHost;
        if (numGroups == 1) {
          if (!rhsNeedTranspose) {
            getSparseMatrixBlocks(dim[1], dim[2], blockSize[1], blockSize[2],
                                  sparsity.data(), nonZeroBlock, rhsHost,
                                  rhsBlocksHost);
          } else {
            getSparseMatrixBlocks(dim[2], dim[1], blockSize[2], blockSize[1],
                                  sparsity.data(), nonZeroBlock, rhsHost,
                                  rhsBlocksHost);
          }
        } else {
          if (!rhsNeedTranspose) {
            getSparseMatrixBlocks(dim[1], dim[2], blockSize[1], blockSize[2],
                                  numGroups, sparsity.data(), nonZeroBlock,
                                  rhsHost3D, rhsBlocksHost);
          } else {
            getSparseMatrixBlocks(dim[2], dim[1], blockSize[2], blockSize[1],
                                  numGroups, sparsity.data(), nonZeroBlock,
                                  rhsHost3D, rhsBlocksHost);
          }
        }
        rhsRawHost =
            allocateHostMemoryForTensor(rhsTensor, debugPrefix + "/matrix_rhs",
                                        graph, uploadProg, downloadProg, tmap);
        copy(target, rhsBlocksHost, dataType, rhsRawHost.get());
      } else {
        // RHS dense matrix
        rhsRawHost =
            allocateHostMemoryForTensor(rhsTensor, debugPrefix + "/matrix_rhs",
                                        graph, uploadProg, downloadProg, tmap);
        if (numGroups == 1) {
          copy(target, rhsHost, dataType, rhsRawHost.get());
        } else {
          copy(target, rhsHost3D, dataType, rhsRawHost.get());
        }
      }
    }

    outputRawHost = poplibs_test::util::allocateHostMemoryForTensor(
        outTensor, "matrix_output", graph, uploadProg, downloadProg, tmap);
  }

  auto engineOptions = defaultEngineOptions;
  if (doProfiling) {
    engineOptions.set("debug.instrumentCompute", "true");
    engineOptions.set("debug.instrumentExternalExchange", "true");
    engineOptions.set("debug.loweredVarDumpFile", profileDir + "/vars.capnp");
  }

  Sequence allSequence;
  // if run many times, we are doing benchmark, so ignore the host data copy
  if (checkResult == 1) {
    allSequence.add(uploadProg);
    allSequence.add(matSeq);
    allSequence.add(downloadProg);
  } else {
    allSequence.add(matSeq);
  }

  Engine engine(graph, allSequence, engineOptions);

  std::cout << "Start run\n";
  poplibs_test::util::attachStreams(engine, tmap);
  device.bind([&](const Device &d) {
    engine.load(d);
    std::cout << "dim: " << dim[0] << " " << dim[1] << " " << dim[2] << "\n";
    std::cout << "block size: " << blockSize[0] << " x " << blockSize[1]
              << " x " << blockSize[2] << "\n";
    if (rhsNeedTranspose)
      std::cout << "Right matrix need be transposed\n";
    std::cout << "batch size: " << batchSize << "\n";
    std::cout << "non zero block: " << nonZeroBlock << "\n";
    std::cout << "sparsity: "
              << 1.0 - nonZeroBlock / (float)(sparsityRows * sparsityCols)
              << "\n";
    auto start = std::chrono::system_clock::now();
    for (unsigned long i = 0; i < runs; i++)
      engine.run(0);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedSeconds = end - start;
    std::cout << "elapsed time: " << elapsedSeconds.count() << " s\n";

    double avgTime = elapsedSeconds.count() / runs;
    std::cout << "average kernel run time: " << avgTime * 1000000 << " mcs\n";
    std::cout << "flop: " << flops << "\n";
    std::cout << "GFLOPS: " << flops / 1000000000.0 / avgTime << "\n";
  });

  if (deviceType != DeviceType::Cpu && doProfiling) {
    engine.printProfileSummary(
        std::cout,
        OptionFlags{
            {"showExecutionSteps", doProfilingExecution ? "true" : "false"},
            {"showVarStorage", doProfilingVars ? "true" : "false"}});
    if (vm.count("profile-dir"))
      savePoplarReport(engine, profileDir);
  }

  if (!checkResult)
    return 0;

  std::cout << "Checking the result...\n";
  boost::multi_array<float, 4> denseMatC(boost::extents[1][1][dim[0]][dim[2]]);
  boost::multi_array<float, 5> denseMatC3D(
      boost::extents[numGroups][1][1][dim[0]][dim[2]]);
  std::fill_n(denseMatC.data(), denseMatC.num_elements(), 0.0f);
  if (numGroups == 1) {
    if (!isResMatrixSparse) {
      copy(target, dataType, outputRawHost.get(), denseMatC);
    } else {
      denseMatC.reshape(std::vector<int>(
          {resBlockRows, resBlockCols, resBlockRowSize, resBlockColSize}));
      boost::multi_array<float, 3> denseMatCTmp(
          boost::extents[nonZeroBlock][resBlockRowSize][resBlockColSize]);
      copy(target, dataType, outputRawHost.get(), denseMatCTmp);
      int blockCount = 0;
      for (int br = 0; br < resBlockRows; ++br) {
        for (int bc = 0; bc < resBlockCols; ++bc) {
          if (sparsity[br * resBlockCols + bc] > 0) {
            denseMatC[br][bc] = denseMatCTmp[blockCount++];
          }
        }
      }
    }
  } else {
    if (!isResMatrixSparse) {
      copy(target, dataType, outputRawHost.get(), denseMatC3D);
    } else {
      denseMatC3D.reshape(
          std::vector<int>({static_cast<int>(numGroups), resBlockRows,
                            resBlockCols, resBlockRowSize, resBlockColSize}));
      const unsigned char *itemSparsity = sparsity.data();
      boost::multi_array<float, 3> denseMatCTmp(
          boost::extents[nonZeroBlock][resBlockRowSize][resBlockColSize]);
      copy(target, dataType, outputRawHost.get(), denseMatCTmp);
      int blockCount = 0;
      for (int g = 0; g < numGroups; ++g) {
        for (int br = 0; br < resBlockRows; ++br) {
          for (int bc = 0; bc < resBlockCols; ++bc) {
            if (itemSparsity[br * resBlockCols + bc] > 0) {
              denseMatC3D[g][br][bc] = denseMatCTmp[blockCount++];
            }
          }
        }
        itemSparsity += sparsityDenseSize;
      }
    }
  }

  std::vector<std::vector<float>> hostMatC(dim[0],
                                           std::vector<float>(dim[2], 0.0f));

  std::vector<std::vector<std::vector<float>>> hostMatC3D(
      numGroups, std::vector<std::vector<float>>(
                     dim[0], std::vector<float>(dim[2], 0.0f)));

  if (!isResMatrixSparse) {
    for (int r = 0; r < dim[0]; r++) {
      for (int c = 0; c < dim[2]; c++) {
        if (numGroups == 1) {
          float sum = 0.0f;
          for (int cmul = 0; cmul < dim[1]; cmul++) {
            sum += lhsHost[r][cmul] *
                   (rhsNeedTranspose ? (rhsHost[c][cmul]) : (rhsHost[cmul][c]));
          }
          hostMatC[r][c] = sum;
        } else {
          for (int g = 0; g < numGroups; ++g) {
            float sum = 0.0f;
            for (int cmul = 0; cmul < dim[1]; cmul++) {
              sum += lhsHost3D[g][r][cmul] * (rhsNeedTranspose
                                                  ? (rhsHost3D[g][c][cmul])
                                                  : (rhsHost3D[g][cmul][c]));
            }
            hostMatC3D[g][r][c] = sum;
          }
        }
      }
    }
  } else {
    for (int r = 0; r < dim[0]; r++) {
      for (int c = 0; c < dim[2]; c++) {
        if (numGroups == 1) {
          float sum = 0.0f;
          if (!((subBlockMask == "ZeroUpperTriangle" && r < c) ||
                (subBlockMask == "ZeroLowerTriangle" && r > c))) {
            int br = r / lhsBlockRowSize;
            int bc = c / rhsBlockSize;
            if (sparsity[br * rhsBlockCols + bc] != 0) {
              for (int cmul = 0; cmul < dim[1]; cmul++) {
                sum +=
                    lhsHost[r][cmul] * (rhsNeedTranspose ? (rhsHost[c][cmul])
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
              int br = r / lhsBlockRowSize;
              int bc = c / rhsBlockSize;
              if (itemSparsity[br * rhsBlockCols + bc] != 0) {
                for (int cmul = 0; cmul < dim[1]; cmul++) {
                  sum += lhsHost3D[g][r][cmul] *
                         (rhsNeedTranspose ? (rhsHost3D[g][c][cmul])
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

  const float threshold = dataType == poplar::FLOAT ? 1.0e-4 : 1.0e-2;
  float maxErr = 0.0f;
  int errRow = 0;
  int errCol = 0;
  float valTruth = 0.0f;
  float valTest = 0.0f;
  bool success = true;
  for (int g = 0; g < numGroups; ++g) {
    for (int r = 0; r < dim[0]; r++) {
      for (int c = 0; c < dim[2]; c++) {
        float err;
        float curValTruth, curValTest;
        if (numGroups == 1) {
          curValTruth = hostMatC[r][c];
        } else {
          curValTruth = hostMatC3D[g][r][c];
        }
        if (!isResMatrixSparse) {
          if (numGroups == 1) {
            curValTest = denseMatC[0][0][r][c];
          } else {
            curValTest = denseMatC3D[g][0][0][r][c];
          }
          err = fabs(curValTest - curValTruth);
        } else {
          int br = r / resBlockRowSize;
          int bc = c / resBlockColSize;
          int rb = r % resBlockRowSize;
          int cb = c % resBlockColSize;
          if (numGroups == 1) {
            curValTest = denseMatC[br][bc][rb][cb];
          } else {
            curValTest = denseMatC3D[g][br][bc][rb][cb];
          }
          err = fabs(curValTest - curValTruth);
        }
        if (!checkIsClose(curValTruth, curValTest, threshold)) {
          success = false;
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

  if (success)
    std::cout << "Result check succeeded.\n";
  else
    std::cout << "Result check failed."
              << " Maximum error = " << std::setprecision(5) << maxErr
              << " test[" << errRow << "][" << errCol
              << "] = " << std::setprecision(5) << valTest << " true[" << errRow
              << "][" << errCol << "] = " << std::setprecision(5) << valTruth
              << "\n";

  return (maxErr > threshold ? -1 : 0);
}
