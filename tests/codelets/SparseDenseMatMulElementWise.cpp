// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "TestDevice.hpp"

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

#include <iostream>
#include <random>
#include <vector>

#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <poplar/Graph.hpp>

#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/VectorUtils.hpp>
#include <poplibs_support/print.hpp>

#include <popsparse/codelets.hpp>

#include <popsolver/Model.hpp>

#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/Util.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;
using namespace poplibs_support;

enum class VertexType {
  Forward,
  GradA,      // version where a separate meta information table is used
  Transposed, // Uses forward meta information table
  GradW,
};

std::ostream &operator<<(std::ostream &os, const VertexType &vt) {
  switch (vt) {
  case VertexType::Forward:
    os << "Forward";
    break;
  case VertexType::GradA:
    os << "GradA";
    break;
  case VertexType::Transposed:
    os << "Transposed";
    break;
  case VertexType::GradW:
    os << "GradW";
    break;
  default:
    throw poplibs_error(
        "Unrecognised vertex type " +
        std::to_string(std::underlying_type<VertexType>::type(vt)));
  }
  return os;
}

std::istream &operator>>(std::istream &is, VertexType &vt) {
  std::string token;
  is >> token;
  if (token == "Forward") {
    vt = VertexType::Forward;
  } else if (token == "GradA") {
    vt = VertexType::GradA;
  } else if (token == "Transposed") {
    vt = VertexType::Transposed;
  } else if (token == "GradW") {
    vt = VertexType::GradW;
  } else {
    throw poplibs_error("Unrecognised vertex type '" + token + "'");
  }
  return is;
}

template <typename RandomEngine>
static std::vector<std::array<unsigned, 2>>
generateSparseIndices(RandomEngine &randomEngine,
                      const std::vector<std::size_t> &shape, std::size_t n) {
  // Generate n random indices that are within the flattened given shape.
  std::vector<unsigned> randomIndices(product(shape));
  std::iota(randomIndices.begin(), randomIndices.end(), 0);
  auto randomGen = [&](unsigned max) {
    boost::random::uniform_int_distribution<unsigned> dist(0, max - 1);
    return dist(randomEngine);
  };
  boost::range::random_shuffle(randomIndices, randomGen);
  randomIndices.resize(n);

  std::vector<std::array<unsigned, 2>> rowColumnIndices(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto unflattenedIndex =
        vectorConvert<unsigned>(unflattenIndex(shape, randomIndices[i]));
    rowColumnIndices[i] = {unflattenedIndex[0], unflattenedIndex[1]};
  }
  return rowColumnIndices;
}

template <typename RandomEngine>
static std::tuple<unsigned, std::vector<unsigned>>
generateSparseSubGroupIds(RandomEngine &randomEngine, std::size_t n,
                          unsigned min, unsigned max) {
  std::vector<unsigned> subGroupIds(max - min + 1);
  std::iota(subGroupIds.begin(), subGroupIds.end(), min);
  auto randomGen = [&](unsigned max) {
    boost::random::uniform_int_distribution<unsigned> dist(0, max);
    return dist(randomEngine);
  };
  boost::range::random_shuffle(subGroupIds, randomGen);
  const auto processedSubGroupId = subGroupIds.back();
  subGroupIds.resize(n - 1);
  return std::make_tuple(processedSubGroupId, subGroupIds);
}

template <typename RandomEngine>
static std::tuple<std::vector<std::vector<unsigned>>,
                  std::vector<std::vector<unsigned>>>
partitionSubGroupElems(RandomEngine &randomEngine, std::size_t numIndices,
                       unsigned numBuckets, unsigned numSplitsPerBucket,
                       unsigned numOtherSubGroups,
                       unsigned numOtherSubGroupElems) {
  std::vector<unsigned> numElemsOtherSubGroups(numOtherSubGroups);
  {
    auto randomDist = boost::random::uniform_int_distribution<unsigned>(
        1, numOtherSubGroupElems);
    unsigned sum = 0;
    for (auto &n : numElemsOtherSubGroups) {
      n = randomDist(randomEngine);
      sum += n;
    }
    unsigned newSum = 0;
    if (sum > 0) {
      for (auto &n : numElemsOtherSubGroups) {
        n = (n * numOtherSubGroupElems) / sum;
        newSum += n;
      }
    }
    // Round robin assign any rounding error to get exactly the total number we
    // asked for.
    for (std::size_t i = 0; numOtherSubGroupElems - newSum != 0;
         ++newSum, i = (i + 1) % numElemsOtherSubGroups.size()) {
      numElemsOtherSubGroups[i]++;
    }
  }
  std::vector<std::vector<unsigned>> processedSubGroupIndices(numBuckets);
  std::vector<std::vector<unsigned>> numElems(numBuckets);
  const auto randomGenForShuffle = [&](unsigned max) {
    boost::random::uniform_int_distribution<unsigned> dist(0, max - 1);
    return dist(randomEngine);
  };
  const auto maxIndicesPerBucket = ceildiv(numIndices, numBuckets);
  for (std::size_t bucket = 0; bucket < numBuckets; ++bucket) {
    numElems[bucket].resize(numSplitsPerBucket + numOtherSubGroups);
    const auto bucketStartIndex = bucket * maxIndicesPerBucket;
    const auto bucketEndIndex =
        std::min((bucket + 1) * maxIndicesPerBucket, numIndices);
    const auto maxIndicesPerSplit =
        ceildiv(bucketEndIndex - bucketStartIndex, numSplitsPerBucket);
    std::vector<unsigned> indices(numOtherSubGroups + numSplitsPerBucket);
    std::iota(indices.begin(), indices.end(), 0);
    boost::range::random_shuffle(indices, randomGenForShuffle);
    for (std::size_t splitIdx = 0; splitIdx < numSplitsPerBucket; ++splitIdx) {
      const auto subGroupIndex = indices.at(splitIdx);
      const auto startIndex = bucketStartIndex + splitIdx * maxIndicesPerSplit;
      const auto endIndex =
          std::min(bucketStartIndex + (splitIdx + 1) * maxIndicesPerSplit,
                   bucketEndIndex);
      numElems[bucket].at(subGroupIndex) = endIndex - startIndex;
    }
    for (std::size_t otherIdx = 0; otherIdx < numOtherSubGroups; ++otherIdx) {
      const auto subGroupIndex = indices.at(otherIdx + numSplitsPerBucket);
      numElems[bucket].at(subGroupIndex) = numElemsOtherSubGroups.at(otherIdx);
    }
    indices.resize(numSplitsPerBucket);
    std::sort(indices.begin(), indices.end());
    std::swap(processedSubGroupIndices[bucket], indices);
  }

  return std::make_tuple(processedSubGroupIndices, numElems);
}

static std::tuple<unsigned, unsigned, unsigned>
getForwardWorkerPartition(const Target &target, const Type &inputType,
                          const unsigned bColumns, const unsigned aRows,
                          const std::vector<unsigned> &aRowColumnCounts) {
  // Split rows of a and columns of b between workers.
  //
  // For this functional test we'll just first split columns of b, then
  // rows of a to try and utilise all workers.
  //
  // NOTE: A problem with this is that by needing a contiguous range of
  // columns to process, in the given order, we are restricting how work
  // can be split. We cannot change the order of columns because that
  // would make things difficult for other passes. We could do some more
  // heavy encoding to allow interleaved columns to be selected for
  // workers but it's more memory to encode.
  const auto bColumnGrainSize = target.getVectorWidth(inputType);
  const auto bColumnGrains = ceildiv(bColumns, bColumnGrainSize);

  popsolver::Model m;
  const auto mWorkerARowsPartition = m.addVariable(1, aRows);
  const auto mWorkerBColumnGrainsPartition = m.addVariable(1, bColumnGrains);

  const auto mNumARows = m.addConstant(aRows);
  const auto mNumBColumnGrains = m.addConstant(bColumnGrains);

  const auto mWorkerARows = m.ceildiv(mNumARows, mWorkerARowsPartition);
  const auto mWorkerBColumnGrains =
      m.ceildiv(mNumBColumnGrains, mWorkerBColumnGrainsPartition);

  const auto numWorkers = target.getNumWorkerContexts();
  const auto mMaxWorkerAElems =
      m.call({mWorkerARows}, [&](const std::vector<unsigned> &values) {
        const auto workerARows = values[0];
        unsigned maxWorkerAElems = 0;
        for (unsigned worker = 0; worker < numWorkers; ++worker) {
          const auto elems = std::accumulate(
              aRowColumnCounts.begin() + std::min(worker * workerARows, aRows),
              aRowColumnCounts.begin() +
                  std::min((worker + 1) * workerARows, aRows),
              0u);
          maxWorkerAElems = std::max(maxWorkerAElems, elems);
        }
        return maxWorkerAElems;
      });

  m.lessOrEqual(
      m.product({mWorkerARowsPartition, mWorkerBColumnGrainsPartition}),
      numWorkers);

  const auto mMaxWorkerGrains =
      m.product({mMaxWorkerAElems, mWorkerBColumnGrains});

  const auto s = m.minimize(mMaxWorkerGrains);
  if (!s.validSolution()) {
    throw poplibs_error("Failed to find a plan to split work between workers!");
  }

  const auto numAPartitions = s[mWorkerARowsPartition];
  const auto numBPartitions = s[mWorkerBColumnGrainsPartition];
  return std::make_tuple(bColumnGrainSize, numAPartitions, numBPartitions);
}

static std::tuple<unsigned, unsigned>
getGradWWorkerPartition(const Target &target, const Type &inputType,
                        const unsigned bColumns, const unsigned aRows,
                        const std::vector<unsigned> &aRowColumnCounts) {
  // Much easier than forward partitions. Just partition the total number
  // of columns of A between workers.
  const auto totalAElems = std::accumulate(
      aRowColumnCounts.begin(), aRowColumnCounts.end(), std::size_t(0));

  // Grain size of columns of b is the same as for forward for GradW codelet.
  const auto bColumnGrainSize = target.getVectorWidth(inputType);
  const auto numWorkers = target.getNumWorkerContexts();
  const auto aElemsPerWorker = ceildiv(totalAElems, numWorkers);
  const auto numAPartitions = ceildiv(totalAElems, aElemsPerWorker);

  return std::make_tuple(bColumnGrainSize, numAPartitions);
}

template <typename RandomEngine>
static std::vector<std::vector<unsigned>> generateMetaInfoAndPartition(
    RandomEngine &randomEngine, std::vector<std::array<unsigned, 2>> &indices,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, unsigned numBuckets,
    unsigned processedSubGroupId, const std::vector<unsigned> &otherSubGroupIds,
    const std::vector<std::vector<unsigned>> processedSubGroupIndices,
    const std::vector<std::vector<unsigned>> &subGroupNumElems,
    const Target &target, const Type &inputType, const Type &partialType,
    VertexType vertexType) {
  const auto generateGradWMetaInfo = vertexType == VertexType::GradW;
  const auto generateGradAMetaInfo = vertexType == VertexType::GradA;

  // Factor by which row and column offsets are scaled
  const auto yOffsetFactor = inputType == FLOAT ? 4 : 2;
  const auto xOffsetFactor = inputType == FLOAT ? 1 : 2;

  // Order indices of a by column then row
  std::sort(indices.begin(), indices.end());

  std::vector<std::vector<unsigned>> metaInfo(numBuckets);
  auto garbageDist =
      boost::random::uniform_int_distribution<unsigned>(0, 0xffff);
  std::size_t nzOffset = 0;
  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    std::size_t splitIdx = 0;
    std::size_t numSplits = processedSubGroupIndices.at(bucket).size();
    for (std::size_t i = 0; i < otherSubGroupIds.size() + numSplits; ++i) {
      if (splitIdx != numSplits &&
          i == processedSubGroupIndices[bucket][splitIdx]) {

        std::vector<unsigned> rows;
        std::vector<unsigned> rowColumnCounts;
        boost::optional<unsigned> lastRowIndex;
        for (std::size_t nzIdx = nzOffset;
             nzIdx < nzOffset + subGroupNumElems[bucket].at(i); ++nzIdx) {
          if (!lastRowIndex || *lastRowIndex != indices.at(nzIdx)[0]) {
            rows.emplace_back();
            rowColumnCounts.emplace_back();
          }
          rows.back() = indices.at(nzIdx)[0];
          ++rowColumnCounts.back();
          lastRowIndex = rows.back();
        }

        const auto bColumns = bShape[1];

        unsigned fwdBColumnGrainSize, fwdNumAPartitions, fwdNumBPartitions;
        std::tie(fwdBColumnGrainSize, fwdNumAPartitions, fwdNumBPartitions) =
            getForwardWorkerPartition(target, inputType, bColumns, rows.size(),
                                      rowColumnCounts);

        const auto fwdBColumnGrains = ceildiv(bColumns, fwdBColumnGrainSize);
        const auto fwdNumUsedWorkers = fwdNumAPartitions * fwdNumBPartitions;
        const auto fwdMaxPartitionARows =
            ceildiv(rows.size(), fwdNumAPartitions);
        const auto fwdMaxPartitionBColumnGrains =
            ceildiv(fwdBColumnGrains, fwdNumBPartitions);

        std::vector<unsigned> fwdPartitionAElemOffsets(fwdNumAPartitions, 0);
        for (unsigned partition = 1; partition < fwdNumAPartitions;
             ++partition) {
          const auto prevPartitionARowStart =
              (partition - 1) * fwdMaxPartitionARows;
          const auto prevPartitionARowEnd = partition * fwdMaxPartitionARows;
          fwdPartitionAElemOffsets[partition] =
              fwdPartitionAElemOffsets[partition - 1] +
              std::accumulate(rowColumnCounts.begin() + prevPartitionARowStart,
                              rowColumnCounts.begin() + prevPartitionARowEnd,
                              unsigned(0));
        }

        unsigned gradWNumUsedWorkers, gradWBColumnGrainSize,
            gradWNumAPartitions = 0;
        if (generateGradWMetaInfo) {
          std::tie(gradWBColumnGrainSize, gradWNumAPartitions) =
              getGradWWorkerPartition(target, inputType, bColumns, rows.size(),
                                      rowColumnCounts);
          gradWNumUsedWorkers = gradWNumAPartitions;
        }

        metaInfo[bucket].emplace_back(processedSubGroupId);
        const auto processedSubGroupNumElems = subGroupNumElems[bucket].at(i);
        const auto elemsPerNzElem = generateGradAMetaInfo ? 2 : 1;
        metaInfo[bucket].emplace_back(processedSubGroupNumElems);
        const auto totalMetaInfoElems =
            7 + fwdNumUsedWorkers * 5 + rows.size() * 2 +
            processedSubGroupNumElems * elemsPerNzElem +
            (generateGradWMetaInfo ? 1 + 4 * gradWNumUsedWorkers : 0);
        const auto offsetToFirstOutputEntry =
            totalMetaInfoElems -
            (rows.size() * 2 + processedSubGroupNumElems * elemsPerNzElem);

        metaInfo[bucket].emplace_back(totalMetaInfoElems);
        metaInfo[bucket].emplace_back(bColumns);
        metaInfo[bucket].emplace_back(rows.size() - 1);
        metaInfo[bucket].emplace_back(offsetToFirstOutputEntry);
        metaInfo[bucket].emplace_back(fwdNumUsedWorkers);

        // Reserve space for worker entries
        std::vector<std::size_t> metaInfoFwdWorkerEntryIndices(
            fwdNumUsedWorkers);
        for (unsigned worker = 0; worker < fwdNumUsedWorkers; ++worker) {
          metaInfoFwdWorkerEntryIndices[worker] = metaInfo[bucket].size();
          for (std::size_t i = 0; i < 5; ++i) {
            metaInfo[bucket].emplace_back(~0u);
          }
        }

        // If needed reserve space for GradW worker entries
        std::vector<unsigned> metaInfoGradWWorkerEntryIndices;
        if (generateGradWMetaInfo) {
          metaInfo[bucket].emplace_back(gradWNumUsedWorkers);

          metaInfoGradWWorkerEntryIndices.resize(gradWNumUsedWorkers);
          for (unsigned worker = 0; worker < gradWNumUsedWorkers; ++worker) {
            metaInfoGradWWorkerEntryIndices[worker] = metaInfo[bucket].size();
            for (std::size_t i = 0; i < 4; ++i) {
              metaInfo[bucket].emplace_back(~0u);
            }
          }
        }

        // Output row -> column list meta-info
        std::vector<unsigned> outputEntryMetaInfoIndices(rows.size());
        std::size_t offsetGradA = 0;
        for (std::size_t r = 0; r < rows.size(); ++r) {
          const auto aRow = indices.at(nzOffset)[0];
          // First entry is offset into output memory to process.
          // bColumns are inner-most dimension.
          const auto aRowOffsetInC = aRow * bColumns;
          outputEntryMetaInfoIndices[r] = metaInfo[bucket].size();
          metaInfo[bucket].push_back(aRowOffsetInC * xOffsetFactor);
          // Use 1 less for float input type
          metaInfo[bucket].push_back(rowColumnCounts[r]);
          for (unsigned c = 0; c < rowColumnCounts[r]; ++c) {
            if (generateGradAMetaInfo) {
              metaInfo[bucket].push_back(yOffsetFactor * offsetGradA++);
            }
            metaInfo[bucket].push_back(indices.at(nzOffset)[1] * bColumns *
                                       yOffsetFactor);
            ++nzOffset;
          }
        }

        // Fill out worklist info for each worker
        for (unsigned worker = 0; worker < fwdNumUsedWorkers; ++worker) {
          const auto aPartitionIdx = worker % fwdNumAPartitions;
          const auto bPartitionIdx = worker / fwdNumAPartitions;
          const auto aRowIndex = aPartitionIdx * fwdMaxPartitionARows;
          const auto aRowEndIndex =
              std::min((aPartitionIdx + 1) * fwdMaxPartitionARows, rows.size());
          const auto bColumnIndex = bPartitionIdx *
                                    fwdMaxPartitionBColumnGrains *
                                    fwdBColumnGrainSize;
          const auto bColumnEndIndex =
              std::min((bPartitionIdx + 1) * fwdMaxPartitionBColumnGrains *
                           fwdBColumnGrainSize,
                       bColumns);
          const auto workerEntryIndex = metaInfoFwdWorkerEntryIndices[worker];
          metaInfo[bucket][workerEntryIndex + 0] =
              generateGradAMetaInfo ? 0
                                    : fwdPartitionAElemOffsets[aPartitionIdx];
          metaInfo[bucket][workerEntryIndex + 1] =
              bColumnEndIndex - bColumnIndex;
          metaInfo[bucket][workerEntryIndex + 2] = bColumnIndex;
          // Number is 1 less
          metaInfo[bucket][workerEntryIndex + 3] = aRowEndIndex - aRowIndex - 1;
          metaInfo[bucket][workerEntryIndex + 4] =
              outputEntryMetaInfoIndices[aRowIndex] - workerEntryIndex;
        }

        if (generateGradWMetaInfo) {
          const auto totalAElems = std::accumulate(
              rowColumnCounts.begin(), rowColumnCounts.end(), std::size_t(0));
          const auto numAElemsPerPartition =
              ceildiv(totalAElems, gradWNumAPartitions);
          unsigned currRowIndex = 0;
          unsigned currRowColumnIndex = 0;
          for (unsigned worker = 0; worker < gradWNumUsedWorkers; ++worker) {
            const auto sparseStartIndex = worker * numAElemsPerPartition;
            const auto sparseEndIndex =
                std::min((worker + 1) * numAElemsPerPartition, totalAElems);
            const auto numSparseElems = sparseEndIndex - sparseStartIndex;

            unsigned startRowIndex = currRowIndex;
            unsigned startRowStartColumnIndex = currRowColumnIndex;
            const auto workerEntryIndex =
                metaInfoGradWWorkerEntryIndices[worker];
            metaInfo[bucket][workerEntryIndex + 0] = sparseStartIndex;
            metaInfo[bucket][workerEntryIndex + 1] =
                outputEntryMetaInfoIndices[startRowIndex] - workerEntryIndex;
            metaInfo[bucket][workerEntryIndex + 2] = startRowStartColumnIndex;
            metaInfo[bucket][workerEntryIndex + 3] = numSparseElems;

            // Advance to next worker's work
            unsigned numRemainingElems = numSparseElems;
            while (numRemainingElems > 0) {
              const auto elemsThisRow =
                  std::min(numRemainingElems,
                           rowColumnCounts[currRowIndex] - currRowColumnIndex);
              numRemainingElems -= elemsThisRow;
              currRowColumnIndex += elemsThisRow;
              if (currRowColumnIndex >= rowColumnCounts[currRowIndex]) {
                currRowColumnIndex = 0;
                currRowIndex++;
              }
            }
          }
        }
        ++splitIdx;
      } else {
        const auto otherSubGroupIdx = i - splitIdx;
        const auto subGroupId = otherSubGroupIds[otherSubGroupIdx];
        const auto numElems = subGroupNumElems[bucket][i];
        metaInfo[bucket].emplace_back(subGroupId);
        metaInfo[bucket].emplace_back(numElems);
        // We also just use this no. of sub-elements as garbage in the meta-info
        // for the other (unprocessed) sub-groups.
        metaInfo[bucket].emplace_back(numElems + 3);
        for (std::size_t i = 0; i < numElems; ++i) {
          metaInfo[bucket].emplace_back(garbageDist(randomEngine));
        }
      }
    }
    constexpr unsigned endSubGroupId = 0;
    metaInfo[bucket].push_back(endSubGroupId);
  }

  return metaInfo;
}

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel;
  Type inputType = HALF;
  Type partialsType = FLOAT;
  ShapeOption<std::size_t> aShape;
  ShapeOption<std::size_t> bShape;
  double sparsityLevel = 0.1;
  unsigned numOtherSubGroups = 5;
  unsigned numOtherSubGroupElems = 30;
  unsigned numBuckets = 1;
  unsigned numSplitsPerBucket = 1;
  VertexType vertexType = VertexType::Forward;
  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type")
    ("profile", "Output profiling information for the program")
    ("ignore-data", "Don't validate outputs, don't add streams etc." 
     " Useful for profiling")
    ("show-execution-steps", "If profiling, show execution steps in the "
     "summary")
    ("show-var-storage", "If profiling, show variable liveness information in "
     " the summary")
    ("input-type",
     po::value<Type>(&inputType)->default_value(inputType),
     "Input type")
    ("partials-type",
     po::value<Type>(&partialsType)->default_value(partialsType),
     "Partials type")
    ("a-shape",
     po::value<ShapeOption<std::size_t>>(&aShape)->required(),
     "Shape of A ")
    ("b-shape",
     po::value<ShapeOption<std::size_t>>(&bShape)->required(),
     "Shape of B (columns must be multiples if 4 for half and multiples of 2 "
     "for float")
    ("sparsity-level",
     po::value<double>(&sparsityLevel)->default_value(sparsityLevel),
     "Level of sparsity of operand A")
    ("num-other-sub-groups",
     po::value<unsigned>(&numOtherSubGroups)->default_value(numOtherSubGroups),
     "Number of other (unprocessed) sub-groups to include in meta-info")
    ("num-other-sub-group-elements",
     po::value<unsigned>(&numOtherSubGroupElems)->
                default_value(numOtherSubGroupElems),
     "Number of elements in meta-info for other sub-groups (unprocessed)")
    ("num-buckets",
     po::value<unsigned>(&numBuckets)->default_value(numBuckets),
     "Number of buckets to generate and give to the codelet. Each bucket "
     "has same number other sub-groups, and same number of other sub-group "
     "elements. Number of non-zero elements to actually process is spread "
     "between buckets")
    ("num-splits-per-bucket",
     po::value<unsigned>(&numSplitsPerBucket)->default_value(numSplitsPerBucket),
     "How many times to split the processed sub-group in each bucket. It is "
     "valid to get the same sub-group multiple times in a bucket and this "
     "allows testing.")
    ("zero-partials", "Whether or not to zero partials. Default is to not "
     "do so")
    ("vertex-type",
     po::value<VertexType>(&vertexType)->default_value(vertexType),
     "Which vertex to test (Forward | GradA | Transposed | GradW)")
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

  bool profile = vm.count("profile");
  bool ignoreData = vm.count("ignore-data");
  bool showExecutionSteps = vm.count("show-execution-steps");
  bool showVarStorage = vm.count("show-var-storage");
  bool zeroPartials = vm.count("zero-partials");

  if (sparsityLevel <= 0 || sparsityLevel >= 1) {
    throw poplibs_error("sparsity-level must be in range (0, 1) but " +
                        std::to_string(sparsityLevel) + " was given");
  }

  if (aShape.val.size() != 2) {
    throw poplibs_error("shape of a must be 2-dimensional");
  }

  if (bShape.val.size() != 2) {
    throw poplibs_error("shape of b must be 2-dimensional");
  }

  if (aShape[1] != bShape[0]) {
    throw poplibs_error("size of inner dimension of a (" +
                        std::to_string(aShape[1]) +
                        ") must match outer dimension of b (" +
                        std::to_string(bShape[0]) + ")");
  }

  if (vertexType == VertexType::GradW && numBuckets != 1) {
    throw poplibs_error("GradW vertex can only handle --num-buckets=1");
  }

  std::size_t modForCheck = inputType == HALF ? 4 : 2;

  if (bShape[1] % modForCheck) {
    throw poplibs_error("sizes of second dimension of b must be multiple of " +
                        std::to_string(modForCheck));
  }

  const std::vector<std::size_t> cShape = {aShape[0], bShape[1]};

  const auto aNumElems = product(aShape.val);
  const auto aNumNonZeroElems =
      static_cast<std::size_t>(std::ceil(aNumElems * sparsityLevel));

  if (aNumNonZeroElems / numBuckets / numSplitsPerBucket == 0) {
    throw poplibs_error("Splitting " + std::to_string(aNumNonZeroElems) +
                        " into " + std::to_string(numBuckets) + " and " +
                        std::to_string(numSplitsPerBucket) +
                        " splits leaves no "
                        "elements in some sub-groups");
  }

  auto device = createTestDevice(deviceType, 1, 1);
  const auto &target = device.getTarget();
  Graph graph(target);
  popsparse::addCodelets(graph);

  std::mt19937 randomEngine;
  auto sparseIndices =
      generateSparseIndices(randomEngine, aShape.val, aNumNonZeroElems);

  unsigned processedSubGroupId;
  std::vector<unsigned> otherSubGroupIds;
  std::tie(processedSubGroupId, otherSubGroupIds) =
      generateSparseSubGroupIds(randomEngine, 1 + numOtherSubGroups, 1, 1000);
  std::vector<std::vector<unsigned>> processedSubGroupIndices;
  std::vector<std::vector<unsigned>> subGroupNumElems;
  std::tie(processedSubGroupIndices, subGroupNumElems) = partitionSubGroupElems(
      randomEngine, sparseIndices.size(), numBuckets, numSplitsPerBucket,
      numOtherSubGroups, numOtherSubGroupElems);
  const auto hostMetaInfoBuckets = generateMetaInfoAndPartition(
      randomEngine, sparseIndices, aShape.val, bShape.val, numBuckets,
      processedSubGroupId, otherSubGroupIds, processedSubGroupIndices,
      subGroupNumElems, target, inputType, partialsType, vertexType);

  // TODO: Check values in meta-info to ensure they are representable by this
  // type.
  const auto metaInfoType = UNSIGNED_SHORT;

  // Allocate operands
  const auto aType = vertexType == VertexType::GradW ? partialsType : inputType;
  const auto bType =
      vertexType == VertexType::Transposed ? partialsType : inputType;
  const auto cType =
      vertexType == VertexType::Forward || vertexType == VertexType::GradA
          ? partialsType
          : inputType;
  std::vector<Tensor> aBuckets(numBuckets);
  std::vector<Tensor> metaInfoBuckets(numBuckets);
  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    aBuckets[bucket] =
        graph.addVariable(aType, {sum(subGroupNumElems[bucket])},
                          "aNonZero (bucket " + std::to_string(bucket) + ")");
    metaInfoBuckets[bucket] =
        graph.addVariable(metaInfoType, {hostMetaInfoBuckets[bucket].size()},
                          "metaInfo (bucket " + std::to_string(bucket) + ")");
    graph.setTileMapping(aBuckets[bucket], 0);
    graph.setTileMapping(metaInfoBuckets[bucket], 0);
  }
  const auto b = graph.addVariable(bType, bShape.val, "b");
  const auto c = graph.addVariable(cType, cShape, "c");

  graph.setTileMapping(b, 0);
  graph.setTileMapping(c, 0);

  const auto cs = graph.addComputeSet("cs0");

  std::string vertexBaseClass = "popsparse::";
  switch (vertexType) {
  case VertexType::Forward:
    vertexBaseClass += "SparseDenseMatMulElementWise";
    break;
  case VertexType::Transposed:
    vertexBaseClass += "SparseDenseMatMulElementWiseTranspose";
    break;
  case VertexType::GradA:
    vertexBaseClass += "SparseDenseMatMulGradAElementWise";
    break;
  case VertexType::GradW:
    vertexBaseClass += "SparseDenseMatMulGradWElementWise";
    break;
  default:
    throw poplibs_error("Unrecognised vertex type");
  }
  const auto vertexClass =
      templateVertex(vertexBaseClass, inputType, partialsType);
  const auto v = graph.addVertex(cs, vertexClass);

  if (vertexType == VertexType::GradW) {
    graph.connect(v["qGrad"], c.flatten());
    graph.connect(v["rGrad"], aBuckets.at(0));
    graph.connect(v["s"], b.flatten());
    graph.connect(v["metaInfo"], metaInfoBuckets.at(0));
    const auto deviceProcessedSubGroupId =
        graph.addConstant(metaInfoType, {}, processedSubGroupId);
    graph.setTileMapping(deviceProcessedSubGroupId, 0);
    graph.connect(v["subGroupIdToProcess"], deviceProcessedSubGroupId);
    graph.setInitialValue(v["numZ"], bShape[1]);
    graph.setInitialValue(v["zeroInfo"],
                          zeroPartials ? aBuckets.at(0).numElements() : 0);
  } else {
    graph.connect(v["q"], c.flatten());
    graph.connect(v["r"], aBuckets);
    graph.connect(v["s"], b.flatten());
    graph.connect(v["metaInfo"], metaInfoBuckets);
    graph.setInitialValue(v["subGroupIdToProcess"], processedSubGroupId);
    const auto numPartials = vertexType == VertexType::Transposed
                                 ? b.numElements()
                                 : c.numElements();
    graph.setInitialValue(v["zeroInfo"], zeroPartials ? numPartials : 0);
  }
  graph.setTileMapping(v, 0);

  Sequence prog;
  prog.add(Execute(cs));

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  std::unique_ptr<char[]> rawHostB, rawHostC;
  std::vector<std::unique_ptr<char[]>> rawHostABuckets(numBuckets),
      rawHostMetaInfoBuckets(numBuckets);
  if (!ignoreData) {
    for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
      rawHostABuckets[bucket] = allocateHostMemoryForTensor(
          aBuckets[bucket], "a[" + std::to_string(bucket) + "]", graph,
          uploadProg, downloadProg, tmap);
      rawHostMetaInfoBuckets[bucket] = allocateHostMemoryForTensor(
          metaInfoBuckets[bucket], "metaInfo[" + std::to_string(bucket) + "]",
          graph, uploadProg, downloadProg, tmap);
    }
    rawHostB = allocateHostMemoryForTensor(b, "b", graph, uploadProg,
                                           downloadProg, tmap);
    rawHostC = allocateHostMemoryForTensor(c, "c", graph, uploadProg,
                                           downloadProg, tmap);
  }

  Engine engine(graph, Sequence(uploadProg, prog, downloadProg));
  attachStreams(engine, tmap);

  std::vector<boost::multi_array<double, 1>> hostABuckets;
  for (std::size_t bucket = 0; bucket < numBuckets; ++bucket) {
    hostABuckets.emplace_back(boost::extents[sum(subGroupNumElems[bucket])]);
  }
  boost::multi_array<double, 2> hostB(boost::extents[bShape[0]][bShape[1]]);
  boost::multi_array<double, 2> hostC(boost::extents[cShape[0]][cShape[1]]);

  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    writeRandomValues(target, aType, hostABuckets[bucket], -1.0, +1.0,
                      randomEngine);
    copy(target, hostABuckets[bucket], aType, rawHostABuckets[bucket].get());
  }
  writeRandomValues(target, bType, hostB, -1.0, +1.0, randomEngine);
  copy(target, hostB, bType, rawHostB.get());
  writeRandomValues(target, cType, hostC, -1.0, +1.0, randomEngine);
  copy(target, hostC, cType, rawHostC.get());
  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    copy(target, hostMetaInfoBuckets[bucket], metaInfoType,
         rawHostMetaInfoBuckets[bucket].get());
  }

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.setPrintStream(std::cerr);
    engine.run();
  });

  // Store initial values of operands before pulling partials from device
  // as these are needed to test partials zeroing (or not zeroing).
  // NOTE: Need deep copy hence type is explicit to avoid getting a view
  const boost::multi_array<double, 1> origA = hostABuckets.at(0);
  const boost::multi_array<double, 2> origB = hostB;
  const boost::multi_array<double, 2> origC = hostC;
  if (vertexType == VertexType::GradW) {
    copy(target, aType, rawHostABuckets.at(0).get(), hostABuckets.at(0));
  } else if (vertexType == VertexType::Transposed) {
    copy(target, bType, rawHostB.get(), hostB);
  } else if (vertexType == VertexType::Forward ||
             vertexType == VertexType::GradA) {
    copy(target, cType, rawHostC.get(), hostC);
  }

  if (!ignoreData) {
    // We use a dense matrix a to model this.
    boost::multi_array<double, 2> hostADense(
        boost::extents[aShape[0]][aShape[1]]);

    if (vertexType != VertexType::GradW) {
      std::size_t nzOffset = 0;
      for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
        for (const auto &idx : processedSubGroupIndices[bucket]) {
          assert(idx < subGroupNumElems[bucket].size());
          const auto bucketNzOffset = std::accumulate(
              subGroupNumElems[bucket].begin(),
              subGroupNumElems[bucket].begin() + idx, std::size_t(0));
          for (std::size_t i = 0; i < subGroupNumElems[bucket][idx]; ++i) {
            assert(bucketNzOffset + i < hostABuckets[bucket].num_elements());
            hostADense[sparseIndices.at(nzOffset + i)[0]]
                      [sparseIndices.at(nzOffset + i)[1]] =
                          hostABuckets[bucket][bucketNzOffset + i];
          }
          nzOffset += subGroupNumElems.at(bucket).at(idx);
        }
      }
    }
    bool matchesModel = true;
    double relativeTolerance =
        inputType == FLOAT ? FLOAT_REL_TOL : HALF_REL_TOL;
    double absoluteTolerance =
        inputType == FLOAT ? FLOAT_ABS_TOL : HALF_ABS_TOL;
    if (vertexType == VertexType::GradW) {
      boost::multi_array<double, 2> modelADense(
          boost::extents[aShape[0]][aShape[1]]);
      poplibs_test::gemm::generalMatrixMultiply(hostC, hostB, modelADense,
                                                false, true);

      // Now get the model sparse a, we do this to see if the actual sparse a
      // overwrote any of the other positions that weren't part of the
      // processed sub-group.
      boost::multi_array<double, 1> modelA(
          boost::extents[sum(subGroupNumElems.at(0))]);
      if (!zeroPartials) {
        for (std::size_t i = 0; i < modelA.num_elements(); ++i) {
          modelA.data()[i] = origA.data()[i];
        }
      }
      std::size_t nzOffset = 0;
      for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
        for (const auto &idx : processedSubGroupIndices[bucket]) {
          const auto bucketNzOffset = std::accumulate(
              subGroupNumElems[bucket].begin(),
              subGroupNumElems[bucket].begin() + idx, std::size_t(0));
          for (std::size_t i = 0; i < subGroupNumElems[bucket][idx]; ++i) {
            modelA[bucketNzOffset + i] +=
                modelADense[sparseIndices.at(nzOffset + i)[0]]
                           [sparseIndices.at(nzOffset + i)[1]];
          }
          nzOffset += subGroupNumElems[bucket][idx];
        }
      }

      matchesModel = checkIsClose("modelA", hostABuckets.at(0), modelA,
                                  relativeTolerance, absoluteTolerance);
    } else if (vertexType == VertexType::Transposed) {
      boost::multi_array<double, 2> modelB(
          boost::extents[bShape[0]][bShape[1]]);
      poplibs_test::gemm::generalMatrixMultiply(hostADense, hostC, modelB, true,
                                                false);
      if (!zeroPartials) {
        for (std::size_t i = 0; i < modelB.num_elements(); ++i) {
          modelB.data()[i] += origB.data()[i];
        }
      }
      matchesModel = checkIsClose("modelB", hostB, modelB, relativeTolerance,
                                  absoluteTolerance);
    } else if (vertexType == VertexType::Forward ||
               vertexType == VertexType::GradA) {
      boost::multi_array<double, 2> modelC(
          boost::extents[cShape[0]][cShape[1]]);
      poplibs_test::gemm::generalMatrixMultiply(hostADense, hostB, modelC,
                                                false, false);
      if (!zeroPartials) {
        for (std::size_t i = 0; i < modelC.num_elements(); ++i) {
          modelC.data()[i] += origC.data()[i];
        }
      }
      matchesModel = checkIsClose("modelC", hostC, modelC, relativeTolerance,
                                  absoluteTolerance);
    } else {
      throw poputil::poplibs_error("Unhandled vertex type");
    }

    if (!matchesModel) {
      std::cerr << "Validation failed\n";
      return 1;
    }
  }

  if (profile) {
    engine.printProfileSummary(
        std::cerr,
        {{"showExecutionSteps", (showExecutionSteps ? "true" : "false")},
         {"showVarStorage", (showVarStorage ? "true" : "false")}});
  }

  return 0;
} catch (const poplar::graph_memory_allocation_error &e) {
  std::cerr << e.what() << std::endl;

  // this exit code has been marked as a "skip" for ctest.
  return 77;
}
