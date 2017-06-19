#ifndef _popconv_ConvUtil_hpp_
#define _popconv_ConvUtil_hpp_
#include <popconv/Convolution.hpp>
#include <tuple>
#include <vector>

/// A collection of utility functions to assist calculation of input/output
/// ranges when moving a 2-dimensional kernel over a larger 2-dimensional
/// space (e.g. in convolution or pooling layers

namespace popconv {

inline unsigned absdiff(unsigned a, unsigned b) {
  return a < b ? b - a : a - b;
}

/// Return the index of the input that is associated with the specified
/// kernel index to be incorporated into the specified output.
/// Return ~0U if there is no
/// such output.
unsigned
getInputIndex(unsigned dim, unsigned outputIndex, unsigned kernelIndex,
              const ConvParams &params);

/// Given an output range, return the subset whose calculation
/// involves the specified kernel.
std::pair<unsigned, unsigned>
getOutputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
               unsigned kernelIndex, const ConvParams &params);

/// Given an output range, return the subset whose calculation
/// involves the specified range of kernel indicies.
std::pair<unsigned, unsigned>
getOutputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
               std::pair<unsigned, unsigned> kernelIndexRange,
               const ConvParams &params);

/// Return the input range that is associated with
/// the specified kernel index when calculating the specified output range.
std::pair<unsigned, unsigned>
getInputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
              unsigned kernelIndex, const ConvParams &params);

/// Return the input range that is associated with the specified kernel index
/// range when calculating the specified output range.
std::pair<unsigned, unsigned>
getInputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
              std::pair<unsigned, unsigned> kernelIndexRange,
              const ConvParams &params);

inline std::pair<unsigned, unsigned>
getInputRange(unsigned dim, unsigned outputIndex,
              std::pair<unsigned, unsigned> kernelIndexRange,
              const ConvParams &params) {
  return getInputRange(dim, {outputIndex, outputIndex + 1},
                       kernelIndexRange, params);
}

inline std::pair<unsigned, unsigned>
getInputRange(unsigned dim, unsigned outputIndex, const ConvParams &params) {
  return getInputRange(dim, outputIndex, {0, params.kernelShape[dim]},
                       params);
}

inline std::pair<unsigned, unsigned>
getInputRange(unsigned dim, std::pair<unsigned, unsigned> outputRange,
              const ConvParams &params) {
  return getInputRange(dim, outputRange, {0, params.kernelShape[dim]},
                       params);
}

struct PartialRow {
  unsigned rowNumber;
  unsigned begin;
  unsigned end;
  PartialRow(unsigned rowNumber, unsigned begin, unsigned end) :
    rowNumber(rowNumber),
    begin(begin),
    end(end) {}
};

std::vector<std::vector<PartialRow>>
partitionConvPartialByWorker(unsigned convHeight, unsigned convWidth,
                             unsigned numContexts,
                             const std::vector<unsigned> &inputDilation);

std::vector<std::size_t> getOutputShape(const ConvParams &params);

// Given a set of parameters, return the set of params that
// represent the convolution to be applied to the output gradients
// to get the input gradients (provided the weights have been
// transposed in the channel axes and flipped in the spatial axes).
ConvParams getGradientParams(const ConvParams &params);

unsigned detectChannelGrouping(const poplar::Tensor &t);

}
#endif // _popconv_ConvUtil_hpp_
