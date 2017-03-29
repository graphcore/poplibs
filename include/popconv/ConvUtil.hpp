#ifndef _popconv_ConvUtil_hpp_
#define _popconv_ConvUtil_hpp_
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
getInputIndex(unsigned outputIndex, unsigned stride, unsigned kernelSize,
              unsigned padding, unsigned inputSize, unsigned kernelIndex,
              bool isFractional);

/// Given an output range, return the subset whose calculation
/// involves the specified kernel.
std::pair<unsigned, unsigned>
getOutputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
               unsigned padding, unsigned kernelSize, unsigned inputSize,
               unsigned kernelIndex, bool isFractional);

/// Given an output range, return the subset whose calculation
/// involves the specified range of kernel indicies.
std::pair<unsigned, unsigned>
getOutputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
               unsigned kernelSize, unsigned padding, unsigned inputSize,
               std::pair<unsigned, unsigned> kernelIndexRange,
               bool isFractional);

/// Return the input range that is associated with
/// the specified kernel index when calculating the specified output range.
std::pair<unsigned, unsigned>
getInputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
              unsigned kernelSize, unsigned padding,
              unsigned inputSize, unsigned kernelIndex,
              bool isFractional);

/// Return the input range that is associated with the specified kernel index
/// range when calculating the specified output range.
std::pair<unsigned, unsigned>
getInputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
              unsigned kernelSize, unsigned padding,
              unsigned inputSize,
              std::pair<unsigned, unsigned> kernelIndexRange,
              bool isFractional);

inline std::pair<unsigned, unsigned>
getInputRange(unsigned outputIndex, unsigned stride,
              unsigned kernelSize, unsigned padding,
              unsigned inputSize,
              std::pair<unsigned, unsigned> kernelIndexRange,
              bool isFractional) {
  return getInputRange({outputIndex, outputIndex + 1}, stride, kernelSize,
                       padding, inputSize, kernelIndexRange, isFractional);
}

inline std::pair<unsigned, unsigned>
getInputRange(unsigned outputIndex, unsigned stride, unsigned kernelSize,
              unsigned padding, unsigned inputSize, bool isFractional) {
  return getInputRange(outputIndex, stride, kernelSize, padding,
                       inputSize, {0, kernelSize}, isFractional);
}

inline std::pair<unsigned, unsigned>
getInputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
              unsigned kernelSize, unsigned padding, unsigned inputSize,
              bool isFractional) {
  return getInputRange(outputRange, stride, kernelSize, padding,
                       inputSize, {0, kernelSize}, isFractional);
}

std::pair<unsigned, unsigned>
getKernelRange(unsigned outputIndex, unsigned stride, unsigned kernelSize,
               unsigned padding, unsigned inputSize,
               bool isFractional);


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
partitionConvPartialByWorker(unsigned numConvolutions, unsigned convSize,
                             unsigned numContexts, unsigned stride);

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSizeY,
             unsigned kernelSizeX,
             unsigned strideY, unsigned strideX, unsigned paddingY,
             unsigned paddingX);

}
#endif // _popconv_ConvUtil_hpp_
