#ifndef _ConvUtil_hpp_
#define _ConvUtil_hpp_
#include <tuple>

/// A collection of utility functions to assist calculation of input/output
/// ranges when moving a 2-dimensional kernel over a larger 2-dimensional
/// space (e.g. in convolution or pooling layers

/// Return the index of the input that is associated with the specified
/// kernel index to be incorporated into the specified output.
/// Return ~0U if there is no
/// such output.
unsigned
getInputIndex(unsigned outputIndex, unsigned stride, unsigned kernelSize,
              unsigned padding, unsigned inputSize, unsigned kernelIndex);

/// Given an output range, return the subset whose calculation involves the
/// specified kernel.
std::pair<unsigned, unsigned>
getOutputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
               unsigned padding, unsigned kernelSize, unsigned inputSize,
               unsigned kernelIndex);

/// Return the input range that is associate with the specified kernel index
/// when calculating the specified output range.
std::pair<unsigned, unsigned>
getInputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
              unsigned kernelSize, unsigned padding,
              unsigned inputSize, unsigned kernelIndex);

std::pair<unsigned, unsigned>
getInputRange(unsigned outputIndex, unsigned stride, unsigned kernelSize,
              unsigned padding, unsigned inputSize);

std::pair<unsigned, unsigned>
getInputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
              unsigned kernelSize, unsigned padding, unsigned inputSize);

std::pair<unsigned, unsigned>
getKernelRange(unsigned outputIndex, unsigned stride, unsigned kernelSize,
               unsigned padding, unsigned inputSize);

std::pair<unsigned, unsigned>
getKernelRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
               unsigned kernelSize, unsigned padding, unsigned inputSize);

#endif // _ConvUtil_hpp_
