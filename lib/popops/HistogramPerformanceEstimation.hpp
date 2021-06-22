// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef _popops_HistogramPerformanceEstimation_hpp_
#define _popops_HistogramPerformanceEstimation_hpp_

#include <cstdint>

std::uint64_t histogram1DByLimitEstimate(
    unsigned elements, unsigned histogramCount, bool isAbsolute, bool isHalf,
    unsigned numWorkers, unsigned vectorWidth, unsigned unpackCostHistogram = 0,
    unsigned unpackCostLimits = 0);

std::uint64_t histogram1DByDataEstimate(
    unsigned elements, unsigned histogramCount, bool isAbsolute, bool isHalf,
    unsigned numWorkers, unsigned vectorWidth, unsigned unpackCostHistogram = 0,
    unsigned unpackCostLimits = 0);

#endif // __popops_HistogramPerformanceEstimation_hpp__
