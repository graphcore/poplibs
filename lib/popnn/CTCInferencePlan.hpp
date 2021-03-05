// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popnn_CTCInferencePlan_hpp
#define popnn_CTCInferencePlan_hpp

#include <popnn/CTCInference.hpp>

namespace popnn {
namespace ctc {

struct CtcInferencePlannerParams {};

struct InferencePlan {
  CtcInferencePlannerParams params;
};

bool operator<(const InferencePlan &a, const InferencePlan &b) noexcept;
bool operator==(const InferencePlan &a, const InferencePlan &b) noexcept;

std::ostream &operator<<(std::ostream &o, const InferencePlan &p);

} // namespace ctc
} // namespace popnn

#endif // #ifndef popnn_CTCInferencePlan_hpp
