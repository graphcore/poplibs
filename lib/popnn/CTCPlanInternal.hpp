// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popnn_CTCPlanInternal_hpp
#define popnn_CTCPlanInternal_hpp

#include "popnn/CTCPlan.hpp"

#include "CTCInferencePlan.hpp"
#include "CTCLossPlan.hpp"

#include <boost/variant.hpp>

#include <iosfwd>

namespace popnn {
namespace ctc {

class Plan::Impl {
public:
  boost::variant<LossPlan, InferencePlan> plan;

  const LossPlan &getAsLossPlan() const;
  std::unique_ptr<Plan::Impl> clone() const;
};

bool operator<(const Plan::Impl &a, const Plan::Impl &b);
bool operator==(const Plan::Impl &a, const Plan::Impl &b);

std::ostream &operator<<(std::ostream &o, const Plan::Impl &p);

} // namespace ctc
} // namespace popnn

#endif // #ifndef popnn_CTCPlanInternal_hpp
