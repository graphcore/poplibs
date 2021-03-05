// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Support for planning Connectionist Temporal Classification (CTC) Operations.
 *
 */

#ifndef popnn_CTCPlan_hpp
#define popnn_CTCPlan_hpp

#include <poputil/DebugInfo.hpp>

namespace popnn {
namespace ctc {

/** An object representing a plan that describes how to map tensors and
 *  implement the CTC Loss or CTC Inference functions.
 */
class Plan {
public:
  Plan();
  ~Plan();
  Plan(const Plan &other);
  Plan(Plan &&other);
  Plan &operator=(const Plan &other);
  Plan &operator=(Plan &&other);

  friend bool operator<(const Plan &a, const Plan &b);
  friend bool operator==(const Plan &a, const Plan &b);

  friend std::ostream &operator<<(std::ostream &o, const Plan &p);
  friend poplar::ProfileValue poputil::toProfileValue<>(const Plan &p);

  // Internal implementation
  class Impl;
  Impl &getImpl() const { return *impl; }
  Plan(std::unique_ptr<Impl> impl);

private:
  std::unique_ptr<Impl> impl;
};

bool operator<(const Plan &a, const Plan &b);
bool operator==(const Plan &a, const Plan &b);
bool operator!=(const Plan &a, const Plan &b);

} // namespace ctc
} // namespace popnn

#endif // popnn_CTCPlan_hpp
