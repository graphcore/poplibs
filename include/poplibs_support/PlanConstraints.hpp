// Copyright (c) 2019, Graphcore Ltd, All rights reserved.
#ifndef poplibs_support_PlanConstraints_hpp
#define poplibs_support_PlanConstraints_hpp

#include "poplibs_support/OptionParsing.hpp"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace poplibs_support {

// Wraps ptree only in order to add custom comparison operators.
class PlanConstraints : public boost::property_tree::ptree {
  using BaseTreeType = boost::property_tree::ptree;

public:
  PlanConstraints() = default;
  PlanConstraints(BaseTreeType t) : BaseTreeType(std::move(t)) {}
  PlanConstraints &operator=(BaseTreeType t) {
    static_cast<BaseTreeType &>(*this) = std::move(t);
    return *this;
  }
};

bool operator<(const PlanConstraints &a, const PlanConstraints &b);

// Make an option handler that will parse PlanConstraints
template <typename Validator>
poplibs::OptionHandler
makePlanConstraintsOptionHandler(PlanConstraints &output) {
  return poplibs::OptionHandler{[&output](const std::string &value) {
    if (!value.empty()) {
      std::stringstream ss(value);
      boost::property_tree::ptree t;
      boost::property_tree::json_parser::read_json(ss, t);

      // Validate the format using the provided functor.
      Validator{}(t);

      output = std::move(t);
    } else {
      output.clear();
    }
  }};
}

// Generic input validation methods. Throws on error.
void validatePlanConstraintsBoolean(const std::string &,
                                    const boost::property_tree::ptree &);
void validatePlanConstraintsUnsigned(const std::string &,
                                     const boost::property_tree::ptree &);
void validatePlanConstraintsUnsignedArray(const std::string &,
                                          const boost::property_tree::ptree &);

} // namespace poplibs_support

#endif // poplibs_support_PlanConstraints_hpp
