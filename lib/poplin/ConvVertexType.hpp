// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef _ConvVertexType_hpp_
#define _ConvVertexType_hpp_

#include "ConvPlan.hpp"
#include <iosfwd>
#include <poplar/Type.hpp>

namespace poplar {
class Target;
}

namespace poplin {

struct ConvParams;
class ConvOptions;

struct ConvVertexType {
  Plan::Method method;
  poplar::Type inputType;
  poplar::Type partialType;

  unsigned convGroupsPerGroup;
  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;

  ConvVertexType(Plan::Method method, poplar::Type inputType,
                 poplar::Type partialType, unsigned convGroupsPerGroup,
                 unsigned inChansPerGroup, unsigned partialChansPerGroup)
      : method(method), inputType(inputType), partialType(partialType),
        convGroupsPerGroup(convGroupsPerGroup),
        inChansPerGroup(inChansPerGroup),
        partialChansPerGroup(partialChansPerGroup) {}
};

bool operator<(const ConvVertexType &a, const ConvVertexType &b);
bool operator==(const ConvVertexType &a, const ConvVertexType &b);

std::ostream &operator<<(std::ostream &os, const ConvVertexType &cvt);

std::vector<ConvVertexType>
getConvVertexTypeCandidates(const poplar::Target &target,
                            poplar::Type inputType, poplar::Type outputType,
                            poplar::Type partialType, const ConvParams &params,
                            const ConvOptions &options, bool isJointPlan);

} // End namespace poplin

#endif // _ConvVertexType_hpp_
