// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popsolver_PriorityGroupID_hpp
#define popsolver_PriorityGroupID_hpp

class PriorityGroupID {
public:
  using underlying_type = unsigned;
  underlying_type id;
  explicit PriorityGroupID(underlying_type id) : id(id) {}
  PriorityGroupID() = default;
  PriorityGroupID(const PriorityGroupID &other) = default;
  PriorityGroupID &operator=(const PriorityGroupID &other) = default;
  operator underlying_type() const { return id; }
};

#endif // popsolver_PriorityGroupID_hpp
