#ifndef popsolver_Variable_hpp
#define popsolver_Variable_hpp

namespace popsolver {

class Variable {
public:
  Variable() = default;
  explicit Variable(unsigned id) : id(id) {}
  unsigned id;
};

} // End namespace popsolver.

#endif // popsolver_Variable_hpp
