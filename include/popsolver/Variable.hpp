#ifndef _popsolver_Variable_hpp_
#define _popsolver_Variable_hpp_

namespace popsolver {

class Variable {
public:
  Variable() = default;
  explicit Variable(unsigned id) : id(id) {}
  unsigned id;
};

} // End namespace popsolver.

#endif // _popsolver_Variable_hpp_
