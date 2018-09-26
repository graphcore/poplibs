#include <poplar/Vertex.hpp>

using namespace poplar;

class Delay1000 : public SupervisorVertex {
public:
  static const bool isExternalCodelet = true;
  bool compute() { return true; }
};
