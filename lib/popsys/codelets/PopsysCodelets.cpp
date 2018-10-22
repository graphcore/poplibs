#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cassert>
#include <cmath>
#include <type_traits>
#include "poplibs_support/ExternalCodelet.hpp"

#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace popsys {

#ifdef __IPU__
  template <unsigned CSR>
  class GetSupervisorCSR: public SupervisorVertex {
  public:
    Output<Vector<unsigned, ONE_PTR>> out;
    IS_EXTERNAL_CODELET(true);
    bool compute() {
      // This codelet should not be compiled by C
      assert(false);
      return true;
    }
  };

  template<int CSR>
  class GetWorkerCSR: public Vertex {
  public:
    Output<Vector<unsigned, ONE_PTR>> out;

    bool compute() {
      unsigned x = 0;
      // This currently does nothing but write 0 to the output
      // Pending D5365 it should use the function
      // __builtin_colossus_get_worker_csr(CSR);
      out[0] = x;
      return true;
    }
  };

  template class GetSupervisorCSR<0>; // PC
  template class GetSupervisorCSR<96>; // cycle count lower
  template class GetWorkerCSR<0>; // PC

  class TimeItStart: public SupervisorVertex {
  public:
    Output<Vector<unsigned, ONE_PTR>> out;

    //static const bool isExternalCodelet = true;
    IS_EXTERNAL_CODELET(true);
    bool compute() {
      // This codelet should not be compiled by C
      assert(false);
      return true;
    }
  };

  class TimeItEnd: public SupervisorVertex {
  public:
    Output<Vector<unsigned, ONE_PTR>> out;
    Input<Vector<unsigned, ONE_PTR>> startCount;

    IS_EXTERNAL_CODELET(true);
    bool compute() {
      // This codelet should not be compiled by C
      assert(false);
      return true;
    }
  };
#endif // __IPU__

} // end namespace popsys
