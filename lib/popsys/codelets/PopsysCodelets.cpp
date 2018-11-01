#include <poplar/Vertex.hpp>
#include <poplar/HalfFloat.hpp>
#include <cassert>
#include <cmath>
#include <type_traits>
#include "poplibs_support/ExternalCodelet.hpp"

#define __IPU_ARCH_VERSION__ 0
#include <tilearch.h>

using namespace poplar;
static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;

namespace popsys {

#ifdef __IPU__
  template <unsigned CSR>
  class GetSupervisorCSR: public SupervisorVertex {
  public:
    Output<Vector<unsigned, ONE_PTR>> out;
    IS_EXTERNAL_CODELET(true);
    bool compute();
      // This codelet should not be compiled by C
  };
  template class GetSupervisorCSR<CSR_C_PC__INDEX>;
  template class GetSupervisorCSR<CSR_S_FP_ICTL__INDEX>;
  template class GetSupervisorCSR<CSR_S_SCOUNT_L__INDEX>;
  template class GetSupervisorCSR<CSR_C_DBG_DATA__INDEX>;


  template<int CSR>
  class GetWorkerCSR: public Vertex {
  public:
    Output<Vector<unsigned, ONE_PTR>> out;

    bool compute() {
      unsigned x = 0;
      x = __builtin_ipu_get(CSR);
      out[0] = x;
      return true;
    }
  };
  template class GetWorkerCSR<CSR_C_PC__INDEX>;
  template class GetWorkerCSR<CSR_C_DBG_DATA__INDEX>;


  template <unsigned CSR>
  class PutSupervisorCSR: public SupervisorVertex {
  public:
    unsigned setVal;
    IS_EXTERNAL_CODELET(true);
    bool compute();
      // This codelet should not be compiled by C
  };
  template class PutSupervisorCSR<CSR_C_PC__INDEX>;
  template class PutSupervisorCSR<CSR_S_FP_ICTL__INDEX>;
  template class PutSupervisorCSR<CSR_S_SCOUNT_L__INDEX>;
  template class PutSupervisorCSR<CSR_C_DBG_DATA__INDEX>;


  template<int CSR>
  class PutWorkerCSR: public Vertex {
  public:
    unsigned setVal;

    bool compute() {
     __builtin_ipu_put(setVal, CSR);
      return true;
    }
  };
  template class PutWorkerCSR<CSR_C_PC__INDEX>;
  template class PutWorkerCSR<CSR_C_DBG_DATA__INDEX>;


  template <unsigned CSR>
  class ModifySupervisorCSR: public SupervisorVertex {
  public:

    unsigned clearVal;
    unsigned setVal;
    IS_EXTERNAL_CODELET(true);
    bool compute();
      // This codelet should not be compiled by C
  };
  template class ModifySupervisorCSR<CSR_C_PC__INDEX>;
  template class ModifySupervisorCSR<CSR_S_FP_ICTL__INDEX>;
  template class ModifySupervisorCSR<CSR_S_SCOUNT_L__INDEX>;
  template class ModifySupervisorCSR<CSR_C_DBG_DATA__INDEX>;


  template<int CSR>
  class ModifyWorkerCSR: public Vertex {
  public:

    unsigned clearVal;
    unsigned setVal;

    bool compute() {
      int x = __builtin_ipu_get(CSR);
      x = (x & clearVal) | setVal;
      __builtin_ipu_put(x, CSR);
      return true;
    }
  };
  template class ModifyWorkerCSR<CSR_C_PC__INDEX>;
  template class ModifyWorkerCSR<CSR_C_DBG_DATA__INDEX>;

  class TimeItStart: public SupervisorVertex {
  public:
    Output<Vector<unsigned, ONE_PTR>> out;

    IS_EXTERNAL_CODELET(true);
    bool compute();
      // This codelet should not be compiled by C
  };

  class TimeItEnd: public SupervisorVertex {
  public:
    Output<Vector<unsigned, ONE_PTR>> out;
    Input<Vector<unsigned, ONE_PTR>> startCount;

    IS_EXTERNAL_CODELET(true);
    bool compute();
      // This codelet should not be compiled by C
  };
#endif // __IPU__

} // end namespace popsys
