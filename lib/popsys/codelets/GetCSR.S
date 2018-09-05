#ifdef __IPU__

#include "tilearch.h"
#include "tileimplconsts_tommy.h"

#define PTR_OFF 0
#define PTR     m0
#define VALUE   m1
#define CSR @CSR@
#define MANGLED_NAME(csr) MANGLED_NAME_(csr)
#define MANGLED_NAME_(csr) __runCodelet_popsys__GetSupervisorCSR___##csr

.section .text.getSupervisorCSR_CSR, "ax"
.align 4
.globl MANGLED_NAME(CSR)
.type MANGLED_NAME(CSR), @function
MANGLED_NAME(CSR):
  ld32     $PTR, $m0, $mzero, PTR_OFF/4
  get      $VALUE, CSR
  st32     $VALUE, $PTR, $mzero, 0
  br       $lr

#endif
