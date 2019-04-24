#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"

#define PTR_OFF 0
#define PTR     m0
#define VALUE   m1

#define MANGLED_NAME __runCodelet_popsys__GetSupervisorCSR___
//**************************************************************
// Define a macro to generate the code, including labels and
// section names
//**************************************************************
	.macro GET_CSR CSR_NUMBER

.section .text.MANGLED_NAME\CSR_NUMBER, "ax"
.align 4
.globl MANGLED_NAME\CSR_NUMBER
.type MANGLED_NAME\CSR_NUMBER, @function
MANGLED_NAME\CSR_NUMBER:
  ld32     $PTR, $m0, $mzero, PTR_OFF/4
  get      $VALUE, \CSR_NUMBER
  st32     $VALUE, $PTR, $mzero, 0
  br       $lr
.size MANGLED_NAME\CSR_NUMBER, .-MANGLED_NAME\CSR_NUMBER

	.endmacro

//**************************************************************
// Now the macro can be used to create functions for some or
// all of the CSR registers
//**************************************************************

//alternate macros allow use of % to evaluate an expression
	.altmacro

	reg_number=0
	.rept 256
		GET_CSR %reg_number
	 	reg_number=reg_number+1
	.endr

#endif
