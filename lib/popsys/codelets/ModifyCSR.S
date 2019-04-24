#ifdef __IPU__

#include "poplibs_support/TileConstants.hpp"

#define CLEARVAL_OFF 0
#define SETVAL_OFF 1
#define MASK    m2
#define VALUE   m1

#define MANGLED_NAME __runCodelet_popsys__ModifySupervisorCSR___

//**************************************************************
// Define a macro to generate the code, including labels and
// section names
//**************************************************************
	.macro MODIFY_CSR CSR_NUMBER

.section .text.MANGLED_NAME\CSR_NUMBER, "ax"
.align 4
.globl MANGLED_NAME\CSR_NUMBER
.type MANGLED_NAME\CSR_NUMBER, @function
MANGLED_NAME\CSR_NUMBER:
  ld32     $MASK, $m0, $mzero, CLEARVAL_OFF
  get 	   $VALUE, \CSR_NUMBER
  and	   $VALUE, $VALUE, $MASK
  ld32     $MASK, $m0, $mzero, SETVAL_OFF
  or	   $VALUE, $VALUE, $MASK
  put      \CSR_NUMBER, $VALUE
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
		MODIFY_CSR %reg_number
	 	reg_number=reg_number+1
	.endr
#endif
