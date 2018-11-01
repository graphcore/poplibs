#ifdef __IPU__

#include "tilearch.h"
#include "tileimplconsts.h"

#define VALUE_OFF 0

#define VALUE   m1

#define MANGLED_NAME __runCodelet_popsys__PutSupervisorCSR___
//**************************************************************
// Define a macro to generate the code, including labels and
// section names
//**************************************************************
	.macro PUT_CSR CSR_NUMBER

.section .text.MANGLED_NAME\CSR_NUMBER, "ax"
.align 4
.globl MANGLED_NAME\CSR_NUMBER
.type MANGLED_NAME\CSR_NUMBER, @function
MANGLED_NAME\CSR_NUMBER:
  ld32     $VALUE, $m0, $mzero, VALUE_OFF/4
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
		PUT_CSR %reg_number
	 	reg_number=reg_number+1
	.endr
#endif
