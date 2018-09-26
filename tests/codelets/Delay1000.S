#ifdef __IPU__

#define NAME __runCodelet_Delay1000

.globl NAME
.type NAME, @function
.section .text.NAME
.align 4
NAME:
  delay 1000
  br $lr

.size NAME, .-NAME

#endif
