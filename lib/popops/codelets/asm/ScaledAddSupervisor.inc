#ifndef __SCALED_ADD_SUPERVISOR_INC__
#define __SCALED_ADD_SUPERVISOR_INC__

// Definitions for the inputs and outputs of the VERTEX(supervisor) section
// of code that is common to all the supervisor variants (ScaledAdd,
// ScaledSubtract, aXplusbY, aXMinusbY, etc.)


// The VERTEX(supervisor) code will create a new vertex state on the supervisor
// stack that has the input values pre-processed for all of the workers to use.
// All values are stored as a full word for ease of access by the worker.
// All offsets are in bytes.
#define SV_STATE_DATA_OFFSET      0 // X ('A') data ptr

#define SV_STATE_COUNT_OFFSET     4 // All workers do at least 'count' elements.

#define SV_STATE_REMM1_OFFSET     8 // Workers [0..remM1] will do count+2 elems

#define SV_STATE_FINAL_OFFSET    12 // Last worker adds this many elems (0 or 1)

// Contains scaleA offset only (HALF or FLOAT), or both scaleA & scaleB for the
// aX+bY vertex, when the scale values are HALF
#define SV_STATE_SCALES_OFFSET   16

#define SV_STATE_DATA_B_OFFSET   20   // Y ('B') data ptr

// 0 or 1; if "1", 'data A' and  'data B' are in different memory elements.
// Not used in the 'float_half' variant.
#define SV_STATE_MEM_CONSTRAINTS 24

// Contains the scaleB value, only for the 'mixed' vertex, where scale values
// are FLOAT and tensor operands are HALF. Note that VERTEX(supervisor) code
// will *NOT* populate this one word in the vertex state (needs to be done
// before calling VERTEX(supervisor))
#define SV_STATE_SCALEB_OFFSET   28

#define SV_STATE_SIZE            32 // Total size of the vertex state


// Total space required on the stack. Max of SV_STATE_SIZE and
// CHECK_STATE_SIZE (used inScaledAddaXPlusbY_mixed.S)
#define STACK_SIZE (SV_STATE_SIZE)




// These 6 registers are the input parameters of VERTEX(supervisor), which is
// called from several source files.
// See comment in the definition of VERTEX(supervisor) for their meaning.
#define sv_count          m1
#define sv_log2AtomSize   m7
#define sv_atomSizeMask   m8
#define sv_workerFunction m6
#define sv_memConstraints m2
#define sv_scale          m4


// Flag for memConstraints
#define MEM_CONSTRAINTS_MASK 0x1


// Values to use for 'sv_log2AtomSize'
// This is to indicate how many elements (2 or 4) at a time are processed (an
// 'atom') to best exploit the load/store instructions.
#define LOG2_FLOAT_ATOM_SIZE 1
#define LOG2_HALF_ATOM_SIZE  2


// Bit to force all accumulators to zero when written in FP_CLR register
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

#endif
