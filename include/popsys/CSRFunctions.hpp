// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popsys_CSRFunctions_hpp
#define popsys_CSRFunctions_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popsys {

/** Structure to specify floating point behaviour.
 *
 * \param inv           If true a floating point invalid operation (defined by
 *                      IEEE 754) will cause an exception.
 *                      Invalid operations:
 *                      - Addition or subtraction where one or both operands
 *                        are + or - infinity (inf).
 *                      - Divisions: (+/-0)/(+/-0) and (+/-inf)/(+/-inf).
 *                      - Multiplications: (+/-0)*(+/-inf) and (+/-inf)*(+/-0).
 *                      - Remainder: x REM y where y=0 or x=(+/-inf)
 *                      - Real operations with complex results such as the
 *                        square root or logarithm of a negative number.
 *                      - Operations with Not-a-Number as at least one operand.
 *                      - Comparisons where one of the operands is Not-a-Number.
 *                      See also `nanoo` below
 * \param div           If true a floating point divide by zero operation will
 *                      cause an exception
 * \param oflo          If true a floating point overflow will cause an
 *                      exception
 * \param esr           Enable stochastic rounding
 * \param nanoo         Enable Not-a-Number on overflow mode.  When enabled half
 *                      precision calculations that have overflowed will
 *                      produce a Not-a-Number result, rather than
 *                      saturating to the half precision max/min value, and the
 *                      invalid operation (`inv`) flag will be set
 *
 */

struct FloatingPointBehaviour {
  bool inv = true;
  bool div0 = true;
  bool oflo = true;
  bool esr = true;
  bool nanoo = true;
  FloatingPointBehaviour(bool inv, bool div0, bool oflo, bool esr, bool nanoo)
    : inv(inv), div0(div0), oflo(oflo), esr(esr), nanoo(nanoo) {}
  FloatingPointBehaviour() = default;
};

/** Set the floating point behaviour of a tile.
 *
 * Configures the floating point behaviour of a tile, affecting the treatment
 * of exceptions and selecting stochastic rounding according to the passed
 * \a behaviour structure.
 *
 * \param graph         The poplar graph
 * \param prog          The program to be extended
 * \param behaviour     A structure of type floatingPointBehaviour
 * \param debugPrefix   The prefix prepended to debugging info
 */

void setFloatingPointBehaviour(poplar::Graph &graph,
                                  poplar::program::Sequence &prog,
                                  const FloatingPointBehaviour &behaviour,
                                  const std::string &debugPrefix = "");

/** Set stochastic rounding on or off for the selected tile.
 *
 * Configures the stochastic rounding operation of a tile according to the
 * passed \a behaviour parameter.
 *
 * \param graph         The poplar graph
 * \param prog          The program to be extended
 * \param behaviour     Select stochastic rounding: true or false
 * \param debugPrefix   The prefix prepended to debugging info
 */
void setStochasticRounding(poplar::Graph &graph,
                                poplar::program::Sequence &prog,
                                bool behaviour,
                                const std::string &debugPrefix = "");

}
#endif // popsys_CSRFunctions_hpp
