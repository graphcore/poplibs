// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef POPFLOAT_HALF_UTILS_H
#define POPFLOAT_HALF_UTILS_H

namespace popfloat {
namespace experimental {

/** Cast a single precision input to half precision
 *
 * \param value          Single precision input
 * \param enNanoo        Enable Nan on overflow
 * \return               The 16-bit representation of the half precision output
 */
uint16_t singleToHalf(float value, bool enNanoo = false);

/** Cast a half precision input to single precision
 *
 * \param ihalf          The 16-bit representation of the half precision input
 * \return               Single precision output
 */
float halfToSingle(uint16_t ihalf);

} // end namespace experimental
} // end namespace popfloat

#endif
