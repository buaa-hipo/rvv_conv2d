
#ifndef __CONV2D_UTILS_HPP__
#define __CONV2D_UTILS_HPP__
#endif

// #define USE_RISCV_VECTOR 1
#define USE_WINOGRAD 1
#ifdef USE_RISCV_VECTOR
#include "../../common/vector_defines.h"

//#define vsetvl_e32m1(vl) __builtin_epi_vsetvl((vl), __epi_e64, __epi_m1)

#endif

// #define USE_IM2COL 1
#define DATA_TYPE float
