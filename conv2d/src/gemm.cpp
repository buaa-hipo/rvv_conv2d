#include "conv2d_utils.hpp"

#ifdef USE_RISCV_VECTOR

inline void gemm_nn(int M, int N, int K, DATA_TYPE ALPHA, DATA_TYPE *A, int lda,
             DATA_TYPE *B, int ldb, DATA_TYPE *C, int ldc) {
  
  // printf(" in gemm\n");
  _MMR_f32 tmp_C;
  _MMR_f32 tmp_B;
  _MMR_f32 tmp_A;

  DATA_TYPE* c_ptr = C;
  DATA_TYPE* a_ptr = A;
  for (int i = 0; i < M; ++i) {
    DATA_TYPE* b_ptr = B;
    for (int k = 0; k < K; ++k) {
      DATA_TYPE A_PART = ALPHA * a_ptr[k];

      int gvl = vsetvl_e32m1(N);
      // puts("set vl");

      // tmp_A = _MM_SET_f32(A_PART, gvl);
      // puts("excuted _MM_SET_f32(APART, gvl)");
      for (int j = 0; j < N; j += gvl) {
        gvl = vsetvl_e32m1(N - j);

        // asm volatile(
        //   "mv   t0,   %[c_ptr]    \n\t"
        //   "mv   t1,   %[j]        \n\t"
        //   "add  t0,   t0,   t1   \n\t"
        //   // load tmp_C (v0)
        //   "vle.v  v0, (t0)        \n\t"
        //   "mv   t2,   %[b_ptr]    \n\t"
        //   "add  t2,   t2,   t1   \n\t"
        //   // load tmp_B (v1)
        //   "vle.v  v1, (t2)        \n\t"
        //   // tmp_B = tmp_B * tmp_A (v2)
        //   // "vdot.vv v1, v1, v2   \n\t"
        //   // tmp_C = tmp_A (v2) * tmp_B (v1) + tmp_C (v0)
        //   "vfmacc.vv v0, v1, v2   \n\t"
        //   // store tmp_C back
        //   "vsw.v  v0, (t0)  \n\t"
        //   : // output
        //   : [c_ptr]"r"(c_ptr), [j]"r"(j), [b_ptr]"r"(b_ptr) // input
        //   : "memory", "t0", "t1", "t2", "v0", "v1", "v2" // reserved list
        // ); 
        tmp_C = _MM_LOAD_f32(c_ptr + j, gvl);
        tmp_B = _MM_LOAD_f32(b_ptr + j, gvl);
        // puts("excuted _MM_LOAD_f32");
        // // C[i*ldc+j] += A_PART*B[k*ldb+j];
        // // tmp_B = _MM_MUL_f32(tmp_B, tmp_A, gvl);
        // tmp_C = _MM_MACC_f32(tmp_C, tmp_A, tmp_B, gvl);
        tmp_C = vfmacc_vf_f32m1(tmp_C, A_PART, tmp_B, gvl);
        // puts("excuted _MM_MACC_f32");

        _MM_STORE_f32(c_ptr + j, tmp_C, gvl);
        // puts("excuted _MM_STORE_f32");
      }
      b_ptr += ldb;
    }
    c_ptr += ldc;
    a_ptr += lda;
  }
  // puts("finished gemm");
  // FENCE();
}


// reference https://github.com/riscv/riscv-v-spec/blob/master/example/sgemm.S
// c += a*b (alpha=1, no transpose on input matrices)
// matrices stored in C row-major order
void sgemm_vec(size_t size_m, size_t size_n, size_t size_k,
                float alpha, // alpha = 1.0
                const float *a, // m * k matrix
                size_t lda,
                const float *b, // k * n matrix
                size_t ldb,
                float *c, // m * n matrix
                size_t ldc) {
  size_t vl;
  for (size_t m = 0; m < size_m; ++m) {
    const float *b_n_ptr = b;
    float *c_n_ptr = c;
    for (size_t c_n_count = size_n; c_n_count; c_n_count -= vl) {
      vl = vsetvl_e32m1(c_n_count );
      const float *a_k_ptr = a;
      const float *b_k_ptr = b_n_ptr;
      vfloat32m1_t acc = vle32_v_f32m1(c_n_ptr, vl);
      for (size_t k = 0; k < size_k; ++k) {
        vfloat32m1_t b_n_data = vle32_v_f32m1(b_k_ptr, vl);
        acc = vfmacc_vf_f32m1(acc, *a_k_ptr, b_n_data, vl);
        b_k_ptr += ldb;
        a_k_ptr++;
      }
      vse32_v_f32m1(c_n_ptr, acc, vl);
      c_n_ptr += vl;
      b_n_ptr += vl;
    }
    a += lda;
    c += ldc;
  } 
}

inline void gemm_nt(int M, int N, int K, DATA_TYPE ALPHA, DATA_TYPE *A, int lda,
             DATA_TYPE *B, int ldb, DATA_TYPE *C, int ldc) {
  int i, j, k;
#pragma omp parallel for
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      DATA_TYPE sum = 0;
      for (k = 0; k < K; ++k) {
        sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
      }
      C[i * ldc + j] += sum;
    }
  }
  // _MMR_f32 tmp_A;
  // _MMR_f32 tmp_B;
  // _MMR_f32 tmp_C;
  // int gvl = vsetvl_e32m1(K);
  // _MMR_f32 zeroV = _MM_SET_f32(0, gvl);
  // for (i = 0; i < M; ++i) {
  //   gvl = vsetvl_e32m1(K);
  //   for (k = 0; k < K; k += gvl) {
  //     gvl = vsetvl_e32m1(K - k);
  //     // DATA_TYPE sum = 0;
  //     tmp_A = _MM_LOAD_f32(A + k, gvl);
  //     DATA_TYPE* b_ptr = B + k;
  //     for (j = 0; j < N; ++j) {
  //       tmp_C = _MM_SET_f32(0, gvl);
  //       tmp_B = _MM_LOAD_f32(b_ptr, gvl);
  //       // C[i * ldc + j] += ALPHA * A[i * lda + k] * B[j * ldb + k];
  //       tmp_C = _MM_MACC_f32(tmp_C, tmp_A, tmp_B, gvl);
  //       tmp_C = _MM_REDUSUM_f32(tmp_C, tmp_C, zeroV, gvl);
  //       C[j] = _MM_VGETFIRST_f32(tmp_C);
  //       b_ptr += ldb;
  //     }
  //   }
  //   A += lda;
  //   C += ldc;
  // }

  // _MMR_f32 tmp_A;
  // _MMR_f32 tmp_B;
  // _MMR_f32 tmp_C;
  // int gvl = vsetvl_e32m1(K);
  // _MMR_f32 zeroV = _MM_SET_f32(0, gvl);
  // for (i = 0; i < M; ++i) {
  //   for (j = 0; j < N; ++j) {
  //     tmp_C = _MM_SET_f32(0, gvl);
  //     // DATA_TYPE sum = 0;
  //     gvl = vsetvl_e32m1(K);
  //     // DATA_TYPE* b_ptr = B + k;
  //     for (k = 0; k < K; k += gvl) {
  //       gvl = vsetvl_e32m1(K - k);
  //       tmp_A = _MM_LOAD_f32(A + k, gvl);
  //       tmp_B = _MM_LOAD_f32(B + k, gvl);
  //       // C[i * ldc + j] += ALPHA * A[i * lda + k] * B[j * ldb + k];
  //       tmp_C = _MM_MACC_f32(tmp_C, tmp_A, tmp_B, gvl);
  //       tmp_C = _MM_REDUSUM_f32(tmp_C, tmp_C, zeroV, gvl);
  //     }
  //     C[j] = _MM_VGETFIRST_f32(tmp_C);
  //     B += ldb;
  //   }
  //   A += lda;
  //   C += ldc;
  // }
}

inline void gemm_tn(int M, int N, int K, DATA_TYPE ALPHA, DATA_TYPE *A, int lda,
             DATA_TYPE *B, int ldb, DATA_TYPE *C, int ldc) {
  int i, j, k;
  // return ;
  _MMR_f32 tmp_B;
  _MMR_f32 tmp_C;
  // version 1
// #pragma omp parallel for
//   for (i = 0; i < M; ++i) {
//     DATA_TYPE* a_ptr = A + i;
//     int gvl = vsetvl_e32m1(N);
//     for (j = 0; j < N; j += gvl) {
//       gvl = vsetvl_e32m1(N - j);
//       // tmp_C = _MM_LOAD_f32(C + j, gvl);
//       tmp_C = _MM_SET_f32(0, gvl);  // C = A * B + 0 * C
//       DATA_TYPE* b_ptr = B + j;
//       for (k = 0; k < K; ++k) {
//         DATA_TYPE A_PART = ALPHA * (*a_ptr);
//         // C[j] += A_PART * B[k * ldb + j];
//         tmp_B = _MM_LOAD_f32(b_ptr, gvl);
//         tmp_C = vfmacc_vf_f32m1(tmp_C, A_PART, tmp_B, gvl);
//         b_ptr += ldb;
//         a_ptr += lda;
//       }
//       _MM_STORE_f32(C + j, tmp_C, gvl);
//     }
//     C += ldc;
//     // A += lda;
//   }
  
  // version 2
  for (i = 0; i < M; ++i) {
    DATA_TYPE* a_ptr = A + i;
    DATA_TYPE* b_ptr = B;
    int gvl = vsetvl_e32m1(N);
    for (k = 0; k < K; ++k) {
      
      DATA_TYPE A_PART = ALPHA * (*a_ptr);

      for (j = 0; j < N; j += gvl) {
        gvl = vsetvl_e32m1(N - j);
        tmp_C = _MM_LOAD_f32(C + j, gvl);
        // tmp_C = _MM_SET_f32(0, gvl);  // C = A * B + 0 * C
        tmp_B = _MM_LOAD_f32(b_ptr + j, gvl);
        // C[i * ldc + j] += A_PART * B[k * ldb + j];
        tmp_C = vfmacc_vf_f32m1(tmp_C, A_PART, tmp_B, gvl);

        _MM_STORE_f32(C + j, tmp_C, gvl);
        
      }
      
      a_ptr += lda;
      b_ptr += ldb;
    }
    C += ldc;
    // A += lda;
  }
}

void gemm_tt(int M, int N, int K, DATA_TYPE ALPHA, DATA_TYPE *A, int lda,
             DATA_TYPE *B, int ldb, DATA_TYPE *C, int ldc) {
  int i, j, k;
#pragma omp parallel for
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      DATA_TYPE sum = 0;
      for (k = 0; k < K; ++k) {
        sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
      }
      C[i * ldc + j] += sum;
    }
  }
}
#else

void gemm_nn(int M, int N, int K, DATA_TYPE ALPHA, DATA_TYPE *A, int lda,
             DATA_TYPE *B, int ldb, DATA_TYPE *C, int ldc) {
  int i, j, k;
#pragma omp parallel for
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      DATA_TYPE A_PART = ALPHA * A[i * lda + k];
      for (j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

void gemm_nt(int M, int N, int K, DATA_TYPE ALPHA, DATA_TYPE *A, int lda,
             DATA_TYPE *B, int ldb, DATA_TYPE *C, int ldc) {
  int i, j, k;
#pragma omp parallel for
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      DATA_TYPE sum = 0;
      for (k = 0; k < K; ++k) {
        sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
      }
      C[i * ldc + j] += sum;
    }
  }
}

void gemm_tn(int M, int N, int K, DATA_TYPE ALPHA, DATA_TYPE *A, int lda,
             DATA_TYPE *B, int ldb, DATA_TYPE *C, int ldc) {
  int i, j, k;
#pragma omp parallel for
  for (i = 0; i < M; ++i) {
    for (k = 0; k < K; ++k) {
      DATA_TYPE A_PART = ALPHA * A[k * lda + i];
      for (j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

void gemm_tt(int M, int N, int K, DATA_TYPE ALPHA, DATA_TYPE *A, int lda,
             DATA_TYPE *B, int ldb, DATA_TYPE *C, int ldc) {
  int i, j, k;
#pragma omp parallel for
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      DATA_TYPE sum = 0;
      for (k = 0; k < K; ++k) {
        sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
      }
      C[i * ldc + j] += sum;
    }
  }
}

#endif

void gemm(int TA, int TB, int M, int N, int K, DATA_TYPE ALPHA, DATA_TYPE *A,
          int lda, DATA_TYPE *B, int ldb, DATA_TYPE BETA, DATA_TYPE *C,
          int ldc) {
  // printf("%d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb,
  // BETA, ldc);
  if (BETA != 1.0) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] *= BETA;
      }
    }
  }
  if (!TA && !TB)
    gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    // sgemm_vec(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  else if (TA && !TB)
    gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  else if (!TA && TB)
    gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  else
    gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}
