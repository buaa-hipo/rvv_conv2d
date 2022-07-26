/**
 *
 */
/* conv2d.cpp */

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <time.h>

#include "conv2d_utils.hpp"
#include "gemm.cpp"
#include "im2col.cpp"
/************************************************************************/

/************************************************************************/

using namespace std;
#define RESULT_PRINT

#define INDEX_3D(i, j, k, N1, N2, N3) ((i) * (N2) * (N3) + (j) * (N3) + (k))
#define INDEX_4D(i, j, k, l, N1, N2, N3, N4)                                   \
  ((i) * (N2) * (N3) * (N4) + (j) * (N3) * (N4) + (k) * (N4) + (l))

/* Array initialization. */
static DATA_TYPE get_rand() { return (DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX; }

static void init_array(DATA_TYPE *A, int n) {
  for (int i = 0; i < n; i++) {
    A[i] = get_rand();
  }
}

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

// Returns the number of seconds elapsed between the two specified times
double elapsed_time(long long start_time, long long end_time) {
  return (double)(end_time - start_time) / (1000 * 1000);
}

void output_printfile(string &outfile, DATA_TYPE *out, int out_c, int out_h,
                      int out_w) {
  // ofstream myfile;
  // myfile.open(outfile);
  // assert(myfile.is_open());

  int print_size = 20;

  for (int i = 0; i < min(out_c, print_size); i++) {
    for (int j = 0; j < min(out_h, print_size); j++) {
      for (int k = 0; k < min(out_w, print_size); k++) {
        // myfile << std::fixed << std::setprecision(4) <<
        // cout << std::fixed << std::setprecision(4)
        //      << out[INDEX_3D(i, j, k, out_c, out_h, out_w)] << " ";
        printf("%.4ff ", out[INDEX_3D(i, j, k, out_c, out_h, out_w)]);
      }
      puts("");
    }
    puts("");
  }
  // myfile.close();
}

void check_result(DATA_TYPE *out1, DATA_TYPE *out2, int out_c, int out_h,
                  int out_w) {
  const DATA_TYPE EPS = 1e-3;
  for (int i = 0; i < out_c; i++) {
    for (int j = 0; j < out_h; j++) {
      for (int k = 0; k < out_w; k++) {
        if (abs(out1[INDEX_3D(i, j, k, out_c, out_h, out_w)] -
                out2[INDEX_3D(i, j, k, out_c, out_h, out_w)]) > EPS) {
          // cout << "Error at " << i << ", " << j << ", " << k << ": ";
          // cout << out1[INDEX_3D(i, j, k, out_c, out_h, out_w)]
          //      << " != " << out2[INDEX_3D(i, j, k, out_c, out_h, out_w)]
          //      << endl;
          printf("Error at %d, %d, %d: ", i, j, k);
          printf("%f != %f\n", out1[INDEX_3D(i, j, k, out_c, out_h, out_w)], 
            out2[INDEX_3D(i, j, k, out_c, out_h, out_w)]);
          return;
        }
      }
    }
  }
  puts("The result is OK!\n");
  // cout << "The result is OK!" << endl;
}

double conv2d(DATA_TYPE *input, int in_c, int in_h, int in_w, DATA_TYPE *kernel,
              int k_num, int k_c, int k_h, int k_w, DATA_TYPE *output,
              int padding, int stride) {
  printf("***** start vector/im2col version *****\n");

  int out_c = k_num;
  int out_h = (in_h - k_h + 2 * padding) / stride + 1;
  int out_w = (in_w - k_w + 2 * padding) / stride + 1;
  
  
  int pad_c = in_c;
  int pad_h = in_h + 2 * padding;
  int pad_w = in_w + 2 * padding;

  DATA_TYPE *pad_in;
  // memset(pad_in, 0, sizeof(DATA_TYPE) * pad_c * pad_h * pad_w);

  if (padding != 0) {
    pad_in = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * pad_c * pad_h * pad_w);
    fill(pad_in, pad_in + pad_c * pad_h * pad_w, 0);
    // printf("padding = %d\n", padding);
    for (int i = 0; i < in_c; i++) {
      for (int j = 0; j < in_h; j++) {
        for (int k = 0; k < in_w; k++) {
          pad_in[INDEX_3D(i, j + padding, k + padding, pad_c, pad_h, pad_w)] =
              input[INDEX_3D(i, j, k, in_c, in_h, in_w)];
          // printf("***");
          assert(pad_in[INDEX_3D(i, j + padding, k + padding, pad_c, pad_h,
                                 pad_w)] ==
                 input[INDEX_3D(i, j, k, in_c, in_h, in_w)]);
        }
      }
    }

    // printf("index(2, 2, 2): %d\n", INDEX_3D(2, 2, 2, 10, 10, 10));
  } else {
    pad_in = input;
  }
  // #ifdef USE_RISCV_VECTOR

  // start timer
  long long start = get_time();

#ifdef USE_IM2COL
  // int col_h = (in_h + 2 * padding - k_h) / stride + 1;
  // int col_w = (in_h + 2 * padding - k_w) / stride + 1;
  int col_c = in_c * k_h * k_w;
  DATA_TYPE *col_in =
      (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * col_c * out_h * out_w);

  // printf("\n***** im2col *****\n");
//   im2col(input, in_c, in_h, in_w, k_h, k_w, stride, padding, col_in);
    im2col(pad_in, in_c, pad_h, pad_w, k_h, k_w, stride, 0, col_in);
  // string outfilename("conv2d_output.txt");
  // output_printfile(outfilename, col_in, 1, k_c * k_w * k_h, out_h * out_w);
  // output_printfile(outfilename, input, 1, 3, in_w);
  // output_printfile(outfilename, kernel, 1, 2, k_w * k_h);

  /*
  gemm(int TA, int TB, int M, int N, int K, DATA_TYPE ALPHA,
      DATA_TYPE *A, int lda,
      DATA_TYPE *B, int ldb,
      DATA_TYPE BETA,
      DATA_TYPE *C, int ldc)
  */
  // M = k_num ( = out_c)
  // N = out_w * out_h
  // K = k_h * k_w * k_c (k_c == in_c)
  gemm(0, 0, k_num, out_h * out_w, k_h * k_w * in_c, 1.0, kernel,
       k_h * k_w * k_c,             // K
       col_in, out_h * out_w,       // N
       1.0, output, out_h * out_w); // N
  // output_printfile(outfilename, output, 1, 3, out_w);

#else

  int gvl = vsetvl_e32m1(k_w);
  _MMR_f32 zeroV = _MM_SET_f32(0, gvl);
  for (int i = 0; i < out_c; i++) {
    for (int j = 0; j < out_h; j++) {
      for (int k = 0; k < out_w; k++) {

        int start_j = j * stride;
        int start_k = k * stride;
        _MMR_f32 sum = _MM_SET_f32(0, gvl);
        // DATA_TYPE sum = 0;
        _MMR_f32 tmp_k;
        _MMR_f32 tmp_p;
        _MMR_f32 tmp_mul;
        
        // _MMR_f32 oneV = _MM_SET_f32(1, gvl);

        for (int ii = 0; ii < k_c; ii++) {
          
          gvl = vsetvl_e32m1(k_w); // gvl = min(k_w, VL_MAX)

          for (int kk = 0; kk < k_w; kk += gvl) {
            gvl = vsetvl_e32m1(k_w - kk);
            
            for (int jj = 0; jj < k_h; jj++) {
              tmp_k = _MM_LOAD_f32(
                  &kernel[INDEX_4D(i, ii, jj, kk, out_c, k_c, k_h, k_w)], gvl);
              tmp_p =
                  _MM_LOAD_f32(&pad_in[INDEX_3D(ii, jj + start_j, kk + start_k,
                                                pad_c, pad_h, pad_w)], gvl);
              // // tmp_mul = _MM_SET_f32(1.0, gvl);
              // tmp_mul = _MM_MUL_f32(tmp_k, tmp_p, gvl); // vector * vector

              // sum = _MM_ADD_f32(sum, tmp_mul, gvl);
              sum =_MM_MACC_f32(sum, tmp_k, tmp_p, gvl);
            }
          }
        }
        gvl = vsetvl_e32m1(k_w);
        sum = _MM_REDUSUM_f32(sum, sum, zeroV, gvl);
        DATA_TYPE res = _MM_VGETFIRST_f32(sum);
        output[INDEX_3D(i, j, k, out_c, out_h, out_w)] = res;
        //_MM_STORE_f32(&output[i][j][k], sum, gvl);
      }
    }
  }
  FENCE();
#endif // if use im2col
       /*
       #else
           printf("***** start serial version *****\n");
           // Start timer.
           long long start = get_time();
           for (int i = 0; i < out_c; i ++) {
               for (int j = 0; j < out_h; j ++) {
                   for (int k = 0; k < out_w; k ++) {
     
                       int start_j = j * stride;
                       int start_k = k * stride;
                       DATA_TYPE sum = 0;
     
                       for (int ii = 0; ii < in_c; ii ++) {
                           for (int jj = 0; jj < k_h; jj ++) {
                               for (int kk = 0; kk < k_w; kk ++) {
                                   sum += kernel[INDEX_4D(i, ii, jj, kk, out_c, k_c,
       k_h, k_w)] *      pad_in[INDEX_3D(ii, jj+start_j, kk+start_k, pad_c, pad_h,
       pad_w)];
     
                                   // assert(kernel[INDEX_4D(i, ii, jj, kk, out_c,
       k_c, k_h, k_w)] != 0);
                                   // assert(pad_in[INDEX_3D(ii, jj+start_j,
       kk+start_k, k_c, k_h, k_w)] != 0);
                                   // assert(sum != 0);
                                   // printf("***");
                               }
                           }
                       }
                       output[INDEX_3D(i, j, k, out_c, out_h, out_w)] = sum;
                   }
               }
           }
     
       //#endif // if use vector
       */
  // stopping time
  long long end = get_time();
  // free(pad_in);

  return elapsed_time(start, end);
}

double conv2d_serial(DATA_TYPE *input, int in_c, int in_h, int in_w,
                     DATA_TYPE *kernel, int k_num, int k_c, int k_h, int k_w,
                     DATA_TYPE *output, int padding, int stride) {
  printf("***** start serial version *****\n");
  

  int out_c = k_num;
  int out_h = (in_h - k_h + 2 * padding) / stride + 1;
  int out_w = (in_w - k_w + 2 * padding) / stride + 1;
  int pad_c = in_c;
  int pad_h = in_h + 2 * padding;
  int pad_w = in_w + 2 * padding;

  DATA_TYPE *pad_in =
      (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * pad_c * pad_h * pad_w);
  // memset(pad_in, 0, sizeof(DATA_TYPE) * pad_c * pad_h * pad_w);

  if (padding != 0) {
    fill(pad_in, pad_in + pad_c * pad_h * pad_w, 0);
    // printf("padding = %d\n", padding);
    for (int i = 0; i < in_c; i++) {
      for (int j = 0; j < in_h; j++) {
        for (int k = 0; k < in_w; k++) {
          pad_in[INDEX_3D(i, j + padding, k + padding, pad_c, pad_h, pad_w)] =
              input[INDEX_3D(i, j, k, in_c, in_h, in_w)];
          // printf("***");
          assert(pad_in[INDEX_3D(i, j + padding, k + padding, pad_c, pad_h,
                                 pad_w)] ==
                 input[INDEX_3D(i, j, k, in_c, in_h, in_w)]);
        }
      }
    }

    // printf("index(2, 2, 2): %d\n", INDEX_3D(2, 2, 2, 10, 10, 10));
  } else
    pad_in = input;
  // string filename("fff");
  // output_printfile(filename, input, 3, 3, 3);
  // output_printfile(filename, pad_in, 3, 10, 10);

  /* Start timer. */
  long long start = get_time();

  for (int i = 0; i < out_c; i++) {
    for (int j = 0; j < out_h; j++) {
      for (int k = 0; k < out_w; k++) {

        int start_j = j * stride;
        int start_k = k * stride;
        DATA_TYPE sum = 0;

        for (int ii = 0; ii < in_c; ii++) {
          for (int jj = 0; jj < k_h; jj++) {
            for (int kk = 0; kk < k_w; kk++) {
              sum += kernel[INDEX_4D(i, ii, jj, kk, out_c, k_c, k_h, k_w)] *
                     pad_in[INDEX_3D(ii, jj + start_j, kk + start_k, pad_c,
                                     pad_h, pad_w)];

              // assert(kernel[INDEX_4D(i, ii, jj, kk, out_c, k_c, k_h, k_w)] !=
              // 0); assert(pad_in[INDEX_3D(ii, jj+start_j, kk+start_k, k_c,
              // k_h, k_w)] != 0); assert(sum != 0); printf("***");
            }
          }
        }
        output[INDEX_3D(i, j, k, out_c, out_h, out_w)] = sum;
      }
    }
  }
  // stopping time
  long long end = get_time();
  free(pad_in);

  return elapsed_time(start, end);
}

#undef INDEX_3D
#undef INDEX_4D

int main(int argc, char **argv) {
  if (argc != 8) {
    printf("Usage: conv2d in_c in_h in_w k_num k_h padding stride\n");
    exit(0);
  }

  // outfilename = argv[3];
  /* Retrieve problem size. */
  int in_c = atoi(argv[1]);
  int in_h = atoi(argv[2]);
  int in_w = atoi(argv[3]);
  int k_num = atoi(argv[4]);
  int k_c = in_c;
  int k_h = atoi(argv[5]);
  int k_w = k_h;
  int padding = atoi(argv[6]);
  int stride = atoi(argv[7]);

  int out_c = k_num;
  int out_h = (in_h - k_h + 2 * padding) / stride + 1;
  int out_w = (in_w - k_w + 2 * padding) / stride + 1;

  printf("in_c = %d, in_h = %d, in_w = %d\n", in_c, in_h, in_w);
  printf("k_num = %d, k_c = %d, k_h = %d, k_w = %d\n", k_num, k_c, k_h, k_w);
  printf("padding = %d, stride = %d\n", padding, stride);

  DATA_TYPE *in;
  DATA_TYPE *out;
  DATA_TYPE *kernel;

  /* Variable declaration/allocation. */
  in = (DATA_TYPE *)malloc(in_c * in_h * in_w * sizeof(DATA_TYPE));
  assert(in != NULL);
  out = (DATA_TYPE *)malloc(out_c * out_h * out_w * sizeof(DATA_TYPE));
  assert(out != NULL);
  kernel = (DATA_TYPE *)malloc(k_num * k_c * k_h * k_w * sizeof(DATA_TYPE));
  assert(kernel != NULL);

  /* Initialize array(s). */
  printf("initialing ...\n");
  init_array((DATA_TYPE *)in, in_c * in_h * in_w);
  //   init_array((DATA_TYPE*)out, out_c * out_h * out_w);
  fill(out, out + out_c * out_h * out_w, 0);
  //   memset(out, 0, out_c * out_h * out_w);
  init_array((DATA_TYPE *)kernel, k_num * k_c * k_h * k_w);

  printf("computing ...\n");

  /* Run kernel. */
  // #ifdef USE_RISCV_VECTOR
  DATA_TYPE *out_v =
      (DATA_TYPE *)malloc(out_c * out_h * out_w * sizeof(DATA_TYPE));
  fill(out_v, out_v + out_c * out_h * out_w, 0);
  double cost_time_v = conv2d(in, in_c, in_h, in_w, kernel, k_num, k_c, k_h,
                              k_w, out_v, padding, stride);
  printf("vector/im2col time: %lf\n", cost_time_v);
  // #endif

  double cost_time_s = conv2d_serial(in, in_c, in_h, in_w, kernel, k_num, k_c,
                                     k_h, k_w, out, padding, stride);

  printf("serial direct conv time: %lf\n", cost_time_s);


  string outfilename("conv2d_output.txt");
  // output_printfile(outfilename, out_v, out_c, out_h, out_w);
  // printf("input:\n");
  // output_printfile(outfilename, in, in_c, in_h, in_w);

  // #ifdef USE_RISCV_VECTOR
  check_result(out, out_v, out_c, out_h, out_w);

  //   output_printfile(outfilename, out, out_c, out_h, out_w);
  // output_printfile(outfilename, in, in_c, in_h, in_w);
  // output_printfile(outfilename, kernel, k_c, k_h, k_w);
  //   output_printfile(outfilename, out, out_c, out_h, out_w);
  // #endif  //

  /* Be clean. */
  free(in);
  free(out);
  free(kernel);

  return 0;
}
