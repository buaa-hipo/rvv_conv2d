// winograd conv2d
// #include <iostream>
// #include <fstream>
// #include <armadillo>
#include <math.h>
// #include <sys/time.h>
#include "conv2d_utils.hpp"

// using namespace std;
// using namespace arma;

#define INDEX_4D(i, j, k, l, N1, N2, N3, N4)                        \
  ((i) * (N2) * (N3) * (N4) + (j) * (N3) * (N4) + (k) * (N4) + (l))

// double timestamp();
// void report_winograd_statistics(int OC, int IC , int P, double time);

// mat** create_fourd_array(int d1, int d2, int d3, int d4) {
//   mat** array = new mat*[d1]();
//   for (int i = 0; i < d1; i++) {
//     array[i] = new mat[d2]();
//     for (int j = 0; j < d2; j++) {
//       array[i][j] = mat(d3, d4);
//     }
//   }
//   return array;
// }

// void free_fourd_array(mat** array, int d1) {
//   for (int i = 0; i < d1; i++) {
//     delete[] array[i];
//   }
//   delete[] array;
// }

// input: OC filters, IC channels, H height, W width, array of filters, image reference,
// result reference. Modifies result.
void conv2d_winograd(int OC, int IC , int H, int W, DATA_TYPE* filters, DATA_TYPE* image, DATA_TYPE* result) {
  // defining constants and values that follow directly from
  // https://arxiv.org/abs/1509.09308
  const int m = 2;
  const int r = 3;
  const int alpha = m + r - 1;
  int out_H = H - r + 1;
  int out_W = W - r + 1;
  int num_h_tiles = ceil(out_H/m);
  int num_w_tiles = ceil(out_W/m);
  int P = num_h_tiles * num_w_tiles;
  DATA_TYPE G[4][3] = { {1.0, 0.0, 0.0},
            {0.5, 0.5, 0.5},
            {0.5, -0.5, 0.5},
            {0.0, 0.0, 1.0} };
  DATA_TYPE B[4][4] = { {1, 0, 0, 0},
            {0, 1, -1, 1},
            {-1, 1, 1, 0},
            {0, 0, 0, -1} };
  DATA_TYPE A[4][2] = { {1, 0},
            {1, 1},
            {1, -1},
            {0, -1}};

  // a helper lambda function that generates b, the tile index,
  // from the y and x tile coordinates.
  auto gen_b = [num_h_tiles, num_w_tiles](int y, int x) -> int {
    return y * num_w_tiles + x;
  };

  // factoring out malloc'ing before measuring runtime
  // mat **U = create_fourd_array(alpha, alpha, OC , IC);
  DATA_TYPE* U = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * alpha * alpha * OC * IC);
  // DATA_TYPE* V = create_fourd_array(alpha, alpha, IC , P);
  DATA_TYPE* V = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * alpha * alpha * IC * P);
  // mat**M = new mat*[alpha]();
  DATA_TYPE* M = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * alpha * alpha * OC * P);
  // for (int xi = 0; xi < alpha; xi++) {
    // M[xi] = new mat[alpha];
  // }
  
  // puts("start generating U");
  // Generates U, an alpha x alpha x OC x IC transformation of the filters.
  for (int oc = 0; oc < OC; oc++) {
    for (int ic = 0; ic < IC ; ic++) {
      // flop: OC * IC * (4 * 3 * 5) * 2 
      // mat u = G * filters[k].slice(c) * G.t();
      DATA_TYPE* filters_st = filters + (oc * IC + ic) * 9;
      DATA_TYPE tmp[4][3] = {0};
      DATA_TYPE u[4][4] = {0};
      // G * filters
      gemm_nn (4, 3, 3, 1.0,
              (DATA_TYPE*)G, 3,
              filters_st, 3,
              (DATA_TYPE*)tmp, 3);
      // for (int i = 0; i < 4; i ++) {
      //   for (int j = 0; j < 3; j ++) {
      //     // DATA_TYPE tmp = 0;
      //     for (int k = 0; k < 3; k ++) {
      //       tmp[i][j] += G[i][k] * filters_st[k * 3 + j];
      //     }
      //   }
      // }

      // tmp * G.t
      gemm_nt (4, 4, 3, 1.0,
              (DATA_TYPE*)tmp, 3,
              (DATA_TYPE*)G, 3,
              (DATA_TYPE*)u, 4);
      // for (int i = 0; i < 4; i ++) {
      //   for (int j = 0; j < 4; j ++) {
      //     for (int k = 0; k < 3; k ++) {
      //       u[i][j] += tmp[i][k] * G[j][k];
      //     }
      //   }
      // }
      for (int xi = 0; xi < alpha; xi++) {
        for (int nu = 0; nu < alpha; nu++) {
          U[INDEX_4D(xi, nu, oc, ic, alpha, alpha, OC, IC)] = 
            ((DATA_TYPE*)u)[xi * alpha + nu];
        }
      }
    }
  }

  // puts("start generating V");
  DATA_TYPE* d = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * alpha * alpha);
  // DATA_TYPE* tmp = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * alpha * alpha);
  // DATA_TYPE* v = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * alpha * alpha);
  // Generates V, an alpha x alpha x IC x P transformation of the image.
  for (int ic = 0; ic < IC; ic++) {
    // mat channel = image.slice(c);
    DATA_TYPE* channel = image + ic * H * W;
    for (int y = 0; y < num_h_tiles; y++) {
      for (int x = 0; x < num_w_tiles; x++) {

        // mat d = channel(span(y * m, y * m + alpha - 1), span(x * m, x * m + alpha - 1)); 
        for (int i = 0; i < alpha; i ++) {
#ifdef USE_RISCV_VECTOR
          int gvl = vsetvl_e32m1(alpha);
          for (int j = 0; j < alpha; j += gvl) {
            gvl = vsetvl_e32m1(alpha - j);
            _MMR_f32 tmp = _MM_LOAD_f32(channel + (y * m + i) * W + x * m + j, gvl);
            _MM_STORE_f32(d + i * alpha + j, tmp, gvl);
          }
#else
          for (int j = 0; j < alpha; j ++) {
            d[i * alpha + j] = channel[(y * m + i) * W + x * m + j];
          }
#endif
        }
        DATA_TYPE tmp[4][4] = {0};
        DATA_TYPE v[4][4] = {0};
        // mat v = B.t() * d * B;
        gemm_tn (alpha, alpha, alpha, 1.0,
              (DATA_TYPE*)B, alpha,
              d, alpha,
              (DATA_TYPE*)tmp, alpha);
        gemm_nn (alpha, alpha, alpha, 1.0,
              (DATA_TYPE*)tmp, alpha,
              (DATA_TYPE*)B, alpha,
              (DATA_TYPE*)v, alpha);
        // flop: IC * P * (4 * 4 * 7) * 2

        int b = gen_b(y, x);
        for (int xi = 0; xi < alpha; xi++) {
          for (int nu = 0; nu < alpha; nu++) {
            V[INDEX_4D(xi, nu, ic, b, alpha, alpha, IC, P)] = 
              ((DATA_TYPE*)v)[xi * alpha + nu];
          }
        }
      }
    }
  }
  
  // puts("start computing M");
  // computes M, an alpha x alpha x OC x P matrix
  for (int xi = 0; xi < alpha; xi++) {
    for (int nu = 0; nu < alpha; nu++) {
      // flop: 16 * OC * P * (2IC - 1) 
      // M[xi][nu] = U[xi][nu] * V[xi][nu];
      gemm_nn (OC, P, IC, 1.0,
              U + (xi * alpha + nu) * OC * IC, IC,
              V + (xi * alpha + nu) * IC * P, P,
              M + (xi * alpha + nu) * OC * P, P);
      // for (int i = 0; i < xi; i ++)
    }
  }

  // puts("start computing conv");
  // computes the final convolution.
  // mat m_hold = zeros<mat>(alpha, alpha);
  DATA_TYPE m_hold[4 * 4] = {0};
  for (int oc = 0; oc < OC; oc++) {
    DATA_TYPE* out = result + oc * out_H * out_W;
    for (int y = 0; y < num_h_tiles; y++) {
      for (int x = 0; x < num_w_tiles; x++) {
        int b = gen_b(y, x);
        for (int xi = 0; xi < alpha; xi++) {
          for (int nu = 0; nu < alpha; nu++) {
            // m_hold(xi, nu) = M[xi][nu](oc, b);
            ((DATA_TYPE*)m_hold)[xi * alpha + nu] = 
              M[INDEX_4D(xi, nu, oc, b, alpha, alpha, OC, P)];
          }
        }
        // flop: OC * P * (2 * 4 * 7) * 2
        // result.slice(oc)(span(y*m, (y+1)*m-1), span(x*m, (x+1)*m-1)) = A.t() * m_hold * A;
        DATA_TYPE tmp1[2*4] = {0};
        DATA_TYPE tmp2[2*2] = {0};
        gemm_tn(2, 4, 4, 1.0,
                (DATA_TYPE*)A, 2,
                m_hold, 4,
                tmp1, 4);
        gemm_nn(2, 2, 4, 1.0,
                tmp1, 4,
                (DATA_TYPE*)A, 2,
                tmp2, 2);
        for (int i = 0; i < m; i ++) {
          for (int j = 0; j < m; j ++) {
            out[(y * m + i) * out_W + x * m + j] = tmp2[i * m + j];
          }
        }
      }
    }
  }

  // time = timestamp() - time;
  // report_winograd_statistics(OC, IC , P, time);

  // free_fourd_array(U, alpha);
  // free_fourd_array(V, alpha);
  // free_fourd_array(M, alpha);
}

// double timestamp()
// {
//   struct timeval tv;
//   gettimeofday (&tv, 0);
//   return tv.tv_sec + 1e-6*tv.tv_usec;
// }

// void report_winograd_statistics(int OC, int IC , int P, double time) {
//   int flop = (OC * IC * (4 * 3 * 5) * 2 +
//               IC * P * (4 * 4 * 7) * 2 + 
//               16 * OC * P * (2 * IC - 1) + 
//               OC * P * (2 * 4 * 7) * 2);
//   double mflops = flop / (1024.0 * 1024.0 * time);
//   cout << "Floating point operations: " << flop << "\n";
//   cout << "Time Elapsed: " << time << "\n";
//   cout << "MFlop/s: " << mflops << "\n";
// }

// int main(int argc, char* argv[])
// {
//   if (argc != 3) {
//     cout << "Usage: ./winograd <input filename> <output filename>\n";
//   }
//   ifstream file;
//   file.open(argv[1]);
//   int OC, IC , H, W;
//   file >> OC >> IC >> H >> W;

//   if (H % 2 != 0 || W % 2 != 0) {
//     cout << "Error: Image dimensions must be even." << endl;
//     return 1;
//   }

//   cube* filters = new cube[OC]();
//   for (int i = 0; i < OC; i++) {
//     filters[i] = cube(3, 3, C);
//     for (int j = 0; j < IC ; j++) {
//       for (int row = 0; row < 3; row++) {
//         for (int col = 0; col < 3; col ++) {
//           file >> filters[i](row, col, j);
//         }
//       }
//     }
//   }

//   cube image = cube(H, W, C);
//   for (int c = 0; c < IC ; c++) {
//     for (int row = 0; row < H; row++) {
//       for (int col = 0; col < W; col++) {
//         file >> image(row, col, c);
//       }
//     }
//   }
//   file.close();

//   cube result = cube(H-3+1, W-3+1, OC);
//   convolute(OC, IC , H, W, filters, image, result);

//   ofstream fileout;
//   fileout.open(argv[2], ofstream::out | ofstream::trunc );
//   fileout << OC << " " << IC << " " << H << " " << W << endl;
//   for (int i = 0; i < OC; i++) {
//     fileout << result.slice(i) << "\n";
//   }
//   fileout.close();

//   delete[] filters;
//   return 0;
// }

#undef INDEX_4D