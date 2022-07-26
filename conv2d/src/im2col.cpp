/**
 *
 */
/* conv2d.cpp */

#include "conv2d_utils.hpp"

// #define INDEX_3D(i, j, k, N1, N2, N3) ((i) * N2 * N3 + (j) * N3 + (k))
// #define INDEX_4D(i, j, k, l, N1, N2, N3, N4) \
//         ((i) * N2 * N3 * N4 + (j) * N3 * N4 + (k) * N4 + (l))

// im2col
DATA_TYPE im2col_get_pixel(DATA_TYPE *im, int height, int width, int channels,
                           int row, int col, int channel, int pad) {
//   row -= pad;
//   col -= pad;

//   if (row < 0 || col < 0 || row >= height || col >= width)
//     return 0;
  return im[col + width * (row + height * channel)];
}

void im2col(DATA_TYPE *data_im, int channels, int height, int width, int k_h,
            int k_w, int stride, int pad, DATA_TYPE *data_col) {
  // printf(" in im2col\n");
  int c, h, w;
  int height_col = (height + 2 * pad - k_h) / stride + 1;
  int width_col = (width + 2 * pad - k_w) / stride + 1;

  int channels_col = channels * k_h * k_w;
  // printf("h = %d, w = %d, channels = %d\n", height_col, width_col,
  // channels_col);

#ifdef USE_RISCV_VECTOR
  if (stride == 1) {
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % k_w;
        int h_offset = (c / k_w) % k_h;
        int c_im = c / k_w / k_h;
        
        int gvl = vsetvl_e32m1(width_col);
        for (h = 0; h < height_col; ++h) {
            // pad == 0
            // int im_row = h_offset + h - pad;
            // edge condition
            // if (im_row < 0) continue;
            int im_row = h_offset + h;
            
            for (w = 0; w < width_col; w += gvl) {
                gvl = vsetvl_e32m1(width_col - w);
                // pad == 0
                int im_col = w_offset + w;
                // int im_col = w_offset + w - pad;
                // if (im_col < 0) {
                //     gvl = 1;
                //     continue;
                // }

                int col_index = (c * height_col + h) * width_col + w;
                int im_index = im_col + width * (im_row + height * c_im);
                
                // edge condition
                // int max_vl = std::min(width - im_col, height - im_row);
                // max_vl = std::min(max_vl, width_col - w);
                _MMR_f32 tmp;
                // puts("[im2col]: before vector load");
                tmp = _MM_LOAD_f32(data_im + im_index, gvl);
                // puts("[im2col]: after vector load");
                _MM_STORE_f32(data_col + col_index, tmp, gvl);
                // data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                                                    // im_row, im_col, c_im, pad);
            }
        }
    }
    // FENCE();
    return;
  }
#endif 

  for (c = 0; c < channels_col; ++c) {
    int w_offset = c % k_w;
    int h_offset = (c / k_w) % k_h;
    int c_im = c / k_w / k_h;
    for (h = 0; h < height_col; ++h) {
      for (w = 0; w < width_col; ++w) {
        int im_row = h_offset + h * stride;
        int im_col = w_offset + w * stride;
        int col_index = (c * height_col + h) * width_col + w;
        data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                                               im_row, im_col, c_im, pad);
      }
    }
  }
}

// // col2im
// void col2im_add_pixel(DATA_TYPE *im, int height, int width, int channels,
//                         int row, int col, int channel, int pad, DATA_TYPE
//                         val)
// {
//     row -= pad;
//     col -= pad;

//     if (row < 0 || col < 0 ||
//         row >= height || col >= width) return;
//     im[col + width*(row + height*channel)] += val;
// }

// void col2im(DATA_TYPE* data_col,
//          int channels,  int height,  int width,
//          int k_h, int k_w,  int stride, int pad, DATA_TYPE* data_im)
// {
//     int c,h,w;
//     int height_col = (height + 2*pad - k_h) / stride + 1;
//     int width_col = (width + 2*pad - k_w) / stride + 1;

//     int channels_col = channels * k_h * k_w;
//     printf("h = %d, w = %d, channels = %d\n", height_col, width_col,
//     channels_col); for (c = 0; c < channels_col; ++c) {
//         int w_offset = c % k_w;
//         int h_offset = (c / k_w) % k_h;
//         int c_im = c / k_h / k_w;
//         for (h = 0; h < height_col; ++h) {
//             for (w = 0; w < width_col; ++w) {
//                 int im_row = h_offset + h * stride;
//                 int im_col = w_offset + w * stride;
//                 int col_index = (c * height_col + h) * width_col + w;
//                 float val = data_col[col_index];
//                 col2im_add_pixel(data_im, height, width, channels,
//                         im_row, im_col, c_im, pad, val);
//             }
//         }
//     }
// }