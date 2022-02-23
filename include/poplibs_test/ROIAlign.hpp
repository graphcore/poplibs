// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef ROI_ALIGN_CPU_H
#define ROI_ALIGN_CPU_H

#include <boost/multi_array.hpp>
#include <vector>

template <typename T> struct PreCalc {
  int pos1, pos2, pos3, pos4;
  T w1, w2, w3, w4;
};

/* ----------------------begin for forward---------------------  */
template <typename T>
void pre_calc_for_bilinear(const int h, const int w, const int pool_h,
                           const int pool_w, int b_grid_h, int b_grid_w,
                           T start_y, T start_x, T b_size_h, T b_size_w,
                           std::vector<PreCalc<T>> &pre_calc) {
  int idx = 0;
  for (int ph = 0; ph < pool_h; ++ph) {
    for (int pw = 0; pw < pool_w; ++pw) {
      for (int iy = 0; iy < b_grid_h; ++iy) {
        const T yy =
            start_y + ph * b_size_h +
            static_cast<T>(iy + 0.5f) * b_size_h / static_cast<T>(b_grid_h);
        for (int ix = 0; ix < b_grid_w; ++ix) {
          const T xx =
              start_x + pw * b_size_w +
              static_cast<T>(ix + 0.5f) * b_size_w / static_cast<T>(b_grid_w);
          T x = xx, y = yy;
          // situation 1: out of range
          if (y < -1.0 || y > h || x < -1.0 || x > w) {
            PreCalc<T> pc{0, 0, 0, 0, 0, 0, 0, 0};
            pre_calc[idx] = pc;
            idx += 1;
            continue;
          }
          // not exceed 1.0
          y = y <= 0 ? 0 : (y >= h - 1 ? h - 1 : y);
          x = x <= 0 ? 0 : (x >= w - 1 ? w - 1 : x);
          int y_low = (int)y;
          int x_low = (int)x;
          int y_high = y_low >= h - 1 ? y_low : y_low + 1;
          int x_high = x_low >= w - 1 ? x_low : x_low + 1;
          T ly = y - y_low, lx = x - x_low;
          T hy = 1.0 - ly, hx = 1.0 - lx;
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
          // in the feature map's position and correspond weights
          PreCalc<T> pc;
          pc.pos1 = y_low * w + x_low;
          pc.pos2 = y_low * w + x_high;
          pc.pos3 = y_high * w + x_low;
          pc.pos4 = y_high * w + x_high;
          pc.w1 = w1, pc.w2 = w2, pc.w3 = w3, pc.w4 = w4;
          pre_calc[idx] = pc;
          idx += 1;
        }
      }
    }
  }
}

template <typename T>
void roi_align_forward(boost::multi_array<T, 4> &feat,
                       boost::multi_array<T, 2> &rois,
                       const std::vector<unsigned> &feat_size,
                       const std::vector<unsigned> &rois_size, float scale,
                       const int ratio, boost::multi_array<T, 4> &out) {
  unsigned n_rois = rois_size[0], col_rois = rois_size[1],
           pool_h = rois_size[2], pool_w = rois_size[3];
  unsigned channel = feat_size[1], h = feat_size[2], w = feat_size[3];
  // #pragma omp parallel for
  for (unsigned n = 0; n < n_rois; ++n) {
    unsigned roi_batch_idx = 0;
    if (col_rois == 5) {
      roi_batch_idx = (unsigned)rois[n][0];
    }
    // Do not using rounding; this implementation detail is critical
    T start_x = rois[n][1] * scale;
    T start_y = rois[n][2] * scale;
    T end_x = rois[n][3] * scale;
    T end_y = rois[n][4] * scale;
    // Force malformed ROIs to be 1x1
    T roi_w = std::max(end_x - start_x, (float)1.);
    T roi_h = std::max(end_y - start_y, (float)1.);
    T bin_size_w = roi_w / static_cast<float>(pool_w);
    T bin_size_h = roi_h / static_cast<float>(pool_h);

    // We use roi_bin_grid to sample the grid and mimic integral
    unsigned bin_grid_h = (ratio > 0) ? ratio : std::ceil(roi_h / pool_h);
    unsigned bin_grid_w = (ratio > 0) ? ratio : std::ceil(roi_w / pool_w);

    // We do average (integral) pooling inside a bin
    const unsigned count = bin_grid_h * bin_grid_w;
    // get each bin's corresponding position and weights
    std::vector<PreCalc<T>> pre_calc(count * pool_h * pool_w);
    pre_calc_for_bilinear(h, w, pool_h, pool_w, bin_grid_h, bin_grid_w, start_y,
                          start_x, bin_size_h, bin_size_w, pre_calc);
    // map to feature map
    for (unsigned c = 0; c < channel; ++c) {
      int pre_calc_idx = 0;
      for (unsigned ph = 0; ph < pool_h; ++ph) {
        for (unsigned pw = 0; pw < pool_w; ++pw) {
          T output_val = 0.;
          for (unsigned iy = 0; iy < bin_grid_h; ++iy) {
            for (unsigned ix = 0; ix < bin_grid_w; ++ix) {
              PreCalc<T> pc = pre_calc[pre_calc_idx];
              output_val +=
                  pc.w1 * feat[roi_batch_idx][c][pc.pos1 / w][pc.pos1 % w] +
                  pc.w2 * feat[roi_batch_idx][c][pc.pos2 / w][pc.pos2 % w] +
                  pc.w3 * feat[roi_batch_idx][c][pc.pos3 / w][pc.pos3 % w] +
                  pc.w4 * feat[roi_batch_idx][c][pc.pos4 / w][pc.pos4 % w];
              pre_calc_idx += 1;
            }
          }
          output_val /= count;
          out[n][c][ph][pw] = output_val;
        }
      }
    }
  }
}

// input: BxCxHxW;  rois: Kx5
template <typename T>
boost::multi_array<T, 4> roi_align_forward_cpu(boost::multi_array<T, 4> &feat,
                                               boost::multi_array<T, 2> &rois,
                                               unsigned pool_h, unsigned pool_w,
                                               float scale, unsigned sample) {
  assert(feat.num_dimensions() == 4);
  assert(rois.num_dimensions() == 2);
  unsigned batch = feat.shape()[0];
  unsigned channel = feat.shape()[1];
  unsigned feat_h = feat.shape()[2];
  unsigned feat_w = feat.shape()[3];
  unsigned num_rois = rois.shape()[0];
  unsigned len_rois = rois.shape()[1];
  const std::vector<unsigned> rois_size = {num_rois, len_rois, pool_h, pool_w};
  const std::vector<unsigned> feat_size = {batch, channel, feat_h, feat_w};
  boost::multi_array<T, 4> output(
      boost::extents[num_rois][channel][pool_h][pool_w]);
  roi_align_forward(feat, rois, feat_size, rois_size, scale, sample, output);
  return output;
}

/*------------------------------end of forward-----------------------------*/

//------------------------------begin for backward-----------------------------
template <typename T>
void bilinear_interpolate_gradient(const int h, const int w, T y, T x,
                                   PreCalc<T> &pc) {
  if (y < -1.0 || y > h || x < -1.0 || x > w) {
    pc = {-1, -1, -1, -1, 0., 0., 0., 0.};
    return;
  }
  // not exceed 1.0
  y = y <= 0 ? 0 : (y >= h - 1 ? h - 1 : y);
  x = x <= 0 ? 0 : (x >= w - 1 ? w - 1 : x);
  int y_low = (int)y;
  int x_low = (int)x;
  int y_high = y_low >= h - 1 ? y_low : y_low + 1;
  int x_high = x_low >= w - 1 ? x_low : x_low + 1;
  pc.pos1 = y_low * w + x_low;
  pc.pos2 = y_low * w + x_high;
  pc.pos3 = y_high * w + x_low;
  pc.pos4 = y_high * w + x_high;
  T ly = y - y_low, lx = x - x_low;
  T hy = 1.0 - ly, hx = 1.0 - lx;
  pc.w1 = hy * hx, pc.w2 = hy * lx, pc.w3 = ly * hx, pc.w4 = ly * lx;
}

template <typename T>
void roi_align_backward(unsigned total, boost::multi_array<T, 2> &rois,
                        boost::multi_array<T, 4> &grad_out, float scale,
                        const std::vector<unsigned> feat_size, unsigned pool_h,
                        unsigned pool_w, unsigned rois_col, unsigned sample,
                        boost::multi_array<T, 4> &grad_in) {
  // total=nxcxphxpw
  unsigned channel = feat_size[0], h = feat_size[1], w = feat_size[2];
  for (unsigned idx = 0; idx < total; ++idx) {
    unsigned pw = idx % pool_w;
    unsigned ph = (idx / pool_w) % pool_h;
    unsigned c = (idx / pool_h / pool_w) % channel;
    unsigned n = idx / pool_h / pool_w / channel;
    unsigned roi_batch_idx = 0;
    if (rois_col == 5) {
      roi_batch_idx = (unsigned)rois[n][0];
    }
    // Do not using rounding; this implementation detail is critical
    T start_x = rois[n][1] * scale;
    T start_y = rois[n][2] * scale;
    T end_x = rois[n][3] * scale;
    T end_y = rois[n][4] * scale;

    // Force malformed ROIs to be 1x1
    T roi_w = std::max(end_x - start_x, (T)1.0);
    T roi_h = std::max(end_y - start_y, (T)1.0);
    T b_size_h = roi_h / static_cast<T>(pool_h);
    T b_size_w = roi_w / static_cast<T>(pool_w);
    T grad_out_this_bin = grad_out[n][c][ph][pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    unsigned roi_bin_grid_h = (sample > 0) ? sample : std::ceil(roi_h / pool_h);
    unsigned roi_bin_grid_w = (sample > 0) ? sample : std::ceil(roi_w / pool_w);
    // We do average (integral) pooling inside a bin
    unsigned count = roi_bin_grid_h * roi_bin_grid_w;
    PreCalc<T> pc;
    for (unsigned iy = 0; iy < roi_bin_grid_h; iy++) {
      T y = start_y + ph * b_size_h +
            static_cast<T>(iy + .5f) * b_size_h /
                static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (unsigned ix = 0; ix < roi_bin_grid_w; ix++) {
        T x = start_x + pw * b_size_w +
              static_cast<T>(ix + .5f) * b_size_w /
                  static_cast<T>(roi_bin_grid_w);
        bilinear_interpolate_gradient(h, w, y, x, pc);
        T g1 = grad_out_this_bin * pc.w1 / count;
        T g2 = grad_out_this_bin * pc.w2 / count;
        T g3 = grad_out_this_bin * pc.w3 / count;
        T g4 = grad_out_this_bin * pc.w4 / count;
        // update grad_out
        if (pc.pos1 >= 0 && pc.pos2 >= 0 && pc.pos3 >= 0 && pc.pos4 >= 0) {
          grad_in[roi_batch_idx][c][pc.pos1 / w][pc.pos1 % w] += g1;
          grad_in[roi_batch_idx][c][pc.pos2 / w][pc.pos2 % w] += g2;
          grad_in[roi_batch_idx][c][pc.pos3 / w][pc.pos3 % w] += g3;
          grad_in[roi_batch_idx][c][pc.pos4 / w][pc.pos4 % w] += g4;
        }
      }
    }
  }
}

template <typename T>
boost::multi_array<T, 4> roi_align_backward_cpu(
    boost::multi_array<T, 2> &rois, boost::multi_array<T, 4> &grad_out,
    unsigned b_size, unsigned channel, unsigned h, unsigned w, unsigned pool_h,
    unsigned pool_w, float scale, unsigned sample) {
  assert(rois.num_dimensions() == 2);
  unsigned rois_col = rois.shape()[1];
  boost::multi_array<T, 4> grad_in(boost::extents[b_size][channel][h][w]);
  unsigned total = grad_out.num_elements();
  roi_align_backward(total, rois, grad_out, scale, {channel, h, w}, pool_h,
                     pool_w, rois_col, sample, grad_in);
  return grad_in;
}
/*------------------------------end of backward-----------------------------*/

#endif // ROI_ALIGN_CPU_H