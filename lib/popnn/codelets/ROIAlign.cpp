// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

static constexpr auto SPAN = poplar::VectorLayout::SPAN;

namespace popnn {
template <class T> class ROIAlignForward : public Vertex {
public:
  Vector<Input<Vector<T, SPAN, 4>>> bottom_data;
  Input<Vector<T, SPAN, 4>> bottom_rois;
  Input<Vector<int, SPAN, 4>> batch_index;
  Output<Vector<T, SPAN, 4>> top_data_grid;
  Input<int> iter;

  T spatial_scale;
  unsigned int height;
  unsigned int width;
  unsigned int aligned_height;
  unsigned int aligned_width;
  unsigned int iy;
  unsigned int ix;
  unsigned int bin_grid_h;
  unsigned int bin_grid_w;
  bool aligned;

  bool compute() {
    int index = batch_index[iter];
    T half_pixel = aligned ? (T)0.5 : (T)0.0;
    T align_boundary = aligned ? (T)0.0 : (T)1.0;

    T roi_start_w = bottom_rois[0] * spatial_scale - half_pixel;
    T roi_start_h = bottom_rois[1] * spatial_scale - half_pixel;
    T roi_end_w = bottom_rois[2] * spatial_scale - half_pixel;
    T roi_end_h = bottom_rois[3] * spatial_scale - half_pixel;

    T roi_width = fmaxf(roi_end_w - roi_start_w, align_boundary);
    T roi_height = fmaxf(roi_end_h - roi_start_h, align_boundary);

    T bin_size_h = roi_height / aligned_height;
    T bin_size_w = roi_width / aligned_width;

    T len_grid_h = bin_size_h / bin_grid_h;
    T len_grid_w = bin_size_w / bin_grid_w;

    unsigned int i = 0u;
    T y_start = roi_start_h + ((T)0.5 + iy) * len_grid_h;
    for (unsigned int ph = 0u; ph < aligned_height; ++ph) {
      T x_start = roi_start_w + ((T)0.5 + ix) * len_grid_w;
      for (unsigned int pw = 0u; pw < aligned_width; ++pw) {
        if (y_start <= height && x_start <= width) {
          T y = fmaxf(y_start, (T)0.0);
          T x = fmaxf(x_start, (T)0.0);
          int y_low = floorf(y);
          int x_low = floorf(x);
          int y_high = y_low >= height - 1 ? y_low : y_low + 1;
          int x_high = x_low >= width - 1 ? x_low : x_low + 1;
          int upleft = x_low + y_low * width;
          int upright = x_high + y_low * width;
          int downleft = x_low + y_high * width;
          int downright = x_high + y_high * width;
          T h_ratio = y - y_low, w_ratio = x - x_low;
          top_data_grid[i] =
              bottom_data[index][upleft] * ((T)1.0 - h_ratio) *
                  ((T)1.0 - w_ratio) +
              bottom_data[index][upright] * ((T)1.0 - h_ratio) * w_ratio +
              bottom_data[index][downleft] * h_ratio * ((T)1.0 - w_ratio) +
              bottom_data[index][downright] * h_ratio * w_ratio;
        }
        i++;
        x_start += bin_size_w;
      }
      y_start += bin_size_h;
    }
    return true;
  }
};

template class ROIAlignForward<float>;
template class ROIAlignForward<half>;

//
template <class T> class ROIAlignBackward : public Vertex {
public:
  Input<Vector<T, SPAN, 4>> top_diff;
  Input<Vector<T, SPAN, 4>> bottom_rois;
  Vector<InOut<Vector<T, SPAN, 4>>> bottom_diff;
  Input<Vector<int, SPAN, 4>> batch_index;
  Input<int> iter;
  Input<int> group;

  T spatial_scale;
  unsigned int height;
  unsigned int width;
  unsigned int aligned_height;
  unsigned int aligned_width;
  unsigned int iy;
  unsigned int ix;
  unsigned int bin_grid_h;
  unsigned int bin_grid_w;
  bool aligned;

  bool compute() {
    T num_grid = bin_grid_h * bin_grid_w;
    int index = batch_index[iter + group];
    T half_pixel = aligned ? (T)0.5 : (T)0.0;
    T align_boundary = aligned ? (T)0.0 : (T)1.0;

    T roi_start_w = bottom_rois[0] * spatial_scale - half_pixel;
    T roi_start_h = bottom_rois[1] * spatial_scale - half_pixel;
    T roi_end_w = bottom_rois[2] * spatial_scale - half_pixel;
    T roi_end_h = bottom_rois[3] * spatial_scale - half_pixel;

    T roi_width = fmaxf(roi_end_w - roi_start_w, align_boundary);
    T roi_height = fmaxf(roi_end_h - roi_start_h, align_boundary);

    T bin_size_h = roi_height / aligned_height;
    T bin_size_w = roi_width / aligned_width;

    T len_grid_h = bin_size_h / bin_grid_h;
    T len_grid_w = bin_size_w / bin_grid_w;

    unsigned int i = 0u;
    T y_start = roi_start_h + ((T)0.5 + iy) * len_grid_h;
    for (unsigned int ph = 0u; ph < aligned_height; ++ph) {
      T x_start = roi_start_w + ((T)0.5 + ix) * len_grid_w;
      for (unsigned int pw = 0u; pw < aligned_width; ++pw) {
        if (y_start <= height && x_start <= width) {
          T y = fmaxf(y_start, (T)0.0);
          T x = fmaxf(x_start, (T)0.0);
          int y_low = floorf(y);
          int x_low = floorf(x);
          int y_high = y_low >= height - 1 ? y_low : y_low + 1;
          int x_high = x_low >= width - 1 ? x_low : x_low + 1;
          int upleft = x_low + y_low * width;
          int upright = x_high + y_low * width;
          int downleft = x_low + y_high * width;
          int downright = x_high + y_high * width;
          T h_ratio = y - y_low, w_ratio = x - x_low;
          bottom_diff[index][upleft] +=
              top_diff[i] * ((T)1.0 - h_ratio) * ((T)1.0 - w_ratio) / num_grid;
          bottom_diff[index][upright] +=
              top_diff[i] * ((T)1.0 - h_ratio) * w_ratio / num_grid;
          bottom_diff[index][downleft] +=
              top_diff[i] * h_ratio * ((T)1.0 - w_ratio) / num_grid;
          bottom_diff[index][downright] +=
              top_diff[i] * h_ratio * w_ratio / num_grid;
        }
        i++;
        x_start += bin_size_w;
      }
      y_start += bin_size_h;
    }
    return true;
  }
};

template class ROIAlignBackward<float>;
template class ROIAlignBackward<half>;
} // namespace popnn