#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/scale_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ScaleConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void ScaleConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }

  //update the correlative filters

  int kernel_length = kernel_size_ * kernel_size_;
  Dtype* const_trans_mat = (Dtype*)transform_matrix_->gpu_data();
  //first step, feed the mater filters with its dependent filters
  for(int g=0; g < scale_num_ * this->num_output_; ++g)
    caffe_gpu_gemv<Dtype>(CblasNoTrans, kernel_length, kernel_length,
    (Dtype)1., const_trans_mat, weight_diff + (scale_table_[g]+1)*kernel_length, (Dtype)1., weight_diff + scale_table_[g]*kernel_length);
  //Second step, broadcast the updated weight_diff 
  for(int g=0; g < scale_num_ * this->num_output_; ++g)
    caffe_gpu_gemv<Dtype>(CblasTrans, kernel_length, kernel_length,
    (Dtype)1., const_trans_mat, weight_diff + scale_table_[g]*kernel_length, (Dtype)0., weight_diff + (scale_table_[g]+1)*kernel_length);
}

INSTANTIATE_LAYER_GPU_FUNCS(ScaleConvolutionLayer);

}  // namespace caffe
