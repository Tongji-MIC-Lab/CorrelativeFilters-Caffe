#ifndef CAFFE_PARAM_CF_CONV_LAYER_HPP_
#define CAFFE_PARAM_CF_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

/**
 * @brief Mostly the same with normal Convolutional Layer
 *  
 *
 *   Several pair of parametric CF corrlations are constructed among 
 *   the filters of conv layer.
 *
 */
template <typename Dtype>
class ParamCfConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:

  explicit ParamCfConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "ParamCfConvolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  void MakeParamCfTable();
 
 private:
  int param_cf_num_;
  int kernel_size_;
  vector<int> param_cf_table_;
};

}  // namespace caffe

#endif  // CAFFE_PARAM_CF_CONV_LAYER_HPP_
