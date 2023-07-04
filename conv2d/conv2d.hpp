#include "op.hpp"
class Conv2d : public Op{
public:
  Conv2d(int batch_size, int height_input, int width_input, int channel_input, 
         int channel_output, int kernel_height, int kernel_width,
         int stride_h = 1, int stride_w = 1, int pad_h = 0, 
         int pad_w = 0, int dilation_h = 1, int dilation_w = 1):
                                            batch_size_(batch_size),
                                            height_input_(height_input),
                                            width_input_(width_input),
                                            channel_input_(channel_input),
                                            channel_output_(channel_output),
                                            kernel_height_(kernel_height),
                                            kernel_width_(kernel_width),
                                            height_output_((height_input + 2 * pad_h - kernel_height) / stride_h + 1),
                                            width_output_((width_input + 2 * pad_w - kernel_width) / stride_w + 1),
                                            stride_h_(stride_h),
                                            stride_w_(stride_w),
                                            pad_h_(pad_h),
                                            pad_w_(pad_w),
                                            dilation_h_(dilation_h),
                                            dilation_w_(dilation_w){}

  void print(int batch_size, int height_input, int width_input, int channel_input, 
              int channel_output, int kernel_height, int kernel_width, int height_output, 
              int width_output, int stride_h, int stride_w, int pad_h, 
              int pad_w, int dilation_h, int dilation_w);
  void result();
  Conv2d& init();
  void init(float *input_images, float *filter, float *output_images);
  virtual void run() override;
  ~Conv2d();
private:
  int batch_size_, height_input_, width_input_, channel_input_, channel_output_, kernel_height_, kernel_width_, height_output_, width_output_;
  int stride_h_, stride_w_, pad_h_, pad_w_, dilation_h_, dilation_w_;
  float *input = nullptr, *kernel = nullptr, *output = nullptr;
};