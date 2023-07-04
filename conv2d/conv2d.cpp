#include "conv2d.hpp"
#include <cstdlib>
#include <string.h>
#include <iostream>
#define index_w(co, kh, kw, ci)  ((((co) * kernel_height_ + kh) * kernel_width_ + kw) * channel_input_ + ci)
#define index_x(n,  hi, wi, ci)  ((((n)  * height_input_  + hi) * width_input_  + wi) * channel_input_ + ci)
#define index_y(n,  ho, wo, co)  ((((n)  * height_output_ + ho) * width_output_  + wo) * channel_output_ + co)

void Conv2d::run(){
  for (int n = 0; n < batch_size_; n++) {
    for (int h = 0; h < height_output_; h++) {
      for (int w = 0; w < width_output_; w++) {
        for (int co = 0; co < channel_output_; co++) {
          int h_index = h * stride_h_;
          int w_index = w * stride_w_;
          for (int kh = 0; kh < kernel_height_; kh++) {
            for (int kw = 0; kw < kernel_width_; kw++) {
              for (int ci =0; ci < channel_input_; ci++) {
                output[index_y(n, h, w, co)] += kernel[index_w(co, kh, kw, ci)] * input[index_x(n, h_index + kh, w_index + kw, ci)];
              }
            }
          }
        }
      }
    }
  }
}

void Conv2d::init(float *input_images, float *filter, float *output_images) {
  input = (float*) malloc(batch_size_ * height_input_ * width_input_ * channel_input_ * sizeof(float));
  kernel = (float*) malloc(channel_output_ * kernel_height_ * kernel_width_ * channel_input_ * sizeof(float)); 
  output = (float*) malloc(batch_size_ * height_output_ * width_output_ * channel_output_ * sizeof(float));

  memcpy(input, input_images, batch_size_ * height_input_ * width_input_ * channel_input_ * sizeof(float));
  memcpy(kernel, filter, channel_output_ * kernel_height_ * kernel_width_ * channel_input_ * sizeof(float));
  memcpy(output, output_images, batch_size_ * height_output_ * width_output_ * channel_output_ * sizeof(float));
  return;
}

Conv2d& Conv2d::init() {
  input = (float*) malloc(batch_size_ * height_input_ * width_input_ * channel_input_ * sizeof(float));
  kernel = (float*) malloc(channel_output_ * kernel_height_ * kernel_width_ * channel_input_ * sizeof(float)); 
  output = (float*) malloc(batch_size_ * height_output_ * width_output_ * channel_output_ * sizeof(float));

  for(int i = 0; i < batch_size_ * height_input_ * width_input_ * channel_input_; i++) {
    input[i] = 1.0f;
  }
  for (int i = 0; i < channel_output_ * kernel_height_ * kernel_width_ * channel_input_; i++) {
    kernel[i] = 1.0f;
  }
  for (int i = 0; i < batch_size_ * height_output_ * width_output_ * channel_output_; i++) {
    output[i] = 0.0f;
  }
  return *this;
}

void Conv2d::result() {
  std::cout <<"conv " << std::endl;
  for (int i = 0; i < batch_size_ * height_output_ * width_output_ * channel_output_; i++) {
    std::cout << output[i] << " ";
  }
  std::cout << std::endl;
}

Conv2d::~Conv2d() {
  free(input);
  free(kernel);
  free(output);
}