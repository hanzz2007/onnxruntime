// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// Test module for NWHC fp16 internal operators
//

#include <algorithm>
#include <random>

#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

class NhwcFp16PoolOpTester {
 private:
  bool is_max_pool_; // max or average pool
  std::vector<MLFloat16> X_data_;
  std::vector<int64_t>   X_shape_;
  std::vector<int64_t>   kernel_shape_;
  std::vector<int64_t>   pads_;
  std::vector<int64_t>   strides_;
  std::vector<int64_t>   dilations_;

  static size_t ShapeSize(const std::vector<int64_t>& shape) {
    return static_cast<size_t>(std::accumulate(shape.cbegin(), shape.cend(), 1LL, std::multiplies<int64_t>()));
  }

  static bool NextPosition(int64_t N, const int64_t* shape, int64_t* dims) {
    // Loop over spatial axes in reverse order to choose an index, like counting.
    bool incremented = false;
    for (int64_t d_i = N - 1; d_i >= 0; --d_i) {
      int64_t d_max = shape[d_i];
      ORT_ENFORCE(dims[d_i] < d_max);
      if (dims[d_i] == d_max - 1) {
        dims[d_i] = 0;
      } else {  // dims[d_i] < d_max - 1
        ++dims[d_i];
        incremented = true;
        break;
      }
    }
    return incremented;
  }

  void ComputeExpectedOutput(std::vector<MLFloat16>& Y_data, std::vector<int64_t>& Y_shape) {
    ORT_ENFORCE(X_shape_.size() >= 2 && X_shape_.size() == kernel_shape_.size() + 2);

    const size_t kernel_rank = kernel_shape_.size();

    const int64_t batch_count = X_shape_[0];
    const int64_t channels = X_shape_[X_shape_.size() - 1];

    std::vector<int64_t> pads(pads_);
    if (pads.empty()) {
      pads.resize(kernel_rank * 2, 0);
    }
    std::vector<int64_t> dilations(dilations_);
    if (dilations.empty()) {
      dilations.resize(kernel_rank, 1);
    }
    std::vector<int64_t> strides(strides_);
    if (strides.empty()) {
      strides.resize(kernel_rank, 1);
    }

    const int64_t* input_shape = X_shape_.data() + 1;

    // Compute the expected shape of the output.
    Y_shape.reserve(kernel_rank + 2);
    Y_shape.push_back(batch_count);
    for (size_t n = 0; n < kernel_rank; n++) {
      Y_shape.push_back(((input_shape[n] + pads[n] + pads[kernel_rank + n]) -
                         (dilations[n] * (kernel_shape_[n] - 1) + 1)) /
                            strides[n] +
                        1);
    }
    Y_shape.push_back(channels);
    Y_data.resize(ShapeSize(Y_shape));

    const int64_t* output_shape = Y_shape.data() + 1;

    const int64_t input_image_size = std::accumulate(
        input_shape, input_shape + kernel_rank, 1LL, std::multiplies<int64_t>());

    const MLFloat16* Xdata = X_data_.data();
    MLFloat16* Ydata = Y_data.data();

    for (int64_t batch = 0; batch < batch_count; batch++) {
      std::vector<int64_t> d_output(kernel_rank, 0);
      std::vector<int64_t> d_kernel(kernel_rank, 0);
      do {
        std::vector<float> accs(channels, is_max_pool_ ? std::numeric_limits<float>::lowest() : 0.f);
        size_t cnt = 0;
        do {
          int64_t input_offset = 0;
          bool is_padding = false;
          for (size_t axis = 0; axis < kernel_rank; ++axis) {
            int64_t input_dim = d_kernel[axis] * dilations[axis] + d_output[axis] * strides[axis] - pads[axis];
            is_padding |= !math::is_a_ge_zero_and_a_lt_b(input_dim, input_shape[axis]);
            input_offset *= input_shape[axis];
            input_offset += input_dim;
          }
          if (!is_padding) {
            const MLFloat16* data_ptr = Xdata + input_offset * channels;
            cnt++;
            for (int64_t c = 0; c < channels; c++) {
              if (is_max_pool_) {
                accs[c] = std::max(accs[c], data_ptr[c].ToFloat());
              } else {
                accs[c] += data_ptr[c].ToFloat();
              }
            }
          }
        } while (NextPosition(kernel_rank, kernel_shape_.data(), d_kernel.data()));
        for (int64_t c = 0; c < channels; c++) {
          if (!is_max_pool_) {
            accs[c] /= cnt;
          }
          Ydata[c] = MLFloat16(accs[c]);
        }
        Ydata += channels;
      } while (NextPosition(kernel_rank, output_shape, d_output.data()));
      Xdata += channels * input_image_size;
    }
  }

 public:
  NhwcFp16PoolOpTester(bool is_max_pool) : is_max_pool_(is_max_pool) {
  }

  void GenerateRandomInput(const std::vector<int64_t>& shape) {
    constexpr float MinimumFillValue = -23.0f;
    static size_t offset = 7;
    size_t shape_size = ShapeSize(shape);
    X_data_.resize(shape_size);

    for (size_t n = 0; n < shape_size; n++) {
      offset = (offset + 31) % 47;
      X_data_[n] = MLFloat16((MinimumFillValue + offset) / 16.0f);
    }
    X_shape_ = shape;
  }

  void SetKernelShape(const std::vector<int64_t>& kernel_shape) {
    kernel_shape_ = kernel_shape;
  }

  void SetPads(const std::vector<int64_t>& pads) {
    pads_ = pads;
  }

  void SetStrides(const std::vector<int64_t>& strides) {
    strides_ = strides;
  }

  void SetDilations(const std::vector<int64_t>& dilations) {
    dilations_ = dilations;
  }

  void Run() {
    std::vector<MLFloat16> Y_data;
    std::vector<int64_t> Y_shape;
    ComputeExpectedOutput(Y_data, Y_shape);

    OpTester test(is_max_pool_ ? "MaxPool" : "AveragePool", 11, onnxruntime::kMSInternalNHWCDomain);
    test.AddInput<MLFloat16>("x", X_shape_, X_data_);
    test.AddOutput<MLFloat16>("y", Y_shape, Y_data);
    test.AddAttribute("kernel_shape", kernel_shape_);
    if (!pads_.empty()) {
      test.AddAttribute("pads", pads_);
    }
    if (!strides_.empty()) {
      test.AddAttribute("strides", strides_);
    }
    if (!dilations_.empty()) {
      test.AddAttribute("dilations", dilations_);
    }
    test.Run(OpTester::ExpectResult::kExpectSuccess, "");
  }
};


TEST(NhwcFp16PoolOpTest, MaxPool1D) {
  for (int64_t channels = 1; channels < 94; channels++) {
    NhwcFp16PoolOpTester test(true);
    test.GenerateRandomInput({1, 23, channels});
    test.SetKernelShape({5});
    test.SetPads({2, 2});
    test.Run();
  }
}

TEST(NhwcFp16PoolOpTest, MaxPool2D) {
  for (int64_t channels = 1; channels < 94; channels++) {
    NhwcFp16PoolOpTester test(true);
    test.GenerateRandomInput({1, 15, 19, channels});
    test.SetKernelShape({3, 5});
    test.SetPads({1, 1, 1, 1});
    test.Run();
  }
}

TEST(NhwcFp16PoolOpTest, MaxPool3D) {
  for (int64_t channels = 1; channels < 94; channels++) {
    NhwcFp16PoolOpTester test(true);
    test.GenerateRandomInput({1, 9, 13, 15, channels});
    test.SetKernelShape({2, 4, 6});
    test.SetPads({0, 0, 0, 1, 1, 1});
    test.Run();
  }
}

TEST(NhwcFp16PoolOpTest, MaxPoolStrides) {
  NhwcFp16PoolOpTester test(true);
  test.GenerateRandomInput({4, 23, 19, 32});
  test.SetKernelShape({3, 3});
  test.SetStrides({2, 2});
  test.Run();
}

TEST(NhwcFp16PoolOpTest, MaxPoolDilations) {
  NhwcFp16PoolOpTester test(true);
  test.GenerateRandomInput({4, 23, 19, 32});
  test.SetKernelShape({3, 3});
  test.SetDilations({2, 2});
  test.Run();
}

TEST(NhwcFp16PoolOpTest, AvgPool1D) {
  for (int64_t channels = 1; channels < 94; channels++) {
    NhwcFp16PoolOpTester test(false);
    test.GenerateRandomInput({1, 23, channels});
    test.SetKernelShape({5});
    test.SetPads({2, 2});
    test.Run();
  }
}

TEST(NhwcFp16PoolOpTest, AvgPool2D) {
  for (int64_t channels = 1; channels < 94; channels++) {
    NhwcFp16PoolOpTester test(false);
    test.GenerateRandomInput({1, 15, 19, channels});
    test.SetKernelShape({3, 5});
    test.SetPads({1, 1, 1, 1});
    test.Run();
  }
}

TEST(NhwcFp16PoolOpTest, AvgPool3D) {
  for (int64_t channels = 1; channels < 94; channels++) {
    NhwcFp16PoolOpTester test(false);
    test.GenerateRandomInput({1, 9, 13, 15, channels});
    test.SetKernelShape({2, 4, 6});
    test.SetPads({0, 0, 0, 1, 1, 1});
    test.Run();
  }
}

TEST(NhwcFp16PoolOpTest, AvgPoolStrides) {
  NhwcFp16PoolOpTester test(false);
  test.GenerateRandomInput({4, 23, 19, 32});
  test.SetKernelShape({3, 3});
  test.SetStrides({2, 2});
  test.Run();
}

/******
 * AveragePool op does not support dilations until version 19
TEST(NhwcFp16PoolOpTest, AvgPoolDilations) {
  NhwcFp16PoolOpTester test(false);
  test.GenerateRandomInput({4, 23, 19, 32});
  test.SetKernelShape({3, 3});
  test.SetDilations({2, 2});
  test.Run();
}
*/

TEST(NhwcFp16PoolOpTest, AvgPoolIncludePadPixel) {
  OpTester test("AveragePool", 11, onnxruntime::kMSInternalNHWCDomain);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{1, 1, 1, 1});
  test.AddAttribute("kernel_shape", std::vector<int64_t>{2, 2});
  test.AddAttribute("count_include_pad", (int64_t)1);
  std::vector<MLFloat16> x_vals = {
      MLFloat16(0.3337f), MLFloat16(0.8794f), MLFloat16(0.3375f),
      MLFloat16(0.6666f), MLFloat16(0.4426f), MLFloat16(0.6474f),
      MLFloat16(0.7675f), MLFloat16(0.8823f), MLFloat16(0.8852f)};

  std::vector<int64_t> x_dims = {1, 3, 3, 1};
  std::vector<int64_t> expected_dims = {1, 4, 4, 1};
  std::vector<MLFloat16> expected_vals = {
      MLFloat16(0.0834f), MLFloat16(0.3033f), MLFloat16(0.3042f), MLFloat16(0.0844f),
      MLFloat16(0.2501f), MLFloat16(0.5806f), MLFloat16(0.5767f), MLFloat16(0.2462f),
      MLFloat16(0.3585f), MLFloat16(0.6897f), MLFloat16(0.7144f), MLFloat16(0.3832f),
      MLFloat16(0.1919f), MLFloat16(0.4124f), MLFloat16(0.4419f), MLFloat16(0.2213f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(NhwcFp16PoolOpTest, GlobalAveragePool) {
  OpTester test("GlobalAveragePool", 1, onnxruntime::kMSInternalNHWCDomain);

  std::vector<MLFloat16> x_vals = {
    MLFloat16(0.687500f),
    MLFloat16(0.062500f),
    MLFloat16(0.312500f),
    MLFloat16(-0.062500f),
    MLFloat16(-0.625000f),
    MLFloat16(-0.437500f),
    MLFloat16(-0.812500f),
    MLFloat16(0.062500f),
    MLFloat16(0.750000f),
    MLFloat16(0.375000f),
    MLFloat16(-0.687500f),
    MLFloat16(-0.125000f),
    MLFloat16(-0.375000f),
    MLFloat16(0.500000f),
    MLFloat16(-0.750000f),
    MLFloat16(0.812500f),
    MLFloat16(-0.250000f),
    MLFloat16(0.437500f),
    MLFloat16(0.125000f),
    MLFloat16(0.937500f),
    MLFloat16(-0.312500f),
    MLFloat16(-0.750000f),
    MLFloat16(0.187500f),
    MLFloat16(0.875000f),
    MLFloat16(0.437500f),
    MLFloat16(-0.562500f),
    MLFloat16(0.125000f),
    MLFloat16(-0.312500f),
    MLFloat16(0.625000f),
    MLFloat16(-0.625000f),
    MLFloat16(0.875000f),
    MLFloat16(-0.125000f),
    MLFloat16(0.562500f),
    MLFloat16(0.125000f),
    MLFloat16(-0.875000f),
    MLFloat16(-0.187500f),
    MLFloat16(-0.625000f),
    MLFloat16(0.312500f),
    MLFloat16(-0.937500f),
    MLFloat16(0.562500f),
    MLFloat16(-0.437500f),
    MLFloat16(0.250000f),
    MLFloat16(-0.187500f),
    MLFloat16(0.750000f),
    MLFloat16(-0.500000f),
    MLFloat16(-0.937500f),
    MLFloat16(0.562500f),
    MLFloat16(0.687500f),
    MLFloat16(-0.687500f),
    MLFloat16(-0.187500f),
    MLFloat16(-0.062500f),
    MLFloat16(0.500000f),
    MLFloat16(-0.937500f),
    MLFloat16(-0.812500f),
    MLFloat16(-0.250000f),
    MLFloat16(0.250000f),
    MLFloat16(0.375000f),
    MLFloat16(0.937500f),
    MLFloat16(-0.500000f),
    MLFloat16(-0.375000f),
    MLFloat16(0.187500f),
    MLFloat16(0.687500f),
    MLFloat16(0.812500f),
    MLFloat16(-0.562500f),
    MLFloat16(-0.062500f),
    MLFloat16(0.062500f),
    MLFloat16(0.625000f),
    MLFloat16(-0.812500f),
    MLFloat16(-0.687500f),
    MLFloat16(-0.125000f),
    MLFloat16(0.375000f),
    MLFloat16(0.500000f),
    MLFloat16(-0.875000f),
    MLFloat16(-0.375000f),
    MLFloat16(-0.250000f),
    MLFloat16(0.312500f),
    MLFloat16(0.812500f),
    MLFloat16(0.937500f),
    MLFloat16(-0.437500f),
    MLFloat16(0.062500f),
    MLFloat16(0.187500f),
    MLFloat16(0.750000f),
    MLFloat16(-0.687500f),
    MLFloat16(-0.562500f),
    MLFloat16(0.187500f),
    MLFloat16(0.500000f),
    MLFloat16(0.625000f),
    MLFloat16(-0.562500f),
    MLFloat16(-0.750000f),
    MLFloat16(-0.125000f),
    MLFloat16(0.625000f),
    MLFloat16(0.437500f),
    MLFloat16(-0.875000f),
    MLFloat16(-0.125000f),
    MLFloat16(-0.312500f),
    MLFloat16(0.312500f),
    MLFloat16(0.125000f),
    MLFloat16(0.875000f),
    MLFloat16(-0.437500f),
    MLFloat16(-0.750000f),
    MLFloat16(0.125000f),
    MLFloat16(0.750000f),
    MLFloat16(0.437500f),
    MLFloat16(-0.625000f),
    MLFloat16(-0.125000f),
    MLFloat16(-0.312500f),
    MLFloat16(0.562500f),
    MLFloat16(0.500000f),
    MLFloat16(0.875000f),
    MLFloat16(-0.187500f),
    MLFloat16(-0.250000f),
    MLFloat16(0.125000f),
    MLFloat16(-0.937500f),
    MLFloat16(0.937500f),
    MLFloat16(0.687500f),
    MLFloat16(0.250000f),
    MLFloat16(0.187500f),
    MLFloat16(-0.062500f),
    MLFloat16(-0.500000f),
    MLFloat16(-0.562500f),
    MLFloat16(-0.812500f),
    MLFloat16(0.687500f),
    MLFloat16(0.625000f),
    MLFloat16(0.375000f),
    MLFloat16(-0.062500f),
    MLFloat16(-0.125000f),
    MLFloat16(-0.375000f),
    MLFloat16(-0.812500f),
    MLFloat16(-0.875000f),
    MLFloat16(0.812500f),
    MLFloat16(0.375000f),
    MLFloat16(0.312500f),
    MLFloat16(-0.875000f),
    MLFloat16(-0.375000f),
    MLFloat16(-0.437500f),
    MLFloat16(-0.125000f),
    MLFloat16(0.812500f),
    MLFloat16(0.750000f),
    MLFloat16(-0.750000f),
    MLFloat16(0.062500f),
    MLFloat16(0.125000f),
    MLFloat16(0.437500f),
    MLFloat16(-0.687500f),
    MLFloat16(-0.750000f),
    MLFloat16(-0.312500f),
    MLFloat16(-0.250000f),
    MLFloat16(0.437500f),
    MLFloat16(0.875000f),
    MLFloat16(0.937500f),
    MLFloat16(-0.312500f),
    MLFloat16(0.125000f),
    MLFloat16(0.250000f),
    MLFloat16(0.875000f),
    MLFloat16(-0.625000f),
    MLFloat16(-0.500000f),
    MLFloat16(0.125000f),
    MLFloat16(0.562500f),
    MLFloat16(0.687500f),
    MLFloat16(-0.625000f),
    MLFloat16(-0.187500f),
    MLFloat16(-0.062500f),
    MLFloat16(0.562500f),
    MLFloat16(-0.937500f),
    MLFloat16(-0.812500f),
    MLFloat16(-0.187500f),
    MLFloat16(0.250000f),
    MLFloat16(0.375000f),
    MLFloat16(-0.937500f),
    MLFloat16(-0.500000f),
    MLFloat16(-0.375000f),
    MLFloat16(0.250000f),
    MLFloat16(-0.687500f),
    MLFloat16(0.812500f),
    MLFloat16(-0.500000f),
    MLFloat16(0.500000f),
    MLFloat16(0.062500f),
    MLFloat16(0.687500f),
    MLFloat16(-0.250000f),
    MLFloat16(0.187500f),
    MLFloat16(-0.062500f),
    MLFloat16(0.937500f),
    MLFloat16(-0.562500f),
    MLFloat16(-0.812500f),
    MLFloat16(0.312500f),
    MLFloat16(0.625000f),
    MLFloat16(0.375000f),
    MLFloat16(-0.437500f),
    MLFloat16(-0.125000f),
    MLFloat16(-0.375000f),
    MLFloat16(0.750000f),
    MLFloat16(-0.875000f),
    MLFloat16(0.812500f)};
  std::vector<int64_t> x_dims = {1, 8, 8, 3};
  std::vector<int64_t> expected_dims = {1, 1, 1, 3};
  std::vector<MLFloat16> expected_vals = {MLFloat16(0.009765625f), MLFloat16(-0.017578125f), MLFloat16(0.017578125f)};

  test.AddInput<MLFloat16>("X", x_dims, x_vals);
  test.AddOutput<MLFloat16>("Y", expected_dims, expected_vals);
  test.Run();
}



#endif
}  // namespace test
}  // namespace onnxruntime
