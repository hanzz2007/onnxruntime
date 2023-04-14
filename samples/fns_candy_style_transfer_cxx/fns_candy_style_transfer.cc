// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <assert.h>
#include <stdio.h>

// #include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#ifdef _WIN32
#ifdef USE_DML
#include "providers.h"
#endif
#include <objbase.h>
#endif
#include "image_file.h"

#ifdef _WIN32
#define tcscmp wcscmp
#else
#define tcscmp strcmp
#endif

/**
 * convert input from HWC format to CHW format
 * \param input A single image. The byte array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A float array. should be freed by caller after use
 * \param output_count Array length of the `output` param
 */
void hwc_to_chw(const uint8_t* input, size_t h, size_t w, float** output, size_t* output_count) {
  size_t stride = h * w;
  *output_count = stride * 3;
  float* output_data = (float*)malloc(*output_count * sizeof(float));
  assert(output_data != NULL);
  for (size_t i = 0; i != stride; ++i) {
    for (size_t c = 0; c != 3; ++c) {
      output_data[c * stride + i] = input[i * 3 + c];
    }
  }
  *output = output_data;
}

/**
 * convert input from CHW format to HWC format
 * \param input A single image. This float array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A byte array. should be freed by caller after use
 */
static void chw_to_hwc(const float* input, size_t h, size_t w, uint8_t** output) {
  size_t stride = h * w;
  uint8_t* output_data = (uint8_t*)malloc(stride * 3);
  assert(output_data != NULL);
  for (size_t c = 0; c != 3; ++c) {
    size_t t = c * stride;
    for (size_t i = 0; i != stride; ++i) {
      float f = input[t + i];
      if (f < 0.f || f > 255.0f) f = 0;
      output_data[i * 3 + c] = (uint8_t)f;
    }
  }
  *output = output_data;
}

static void usage() { printf("usage: <model_path> <input_file> <output_file> [cpu|cuda|dml] \n"); }


#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  if (argc < 4) {
    usage();
    return -1;
  }

  // g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  // if (!g_ort) {
  //   fprintf(stderr, "Failed to init ONNX Runtime engine.\n");
  //   return -1;
  // }

#ifdef _WIN32
  // CoInitializeEx is only needed if Windows Image Component will be used in this program for image loading/saving.
  HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
  if (!SUCCEEDED(hr)) return -1;
#endif

  ORTCHAR_T* model_path = argv[1];
  ORTCHAR_T* input_file = argv[2];
  ORTCHAR_T* output_file = argv[3];
  // By default it will try CUDA first. If CUDA is not available, it will run all the things on CPU.
  // But you can also explicitly set it to DML(directml) or CPU(which means cpu-only).
  ORTCHAR_T* execution_provider = (argc >= 5) ? argv[4] : NULL;

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;

  if (execution_provider == std::string("cuda")) {
    OrtCUDAProviderOptions o;
    // Here we use memset to initialize every field of the above data struct to zero.
    memset(&o, 0, sizeof(o));
    // But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
    // type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
    o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    o.gpu_mem_limit = SIZE_MAX;
    session_options.AppendExecutionProvider_CUDA(o);
  }
  session_options.SetOptimizedModelFilePath("fns_candy_opt_cxx.onnx");
  session_options.SetLogSeverityLevel(0);
  session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

  Ort::Session session(env, model_path, session_options);
  assert(session.GetInputCount() == 1);
  assert(session.GetOutputCount() == 1);

  size_t input_height;
  size_t input_width;
  float* model_input;
  size_t model_input_ele_count;

  if (read_image_file(input_file, &input_height, &input_width, &model_input, &model_input_ele_count) != 0) {
    return -1;
  }
  if (input_height != 720 || input_width != 720) {
    printf("please resize to image to 720x720\n");
    free(model_input);
    return -1;
  }

  Ort::MemoryInfo mem_info("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault);
  const int64_t input_shape[] = {1, 3, 720, 720};
  const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
  const size_t model_input_len = model_input_ele_count;

  Ort::Allocator allocator(session, mem_info);

  auto input_tensor = Ort::Value::CreateTensor<float>(mem_info, model_input, model_input_len, input_shape, input_shape_len);
  assert(input_tensor.IsTensor());

  const char* input_names[] = {"inputImage"};
  const char* output_names[] = {"outputImage"};

  Ort::RunOptions run_options;
  auto output_tensors = session.Run(run_options, input_names, &input_tensor, 1, output_names, 1);
  assert(output_tensors.front().IsTensor());
  auto& output_tensor = output_tensors.front();

  auto output_tensor_data = output_tensor.GetTensorMutableData<float>();
  // auto output_tensor_data2 = output_tensor.GetTensorMutableRawData<float>();

  uint8_t* output_image_data = NULL;
  chw_to_hwc(output_tensor_data, 720, 720, &output_image_data);
  if (write_image_file(output_image_data, 720, 720, output_file) != 0) {
    return -1;
  }

  return 0;

  // OrtValue* output_tensor = NULL;
  // ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1,
  //                               &output_tensor));
  // assert(output_tensor != NULL);
  // ORT_ABORT_ON_ERROR(g_ort->IsTensor(output_tensor, &is_tensor));
  // assert(is_tensor);
  // int ret = 0;
  // float* output_tensor_data = NULL;
  // ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, (void**)&output_tensor_data));
  // uint8_t* output_image_data = NULL;
  // chw_to_hwc(output_tensor_data, 720, 720, &output_image_data);
  // if (write_image_file(output_image_data, 720, 720, output_file) != 0) {
  //   ret = -1;
  // }
  // g_ort->ReleaseValue(output_tensor);
  // g_ort->ReleaseValue(input_tensor);
  // free(model_input);
  // return ret;



  // g_ort->EnableProfiling(session_options, "./fns_candy_prof.json");

  // OrtSession* session;
  // ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));
  // verify_input_output_count(session);
  // ret = run_inference(session, input_file, output_file);
  // g_ort->ReleaseSessionOptions(session_options);
  // g_ort->ReleaseSession(session);
  // g_ort->ReleaseEnv(env);
  // if (ret != 0) {
  //   fprintf(stderr, "fail\n");
  // }
#ifdef _WIN32
  CoUninitialize();
#endif
  // return ret;
}
