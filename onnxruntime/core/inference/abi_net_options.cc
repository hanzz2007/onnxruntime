// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>
#include <cassert>

#include "core/graph/onnx_protobuf.h"
#include "core/common/inlined_containers.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"
#include "core/session/inference_session.h"
#include "core/providers/cuda/cuda_provider_factory_creator.h"
#include "core/providers/cpu/cpu_provider_factory_creator.h"
#include "core/session/abi_session_options_impl.h"

#include "abi_net_options_impl.h"

using OrtApis::CreateStatus;

OrtNetOptions::~OrtNetOptions() = default;

OrtNetOptions& OrtNetOptions::operator=(const OrtNetOptions&) {
  ORT_THROW("not implemented");
}

OrtNetOptions::OrtNetOptions(const OrtNetOptions& other)
    : value(other.value), provider_factory_creators(other.provider_factory_creators) {
}

OrtStatusPtr OrtNetOptions_Create(OrtNetOptions** out) {
  API_IMPL_BEGIN
  GSL_SUPPRESS(r .11)
  *out = new OrtNetOptions();
  return nullptr;
  API_IMPL_END
}

void OrtNetOptions_Destroy(OrtNetOptions* ptr) {
  delete ptr;
}

OrtStatusPtr OrtExecOptions_Create(OrtExecOptions** out) {
  API_IMPL_BEGIN
  GSL_SUPPRESS(r .11)
  *out = new OrtExecOptions();
  return nullptr;
  API_IMPL_END
}

OrtStatusPtr OrtExecOptions_SetUserStream(OrtExecOptions* out, const char* provider, void* stream) {
  API_IMPL_BEGIN
  GSL_SUPPRESS(r .11)
  out->streams_map[provider] = stream;
  return nullptr;
  API_IMPL_END
}

void OrtExecOptions_Destroy(OrtExecOptions* ptr) {
  delete ptr;
}

OrtStatusPtr OrtNetOptions_SetParam(OrtNetOptions* ptr, const char* sz_key, const char* sz_value) {
  auto& options = *ptr;
  const std::string key = sz_key;
  if (key == "optimized_model_file_path") {
    options.value.optimized_model_filepath = sz_value;
  } else if (key == "enable_profiling") {
    options.value.enable_profiling = atoi(sz_value);
  } else if (key == "profile_file_prefix") {
    options.value.profile_file_prefix = sz_value;
  } else if (key == "enable_mem_pattern") {
    options.value.enable_mem_pattern = atoi(sz_value);
  } else if (key == "enable_cpu_mem_arena") {
    options.value.enable_cpu_mem_arena = atoi(sz_value);
  } else if (key == "log_id") {
    options.value.session_logid = sz_value;
  } else if (key == "log_verbosity_level") {
    options.value.session_log_verbosity_level = atoi(sz_value);
  } else if (key == "log_severity_level") {
    options.value.session_log_severity_level = atoi(sz_value);
  } else if (key == "graph_optimization_level") {
    options.value.graph_optimization_level = (onnxruntime::TransformerLevel)atoi(sz_value);
    // } else if (key == "intra_op_num_threads") {
    //   options.value.intra_op_num_threads = atoi(sz_value);
    // } else if (key == "inter_op_num_threads") {
    //   options.value.inter_op_num_threads = atoi(sz_value);
  } else if (key == "disable_per_session_threads") {
    options.value.use_per_session_threads = atoi(sz_value) == 0;
  } else {
    return onnxruntime::ToOrtStatus(options.value.config_options.AddConfigEntry(key.c_str(), sz_value));
  }
  return nullptr;
}

OrtStatusPtr OrtNetOptions_AppendExecutionProvider(OrtNetOptions* options, const char* p_name, void* p_ep_options) {
  std::string name = p_name;
  if (name == "cuda") {
    const OrtCUDAProviderOptions ep_options = *(OrtCUDAProviderOptions*)p_ep_options;
    auto creator_fn = [ep_options](void* user_stream, std::shared_ptr<onnxruntime::IExecutionProviderFactory>& out_factory) -> OrtStatusPtr {
      auto ep_options_new = ep_options;
      if (user_stream) {
        ep_options_new.user_compute_stream = user_stream;
        ep_options_new.has_user_compute_stream = 1;
      }
      out_factory = onnxruntime::CudaProviderFactoryCreator::Create(&ep_options_new);
      if (!out_factory) {
        return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Cuda: Failed to load shared library");
      }
      return nullptr;
    };
    options->provider_factory_creators.emplace_back("cuda", creator_fn);
  } else if (name == "cpu") {
    auto creator_fn = [](void* user_stream, std::shared_ptr<onnxruntime::IExecutionProviderFactory>& out_factory) -> OrtStatusPtr {
      ORT_ENFORCE(!user_stream, "non-cuda ep should not have user stream");
      out_factory = onnxruntime::CPUProviderFactoryCreator::Create(0);
      if (!out_factory) {
        return OrtApis::CreateStatus(ORT_FAIL, "OrtSessionOptionsAppendExecutionProvider_Cpu: Failed to load shared library");
      }
      return nullptr;
    };
    options->provider_factory_creators.emplace_back("cpu", creator_fn);
  } else {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid execution provider");
  }
  return nullptr;
}

// struct OrtSessionOptions {
//   onnxruntime::SessionOptions value;
//   std::vector<OrtCustomOpDomain*> custom_op_domains_;
//   std::vector<std::shared_ptr<onnxruntime::IExecutionProviderFactory>> provider_factories;
//   OrtSessionOptions() = default;
//   ~OrtSessionOptions();
//   OrtSessionOptions(const OrtSessionOptions& other);
//   OrtSessionOptions& operator=(const OrtSessionOptions& other);
// };

// OrtStatusPtr OrtNetOptions_RegisterCustomOpsLibrary(OrtNetOptions* options, const char* library_path) {
//   API_IMPL_BEGIN
//   options->custom_op_libraries.push_back(library_path);

//   using namespace onnxruntime;

//  void* library_handle = nullptr;
//   ORT_API_RETURN_IF_STATUS_NOT_OK(Env::Default().LoadDynamicLibrary(library_path, false, &library_handle));
//   if (!library_handle)
//     return CreateStatus(ORT_FAIL, "RegisterCustomOpsLibrary: Failed to load library");

//   OrtStatusPtr(ORT_API_CALL * RegisterCustomOps)(OrtSessionOptions * options, const OrtApiBase* api);

//   ORT_API_RETURN_IF_STATUS_NOT_OK(Env::Default().GetSymbolFromLibrary(library_handle, "RegisterCustomOps",
//                                                                       (void**)&RegisterCustomOps));
//   if (!RegisterCustomOps)
//     return CreateStatus(ORT_FAIL, "RegisterCustomOpsLibrary: Entry point RegisterCustomOps not found in library");

//   OrtSessionOptions session_options;
//   session_options.value = options->value;
//   session_options.custom_op_domains_ = options->custom_op_domains;
//   return RegisterCustomOps(&session_options, OrtGetApiBase());
//   API_IMPL_END
// }
