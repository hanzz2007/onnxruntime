// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/allocator_adapters.h"
#include "core/session/inference_session_utils.h"
#include "core/session/IOBinding.h"
#include "core/framework/allocator.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/utils.h"
#include <cassert>
#include <cstring>
#include <functional>
#include <sstream>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/narrow.h"
#include "core/common/status.h"
#include "core/common/safeint.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/ort_value.h"
#include "core/providers/get_execution_providers.h"
#include "core/session/environment.h"
#include "core/framework/callback.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/session/inference_session.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/framework/data_types.h"
#include "core/framework/TensorSeq.h"
#include "core/platform/ort_mutex.h"
#include "core/common/string_helper.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
namespace onnxruntime {
ProviderInfo_CUDA* TryGetProviderInfo_CUDA();
}
#endif

#ifdef ENABLE_TRAINING_ON_DEVICE
#include "orttraining/training_api/include/onnxruntime_training_c_api.h"
#include "orttraining/training_api/include/ort_training_apis.h"
#endif

#ifdef USE_CANN
#include "core/providers/cann/cann_provider_factory.h"
#include "core/providers/cann/cann_execution_provider_info.h"
namespace onnxruntime {
ProviderInfo_CANN* TryGetProviderInfo_CANN();
}
#endif

#ifdef USE_DML
#include "core/providers/dml/dml_provider_factory.h"
const OrtDmlApi* GetOrtDmlApi(_In_ uint32_t version) NO_EXCEPTION;
#endif

#ifdef ENABLE_EXTENSION_CUSTOM_OPS
#include "onnxruntime_extensions.h"
#endif

#include "core/inference/abi_net_options_impl.h"
#include "core/inference/infer_c_api.h"
#include "core/session/ort_apis.h"

#if defined(_MSC_VER) && !defined(__clang__)
// The warning is: "Do not assign the result of an allocation or a function call with an owner<T> return value to a raw pointer, use owner<T> instead(i .11)."
// But this file is for C API. It can't use unique_ptr/shared_ptr in function signature.
#pragma warning(disable : 26400)
#endif
using namespace onnxruntime::logging;
using onnxruntime::BFloat16;
using onnxruntime::DataTypeImpl;
using onnxruntime::Environment;
using onnxruntime::IAllocator;
using onnxruntime::InputDefList;
using onnxruntime::MLFloat16;
using onnxruntime::narrow;
using onnxruntime::OutputDefList;
using onnxruntime::Tensor;
using onnxruntime::ToOrtStatus;
using onnxruntime::common::Status;

using namespace onnxruntime;

#ifndef ORT_STATUS_PTR
#ifdef _WIN32
#define ORT_STATUS_PTR _Check_return_ _Ret_maybenull_ OrtStatusPtr
#else
#define ORT_STATUS_PTR OrtStatus*
#endif
#endif

#define TENSOR_READ_API_BEGIN                          \
  API_IMPL_BEGIN                                       \
  auto v = reinterpret_cast<const ::OrtValue*>(value); \
  auto& tensor = v->Get<onnxruntime::Tensor>();

#define TENSOR_READWRITE_API_BEGIN \
  API_IMPL_BEGIN                   \
  auto v = (value);                \
  auto tensor = v->GetMutable<onnxruntime::Tensor>();

namespace {
// provider either model_path, or modal_data + model_data_length.
static ORT_STATUS_PTR CreateSessionAndLoadModel(const OrtNetOptions* options,
                                                const OrtEnv* env,
                                                const char* model_path,
                                                const void* model_data,
                                                size_t model_data_length,

                                                std::unique_ptr<onnxruntime::InferenceSession>& sess) {
  // quick check here to decide load path. InferenceSession will provide error message for invalid values.
  // TODO: Could move to a helper
  const Env& os_env = Env::Default();  // OS environment (!= ORT environment)
  bool load_config_from_model =
      os_env.GetEnvironmentVar(inference_session_utils::kOrtLoadConfigFromModelEnvVar) == "1";

  if (load_config_from_model) {
#if !defined(ORT_MINIMAL_BUILD)
    if (model_path != nullptr) {
      sess = std::make_unique<onnxruntime::InferenceSession>(
          options == nullptr ? onnxruntime::SessionOptions() : options->value,
          env->GetEnvironment(),
          model_path);
    } else {
      sess = std::make_unique<onnxruntime::InferenceSession>(
          options == nullptr ? onnxruntime::SessionOptions() : options->value,
          env->GetEnvironment(),
          model_data, static_cast<int>(model_data_length));
    }
#else
    return OrtApis::CreateStatus(ORT_FAIL, "Loading config from ONNX models is not supported in this build.");
#endif
  } else {
    sess = std::make_unique<onnxruntime::InferenceSession>(
        options == nullptr ? onnxruntime::SessionOptions() : options->value,
        env->GetEnvironment());
  }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  // Add custom domains
  if (options && !options->custom_op_domains.empty()) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->AddCustomOpDomains(options->custom_op_domains));
  }
#endif

  // Finish load
  if (load_config_from_model) {
#if !defined(ORT_MINIMAL_BUILD)
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load());
#endif
  } else {
    if (model_path != nullptr) {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load(model_path));
    } else {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Load(model_data, static_cast<int>(model_data_length)));
    }
  }

  return nullptr;
}

static ORT_STATUS_PTR InitializeSession(const OrtNetOptions* options,
                                        const std::unordered_map<std::string, void*>& user_streams,
                                        ::onnxruntime::InferenceSession* sess,
                                        OrtPrepackedWeightsContainer* prepacked_weights_container,
                                        const ModelWeightPtr model_weight) {
  // we need to disable mem pattern if DML is one of the providers since DML doesn't have the concept of
  // byte addressable memory
  std::vector<std::unique_ptr<IExecutionProvider>> provider_list;
  if (options) {
    for (const auto& kv : options->provider_factory_creators) {
      auto it_stream = user_streams.find(kv.first);
      std::shared_ptr<onnxruntime::IExecutionProviderFactory> factory;
      if (it_stream != user_streams.end()) {
        ORT_API_RETURN_IF_ERROR(kv.second(it_stream->second, factory));
      } else {
        ORT_API_RETURN_IF_ERROR(kv.second(nullptr, factory));
      }
      provider_list.push_back(factory->CreateProvider());
    }
  }

  // register the providers
  for (auto& provider : provider_list) {
    if (provider) {
      ORT_API_RETURN_IF_STATUS_NOT_OK(sess->RegisterExecutionProvider(std::move(provider)));
    }
  }

  if (prepacked_weights_container != nullptr) {
    ORT_API_RETURN_IF_STATUS_NOT_OK(sess->AddPrePackedWeightsContainer(
        reinterpret_cast<PrepackedWeightsContainer*>(prepacked_weights_container)));
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(sess->Initialize(model_weight));

  return nullptr;
}

}  // namespace

OrtStatusPtr OrtNetwork_GetInputCount(const OrtNetwork* network, size_t* out) {
  return OrtApis::SessionGetInputCount((const OrtSession*)network->session.get(), out);
}

OrtStatusPtr OrtNetwork_GetOutputCount(const OrtNetwork* network, size_t* out) {
  return OrtApis::SessionGetOutputCount((const OrtSession*)network->session.get(), out);
}

OrtStatusPtr OrtNetwork_GetInputTypeInfo(const OrtNetwork* network, size_t index, OrtTypeInfo** type_info) {
  return OrtApis::SessionGetInputTypeInfo((const OrtSession*)network->session.get(), index, type_info);
}

OrtStatusPtr OrtNetwork_GetOutputTypeInfo(const OrtNetwork* network, size_t index, OrtTypeInfo** type_info) {
  return OrtApis::SessionGetOutputTypeInfo((const OrtSession*)network->session.get(), index, type_info);
}

OrtStatusPtr OrtNetwork_GetInputName(const OrtNetwork* network, size_t index, OrtAllocator* allocator, char** value) {
  return OrtApis::SessionGetInputName((const OrtSession*)network->session.get(), index, allocator, value);
}

OrtStatusPtr OrtNetwork_GetOutputName(const OrtNetwork* network, size_t index, OrtAllocator* allocator, char** value) {
  return OrtApis::SessionGetOutputName((const OrtSession*)network->session.get(), index, allocator, value);
}

OrtStatusPtr OrtNetwork_CreateExecutor(OrtNetwork* network, const OrtExecOptions* exec_options, OrtExecutor** out) {
  API_IMPL_BEGIN

  OrtStatusPtr status = nullptr;
  *out = nullptr;

  ORT_TRY {
    std::unique_ptr<OrtExecutor> executor(new OrtExecutor);
    std::unique_lock<std::mutex> lock(network->init_mutex);

    if (!network->session_inited) {
      ORT_API_RETURN_IF_ERROR(InitializeSession(&network->options, exec_options->streams_map, network->session.get(),
                                                nullptr, nullptr));
      network->session_inited = true;
      network->streams_map = exec_options->streams_map;
    }

    if (network->session_assigned_ || network->streams_map != exec_options->streams_map) {
      lock.unlock();
      // Not first executor or streams not matched, create a new session
      std::unique_ptr<onnxruntime::InferenceSession> session;
      ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(&network->options, network->env, nullptr, network->model_data.data(),
                                                        network->model_data.size(), session));
      ORT_API_RETURN_IF_ERROR(InitializeSession(&network->options, exec_options->streams_map, session.get(),
                                                nullptr, network->session->GetSessionState().GetWeight()));
      executor->session = std::move(session);
    } else {
      // First executor, reuse the network session
      executor->session = network->session;
      network->session_assigned_ = executor.get();
    }

    *out = executor.release();
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

void OrtNetwork_DestroyExecutor(OrtNetwork* network, OrtExecutor* executor) {
  if (executor == network->session_assigned_) {
    std::lock_guard<std::mutex> lock(network->init_mutex);
    network->session_assigned_ = nullptr;
  }
  delete executor;
}

OrtStatusPtr OrtExecutor_Run(OrtExecutor* executor, const OrtRunOptions* run_options,
                             const char* const* input_names,
                             const OrtValue* const* input, size_t input_len,
                             const char* const* output_names1, size_t output_names_len,
                             OrtValue** output) {
  return OrtApis::Run((OrtSession*)executor->session.get(), run_options, input_names, input, input_len,
                      output_names1, output_names_len, output);
}

OrtStatusPtr OrtNetwork_Create(const OrtEnv* env, OrtNetOptions* options, const void* model_data, size_t model_data_length, OrtNetwork** out) {
  API_IMPL_BEGIN
  std::unique_ptr<OrtNetwork> network(new OrtNetwork);
  network->options = *options;
  network->model_data.resize(model_data_length);
  network->env = env;
  std::copy((const uint8_t*)model_data, (const uint8_t*)model_data + model_data_length, network->model_data.begin());

  std::unique_ptr<InferenceSession> session;
  ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(&network->options, network->env, nullptr, model_data,
                                                    model_data_length, session));
  network->session = std::move(session);

  *out = network.release();
  return nullptr;
  API_IMPL_END
}

void OrtNetwork_Destroy(OrtNetwork* network) {
  // auto weight = network->session->GetSessionState().GetWeight();
  network->session.reset();
  // ORT_ENFORCE(weight.use_count() == 1, "Network release while executors not been released, ",
  //             weight.use_count());
  delete network;
}

#define ORT_INFER_API_IMPL_ITEM(name) &::name

static const OrtInferenceApi inference_api = {
    ORT_INFER_API_IMPL_ITEM(OrtNetOptions_Create),
    ORT_INFER_API_IMPL_ITEM(OrtNetOptions_Destroy),
    ORT_INFER_API_IMPL_ITEM(OrtNetOptions_SetParam),
    ORT_INFER_API_IMPL_ITEM(OrtNetOptions_AppendExecutionProvider),
    ORT_INFER_API_IMPL_ITEM(OrtNetwork_Create),
    ORT_INFER_API_IMPL_ITEM(OrtNetwork_Destroy),
    ORT_INFER_API_IMPL_ITEM(OrtNetwork_GetInputCount),
    ORT_INFER_API_IMPL_ITEM(OrtNetwork_GetOutputCount),
    ORT_INFER_API_IMPL_ITEM(OrtNetwork_GetInputTypeInfo),
    ORT_INFER_API_IMPL_ITEM(OrtNetwork_GetOutputTypeInfo),
    ORT_INFER_API_IMPL_ITEM(OrtNetwork_GetInputName),
    ORT_INFER_API_IMPL_ITEM(OrtNetwork_GetOutputName),
    ORT_INFER_API_IMPL_ITEM(OrtExecOptions_Create),
    ORT_INFER_API_IMPL_ITEM(OrtExecOptions_Destroy),
    ORT_INFER_API_IMPL_ITEM(OrtExecOptions_SetUserStream),
    ORT_INFER_API_IMPL_ITEM(OrtNetwork_CreateExecutor),
    ORT_INFER_API_IMPL_ITEM(OrtNetwork_DestroyExecutor),
    ORT_INFER_API_IMPL_ITEM(OrtExecutor_Run)};

const OrtInferenceApi* OrtGetInferenceApi() {
  return &inference_api;
}
