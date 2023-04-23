
#pragma once

#include "core/session/onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_RUNTIME_CLASS(NetOptions);
ORT_RUNTIME_CLASS(Network);

ORT_RUNTIME_CLASS(ExecOptions);
ORT_RUNTIME_CLASS(Executor);

#define ORT_INFER_API_IMPL(RET, NAME, ARGS) \
  ORT_EXPORT RET ORT_API_CALL NAME ARGS ;

#define ORT_INFER_API(RET, NAME, ARGS) \
  ORT_INFER_API_IMPL(RET, NAME, ARGS)  \
  typedef RET(ORT_API_CALL * Proc_##NAME) ARGS;

ORT_INFER_API(OrtStatusPtr, OrtNetOptions_Create, (OrtNetOptions * *out));
ORT_INFER_API(void, OrtNetOptions_Destroy, (OrtNetOptions * in));
ORT_INFER_API(OrtStatusPtr, OrtNetOptions_SetParam, (OrtNetOptions * options, const char* key, const char* value));
ORT_INFER_API(OrtStatusPtr, OrtNetOptions_AppendExecutionProvider, (OrtNetOptions* options, const char* p_name, void* p_ep_options));

// ORT_INFER_API(OrtStatusPtr, OrtNetOptions_RegisterCustomOpsLibrary, (OrtNetOptions * options, const char* library_path, void** library_handle));

ORT_INFER_API(OrtStatusPtr, OrtNetwork_Create, (const OrtEnv* env, OrtNetOptions* options, const void* model_data, size_t model_data_length, OrtNetwork** out));
ORT_INFER_API(void, OrtNetwork_Destroy, (OrtNetwork* network));

ORT_INFER_API(OrtStatusPtr, OrtNetwork_GetInputCount, (const OrtNetwork* network, size_t* out));
ORT_INFER_API(OrtStatusPtr, OrtNetwork_GetOutputCount, (const OrtNetwork* network, size_t* out));
ORT_INFER_API(OrtStatusPtr, OrtNetwork_GetInputTypeInfo, (const OrtNetwork* network, size_t index, OrtTypeInfo** type_info));
ORT_INFER_API(OrtStatusPtr, OrtNetwork_GetOutputTypeInfo, (const OrtNetwork* network, size_t index, OrtTypeInfo** type_info));
ORT_INFER_API(OrtStatusPtr, OrtNetwork_GetInputName, (const OrtNetwork* network, size_t index, OrtAllocator* allocator, char** value));
ORT_INFER_API(OrtStatusPtr, OrtNetwork_GetOutputName, (const OrtNetwork* network, size_t index, OrtAllocator* allocator, char** value));

ORT_INFER_API(OrtStatusPtr, OrtExecOptions_Create, (OrtExecOptions** out));
ORT_INFER_API(void, OrtExecOptions_Destroy, (OrtExecOptions* ptr));
ORT_INFER_API(OrtStatusPtr, OrtExecOptions_SetUserStream, (OrtExecOptions* out, const char* provider, void* stream));

ORT_INFER_API(OrtStatusPtr, OrtNetwork_CreateExecutor, (OrtNetwork * network, const OrtExecOptions* exec_options, OrtExecutor** out));
ORT_INFER_API(void, OrtNetwork_DestroyExecutor, (OrtNetwork * network, OrtExecutor* executor));

ORT_INFER_API(OrtStatusPtr, OrtExecutor_Run, (OrtExecutor * executor, const OrtRunOptions* run_options, const char* const* input_names, const OrtValue* const* input, size_t input_len, const char* const* output_names1, size_t output_names_len, OrtValue** output));

#define ORT_INFER_API_ITEM(name) Proc_##name name
struct OrtInferenceApi
{
ORT_INFER_API_ITEM(OrtNetOptions_Create);
ORT_INFER_API_ITEM(OrtNetOptions_Destroy);
ORT_INFER_API_ITEM(OrtNetOptions_SetParam);
ORT_INFER_API_ITEM(OrtNetOptions_AppendExecutionProvider);
ORT_INFER_API_ITEM(OrtNetwork_Create);
ORT_INFER_API_ITEM(OrtNetwork_Destroy);
ORT_INFER_API_ITEM(OrtNetwork_GetInputCount);
ORT_INFER_API_ITEM(OrtNetwork_GetOutputCount);
ORT_INFER_API_ITEM(OrtNetwork_GetInputTypeInfo);
ORT_INFER_API_ITEM(OrtNetwork_GetOutputTypeInfo);
ORT_INFER_API_ITEM(OrtNetwork_GetInputName);
ORT_INFER_API_ITEM(OrtNetwork_GetOutputName);
ORT_INFER_API_ITEM(OrtExecOptions_Create);
ORT_INFER_API_ITEM(OrtExecOptions_Destroy);
ORT_INFER_API_ITEM(OrtExecOptions_SetUserStream);
ORT_INFER_API_ITEM(OrtNetwork_CreateExecutor);
ORT_INFER_API_ITEM(OrtNetwork_DestroyExecutor);
ORT_INFER_API_ITEM(OrtExecutor_Run);
};

typedef struct OrtInferenceApi OrtInferenceApi;
ORT_INFER_API(const OrtInferenceApi*, OrtGetInferenceApi, ());

#ifdef __cplusplus
}
#endif

//! @}
