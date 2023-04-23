// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <atomic>

#include "core/common/gsl.h"

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/common/profiler.h"
#include "core/framework/allocation_planner.h"
#include "core/framework/callback.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_providers.h"
#include "core/framework/session_options.h"
#include "core/inference/infer_c_api.h"
#include "core/providers/providers.h"

using ProviderFactoryCreator = std::function<OrtStatusPtr(void*, std::shared_ptr<onnxruntime::IExecutionProviderFactory>&)>;

struct OrtNetOptions {
  onnxruntime::SessionOptions value;
  std::vector<OrtCustomOpDomain*> custom_op_domains;
  std::vector<std::pair<std::string, ProviderFactoryCreator>> provider_factory_creators;
  // std::vector<std::string> custom_op_libraries;

  OrtNetOptions() = default;
  // ~OrtNetOptions();
  // OrtNetOptions(const OrtNetOptions& other);
  // OrtNetOptions& operator=(const OrtNetOptions& other);
};

struct ModelWeight;
using ModelWeightPtr = std::shared_ptr<ModelWeight>;

using SubgraphModelWeightMap =
    std::unordered_map<onnxruntime::NodeIndex, std::unordered_map<std::string, ModelWeightPtr>>;

struct ModelWeight {
  ~ModelWeight() {
    for (auto& kvp : deleter_for_initialized_tensors_) {
      kvp.second.f(kvp.second.param);
    }
  }
  // initialized tensors
  std::unordered_map<int, OrtValue> initialized_tensors_;  // key is ort_value_index
  // subset of initialized_tensors_ that are constant and cannot be overridden at runtime
  std::unordered_map<int, OrtValue> constant_initialized_tensors_;

#if !defined(DISABLE_SPARSE_TENSORS)
  // This is an auxiliary lookup to check if the OrtValue was actually a sparse tensor
  // this is needed because we currently convert all sparse initializer into dense Tensors
  // if and when we actually place SparseTensor instances (we should) into OrtValues, we
  // will not need this structure.
  onnxruntime::InlinedHashSet<int> sparse_initialized_tensors_;
#endif
  // Container to store pre-packed weights to share between sessions.
  // The life-cycle of the cache itself is maintained by the user and the user will ensure
  // the cache is valid until any session reliant on it is still in scope.
  std::unique_ptr<onnxruntime::PrepackedWeightsContainer> prepacked_weights_container_;

  onnxruntime::InlinedHashMap<int, onnxruntime::OrtCallback> deleter_for_initialized_tensors_;

  SubgraphModelWeightMap subgraph_weight_map;
};

struct OrtExecOptions {
  std::unordered_map<std::string, void*> streams_map;
};

class OrtExecutor;

namespace onnxruntime {
class InferenceSession;
}

struct OrtNetwork {
  bool session_inited{false};
  std::shared_ptr<onnxruntime::InferenceSession> session;
  std::unordered_map<std::string, void*> streams_map;
  OrtNetOptions options;
  const OrtEnv* env{nullptr};

  std::mutex init_mutex;
  const OrtExecutor* session_assigned_{nullptr};
  std::vector<uint8_t> model_data;
};

struct OrtExecutor {
  std::shared_ptr<onnxruntime::InferenceSession> session;
};
