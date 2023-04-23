# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB onnxruntime_inference_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_INCLUDE_DIR}/core/inference/*.h"
    "${ONNXRUNTIME_ROOT}/core/inference/*.h"
    "${ONNXRUNTIME_ROOT}/core/inference/*.cc"
    )

if (onnxruntime_MINIMAL_BUILD)
  set(onnxruntime_inference_src_exclude
    "${ONNXRUNTIME_ROOT}/core/inference/provider_bridge_ort.cc"
  )

  list(REMOVE_ITEM onnxruntime_inference_srcs ${onnxruntime_inference_src_exclude})
endif()

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_inference_srcs})

onnxruntime_add_static_library(onnxruntime_inference ${onnxruntime_inference_srcs})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/inference  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
onnxruntime_add_include_to_target(onnxruntime_inference onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers Boost::mp11 safeint_interface nlohmann_json::nlohmann_json)
if(onnxruntime_ENABLE_INSTRUMENT)
  target_compile_definitions(onnxruntime_inference PUBLIC ONNXRUNTIME_ENABLE_INSTRUMENT)
endif()

if(NOT MSVC)
  set_source_files_properties(${ONNXRUNTIME_ROOT}/core/inference/environment.cc PROPERTIES COMPILE_FLAGS  "-Wno-parentheses")
endif()
target_include_directories(onnxruntime_inference PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
if (onnxruntime_USE_EXTENSIONS)
  target_link_libraries(onnxruntime_inference PRIVATE onnxruntime_extensions)
else()
target_link_libraries(onnxruntime_inference PUBLIC onnxruntime_session)
endif()
add_dependencies(onnxruntime_inference ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_inference PROPERTIES FOLDER "ONNXRuntime")
if (onnxruntime_USE_CUDA)
  target_include_directories(onnxruntime_inference PRIVATE ${onnxruntime_CUDNN_HOME}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
if (onnxruntime_USE_ROCM)
  target_compile_options(onnxruntime_inference PRIVATE -Wno-sign-compare -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1)
  target_include_directories(onnxruntime_inference PRIVATE ${onnxruntime_ROCM_HOME}/hipfft/include ${onnxruntime_ROCM_HOME}/include ${onnxruntime_ROCM_HOME}/hipcub/include ${onnxruntime_ROCM_HOME}/hiprand/include ${onnxruntime_ROCM_HOME}/rocrand/include)
# ROCM provider sources are generated, need to add include directory for generated headers
  target_include_directories(onnxruntime_inference PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/onnxruntime ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/orttraining)
endif()
if (onnxruntime_ENABLE_TRAINING_OPS)
  target_include_directories(onnxruntime_inference PRIVATE ${ORTTRAINING_ROOT})
endif()

if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  onnxruntime_add_include_to_target(onnxruntime_inference Python::Module)
endif()

if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_inference
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if (onnxruntime_USE_NCCL AND onnxruntime_USE_ROCM)
  add_dependencies(onnxruntime_inference generate_hipified_files)
endif()
