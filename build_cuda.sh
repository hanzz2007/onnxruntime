#!/bin/bash

./build.sh --config Debug --skip_submodule_sync  --parallel --skip_tests --disable_ml_ops \
           --cmake_path /bin/cmake --use_cuda --cuda_home /usr/local/cuda-11.8 \
           --cudnn_home /usr/local/cudnn-linux-x86_64-8.8.1.3_cuda11-archive --build_shared_lib \
           --cmake_extra_defines CMAKE_EXPORT_COMPILE_COMMANDS=ON
