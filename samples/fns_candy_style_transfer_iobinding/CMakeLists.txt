# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# On Linux the samples use libjpeg and libpng for decoding images.
# On Windows they use Windows Image Component(WIC)
if(NOT WIN32)
    find_package(JPEG)
    if(LIBPNG_ROOTDIR)
      set(PNG_FOUND true)
      set(PNG_LIBRARIES png16)
      set(PNG_INCLUDE_DIRS "${LIBPNG_ROOTDIR}/include")
      set(PNG_LIBDIR "${LIBPNG_ROOTDIR}/lib")
    else()
      find_package(PNG)
    endif()
endif()

if(onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()
if(onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT)
endif()
if(onnxruntime_USE_DML)
  message("Enabling DML")
  add_definitions(-DUSE_DML)
endif()

find_package(CUDA REQUIRED)

add_executable(fns_candy_style_transfer_iobinding "fns_candy_style_transfer.cc" "image_file.h")
if(WIN32)
  target_sources(fns_candy_style_transfer_iobinding PRIVATE image_file_wic.cc)
else()
  target_sources(fns_candy_style_transfer_iobinding PRIVATE image_file_libpng.c)
endif()
target_include_directories(fns_candy_style_transfer_iobinding PRIVATE ${PROJECT_SOURCE_DIR}/include ${PNG_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(fns_candy_style_transfer_iobinding PRIVATE onnxruntime ${PNG_LIBRARIES} ${WIL_LIB} ${CUDA_CUDART_LIBRARY})
if(PNG_LIBDIR)
  target_link_directories(fns_candy_style_transfer_iobinding PRIVATE ${PNG_LIBDIR})
endif()

# copy_ort_dlls(fns_candy_style_transfer_iobinding)
