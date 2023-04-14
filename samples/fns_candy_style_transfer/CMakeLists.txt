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


add_executable(fns_candy_style_transfer "fns_candy_style_transfer.c" "image_file.h")
if(WIN32)
  target_sources(fns_candy_style_transfer PRIVATE image_file_wic.cc)
else()
  target_sources(fns_candy_style_transfer PRIVATE image_file_libpng.c)
endif()
target_include_directories(fns_candy_style_transfer PRIVATE ${PROJECT_SOURCE_DIR}/include ${PNG_INCLUDE_DIRS})
target_link_libraries(fns_candy_style_transfer PRIVATE onnxruntime ${PNG_LIBRARIES} ${WIL_LIB})
if(PNG_LIBDIR)
  target_link_directories(fns_candy_style_transfer PRIVATE ${PNG_LIBDIR})
endif()

# copy_ort_dlls(fns_candy_style_transfer)
