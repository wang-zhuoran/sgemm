cmake_minimum_required(VERSION 3.18)

project(MySGEMMProject LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(my_sgemm_v0_global_mem my_sgemm_v0_global_mem.cu)

target_link_libraries(my_sgemm_v0_global_mem PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v0_global_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


add_executable(my_sgemm_v1_shared_mem my_sgemm_v1_shared_mem.cu)

target_link_libraries(my_sgemm_v1_shared_mem PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v1_shared_mem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_sgemm_v2_shared_mem_tiling my_sgemm_v2_shared_mem_tiling.cu)

target_link_libraries(my_sgemm_v2_shared_mem_tiling PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v2_shared_mem_tiling PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


add_executable(my_sgemm_v3_increase_work_per_thread my_sgemm_v3_increase_work_per_thread.cu)

target_link_libraries(my_sgemm_v3_increase_work_per_thread PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v3_increase_work_per_thread PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_sgemm_v4_float4 my_sgemm_v4_float4.cu)

target_link_libraries(my_sgemm_v4_float4 PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v4_float4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()



add_executable(my_sgemm_v5_register_with_outer_product my_sgemm_v5_register_with_outer_product.cu)

target_link_libraries(my_sgemm_v5_register_with_outer_product PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v5_register_with_outer_product PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_sgemm_v6_register_with_outer_product_float4 my_sgemm_v6_register_with_outer_product_float4.cu)

target_link_libraries(my_sgemm_v6_register_with_outer_product_float4 PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v6_register_with_outer_product_float4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


add_executable(my_sgemm_v7_smem_transpose my_sgemm_v7_smem_transpose.cu)

target_link_libraries(my_sgemm_v7_smem_transpose PRIVATE CUDA::cudart CUDA::cublas)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_sgemm_v7_smem_transpose PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()


