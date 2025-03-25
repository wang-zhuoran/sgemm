#include<cstdio>
#include <iostream>
#include <chrono>
#include <cstring>       // for memset

#define A(i, j) a[(i) * n + (j)]
#define B(i, j) b[(i) * n + (j)]
#define abs(x) ((x) < 0.0 ? -(x) : (x))
void random_matrix(int m, int n, float* a) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
#if 1
            A(i, j) = 2.0 * (float)drand48() - 1.0;
#else   
            A(i, j) = (j - i) % 3;
#endif 
        }
    }
}

float compare_matrices(int m, int n, float *a, float *b, int a_cols, int b_cols) {
    int i, j;
    float max_diff = 0.0, diff;
    int printed = 0;
  
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        float a_val = a[i * a_cols + j];  // 正确按 stride 访问
        float b_val = b[i * b_cols + j];
        diff = abs(a_val - b_val);
        max_diff = (diff > max_diff ? diff : max_diff);
        if (0 == printed)
          if (max_diff > 0.5f || max_diff < -0.5f) {
            printf("\n error: i %d  j %d diff %f  got %f  expect %f ", i, j, max_diff, a_val, b_val);
            printed = 1;
          }
      }
    }
  
    return max_diff;
}

// A: [M][K]
// B: [K][N]
// C: [M][N]
void cpu_sgemm(float* A_ptr, float* B_ptr, float* C_ptr, const int M, const int N, const int K) {
    for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
            float temp = 0.0f;
            for(int k = 0; k < K; k++) {
                temp += A_ptr[m * K + k] * B_ptr[k * N + n];
            }
            C_ptr[m * N + n] = temp;
        }
    }
}
// A: [M][K]
// B: [K][N]
// C: [M][N]
template <unsigned int BLOCK_SIZE>
__global__ void cuda_gemm(float* A_ptr, float* B_ptr, float* C_ptr, const int M, const int N, const int K) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= N || y >= M) return; 
    float* A_ptr_start = A_ptr + blockDim.y * blockIdx.y * K; 
    float* B_ptr_start = B_ptr + blockDim.x * blockIdx.x;

    __shared__ float a_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE][BLOCK_SIZE];
    float temp = 0.f; // 每个线程都会私有一个temp
    for(int s = 0; s < K; s += blockDim.x) {
        a_shared[threadIdx.y][threadIdx.x] = A_ptr_start[threadIdx.y * K + threadIdx.x + s];
        b_shared[threadIdx.y][threadIdx.x] = B_ptr_start[(threadIdx.y + s) * N + threadIdx.x];
        __syncthreads();
        for(int k = 0; k < BLOCK_SIZE; k++) {
            temp += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
        }
        __syncthreads();
    }
    C_ptr[x + y * N] = temp; // 在同一时刻，有BLOCK_SIZE * BLOCK_SIZE个线程在做写回操作，这是由dim3 block决定的 
    
}

int main(){
    printf("Hello, SGEMM!\n");

    int m = 512;
    int n = 512;
    int k = 512;

    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);

    float* matrix_A_host = (float*)malloc(mem_size_A);
    float* matrix_B_host = (float*)malloc(mem_size_B);
    float* matrix_C_host_gpu_calc = (float*)malloc(mem_size_C);
    float* matrix_C_host_cpu_calc = (float*)malloc(mem_size_C);
    memset(matrix_A_host, 0, mem_size_A);
    memset(matrix_B_host, 0, mem_size_B);

    random_matrix(m, k, matrix_A_host);
    random_matrix(k, n, matrix_B_host);



    float* matrix_A_device, *matrix_B_device, *matrix_C_device;

    cudaMalloc((void**)&matrix_A_device, mem_size_A);
    cudaMalloc((void**)&matrix_B_device, mem_size_B);
    cudaMalloc((void**)&matrix_C_device, mem_size_C);

    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);

    // cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, m, n, k);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, m, n, k);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    printf("CPU SGEMM time: %.3f ms\n", cpu_duration.count());


    // std::cout << &matrix_C_host_cpu_calc << std::endl;

    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    // cuda_gemm<BLOCK, BLOCK><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cuda_gemm<BLOCK><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaEventRecord(stop);

    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU SGEMM time: %.9f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);

    float diff = compare_matrices(m, n, matrix_C_host_gpu_calc, matrix_C_host_cpu_calc, k, n);

    printf("error: %f \n", diff);

    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_cpu_calc);
    free(matrix_C_host_gpu_calc);

    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);

    return 0;
}