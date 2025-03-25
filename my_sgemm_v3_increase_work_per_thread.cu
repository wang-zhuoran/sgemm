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
template <unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void cuda_gemm(float* A_ptr, float* B_ptr, float* C_ptr, const int M, const int N, const int K) {
    // const int x = blockDim.x * blockIdx.x + threadIdx.x;
    // const int y = blockDim.y * blockIdx.y + threadIdx.y;
    constexpr int STEP = BLOCK_SIZE * STRIDE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // if (x >= N || y >= M) return; 
    float* A_ptr_start = A_ptr + STEP * blockIdx.y * K; 
    float* B_ptr_start = B_ptr + STEP * blockIdx.x;

    __shared__ float a_shared[STEP][STEP];
    __shared__ float b_shared[STEP][STEP];

    float temp[STRIDE][STRIDE] = {0.f}; // 每个线程私有STRIDE * STRIDE 个temp临时数据

    for(int s = 0; s < K; s += STEP) {
        for(int i = 0; i < STRIDE; i++) {
            for(int j = 0; j < STRIDE; j++) {
                a_shared[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = A_ptr_start[(ty + i * BLOCK_SIZE) * K + tx + j * BLOCK_SIZE + s];
                b_shared[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = B_ptr_start[(ty + i * BLOCK_SIZE + s) * N + tx + j * BLOCK_SIZE];
            }
        }
        
        __syncthreads();

        for(int i = 0; i < STRIDE; i++) {
            for(int j = 0; j < STRIDE; j++) {
                for(int k = 0; k < STEP; k++) { // tile的大小是STEP*STEP
                    temp[i][j] += a_shared[ty + i * BLOCK_SIZE][k] * b_shared[k][tx + j * BLOCK_SIZE];
                }
            }
        }
        __syncthreads();
    }
    float* C_ptr_start = C_ptr + blockIdx.y * STEP * N + blockIdx.x * STEP; 
    for(int i = 0; i < STRIDE; i++) {
        for(int j = 0; j < STRIDE; j++) {
            C_ptr_start[(ty + i * BLOCK_SIZE) * N + tx + j * BLOCK_SIZE] = temp[i][j];
        }
    }
    
    
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
    constexpr int STRIDE = 2;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK / STRIDE, (n + BLOCK - 1) / BLOCK / STRIDE);
    // cuda_gemm<BLOCK, BLOCK><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cuda_gemm<BLOCK, STRIDE><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
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