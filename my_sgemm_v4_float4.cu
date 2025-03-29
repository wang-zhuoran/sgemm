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

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// A: [M][K]
// B: [K][N]
// C: [M][N]
template <unsigned int M_NUM_PER_BLOCK, 
        unsigned int N_NUM_PER_BLOCK, 
        unsigned int K_NUM_PER_BLOCK, 
        unsigned int NUM_PER_THREAD>
__global__ void cuda_gemm(float* A_ptr, float* B_ptr, float* C_ptr, const int M, const int N, const int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float* A_ptr_start = A_ptr + blockIdx.y * M_NUM_PER_BLOCK * K;
    float* B_ptr_start = B_ptr + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    float temp[NUM_PER_THREAD] = {0.f};

    for(int s = 0; s < K; s += K_NUM_PER_BLOCK) {
        FETCH_FLOAT4(a_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(A_ptr_start[K * ty + s + tx * NUM_PER_THREAD]);
        // a_shared[ty][tx * NUM_PER_THREAD + 0] = A_ptr_start[K * ty + s + tx * NUM_PER_THREAD + 0];
        // a_shared[ty][tx * NUM_PER_THREAD + 1] = A_ptr_start[K * ty + s + tx * NUM_PER_THREAD + 1];
        // a_shared[ty][tx * NUM_PER_THREAD + 2] = A_ptr_start[K * ty + s + tx * NUM_PER_THREAD + 2];
        // a_shared[ty][tx * NUM_PER_THREAD + 3] = A_ptr_start[K * ty + s + tx * NUM_PER_THREAD + 3];
        FETCH_FLOAT4(b_shared[ty][tx * NUM_PER_THREAD]) = FETCH_FLOAT4(B_ptr_start[(ty + s) * N + tx * NUM_PER_THREAD]);
        // b_shared[ty][tx * NUM_PER_THREAD + 0] = B_ptr_start[(ty + s) * N + tx * NUM_PER_THREAD + 0];
        // b_shared[ty][tx * NUM_PER_THREAD + 1] = B_ptr_start[(ty + s) * N + tx * NUM_PER_THREAD + 1];
        // b_shared[ty][tx * NUM_PER_THREAD + 2] = B_ptr_start[(ty + s) * N + tx * NUM_PER_THREAD + 2];
        // b_shared[ty][tx * NUM_PER_THREAD + 3] = B_ptr_start[(ty + s) * N + tx * NUM_PER_THREAD + 3];
        __syncthreads();
        // 这个循环顺序是否可以改变？
        // for(int i = 0; i < NUM_PER_THREAD; i++) {
        //     for(int k = 0; k < N_NUM_PER_BLOCK; k++) {
        //         temp[i] += a_shared[ty][k] * b_shared[k][tx * NUM_PER_THREAD + i];
        //     }
        // }
        for (int k = 0; k < N_NUM_PER_BLOCK; ++k) {
            float a_val = a_shared[ty][k];
            for (int i = 0; i < NUM_PER_THREAD; ++i) {
                temp[i] += a_val * b_shared[k][tx * NUM_PER_THREAD + i];
            }
        }
        
        __syncthreads();
    }

    float* C_ptr_start = C_ptr + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK; 
    for(int i = 0; i < NUM_PER_THREAD; i++) {
        C_ptr_start[ty * N + tx * NUM_PER_THREAD + i] = temp[i];
        /*
        temp[0] → C[global_row][global_col + 0]
        temp[1] → C[global_row][global_col + 1]
        ...
        temp[i] → C[global_row][global_col + i] （i = 4）
        global_col = blockIdx.x * N_NUM_PER_BLOCK + tx * NUM_PER_THREAD
        */
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

    constexpr int M_NUM_PER_BLOCK = 32;
    constexpr int N_NUM_PER_BLOCK = 32;
    constexpr int K_NUM_PER_BLOCK = 32;
    constexpr int NUM_PER_THREAD = 4;
    dim3 block(8, 32); // 原本是16,16 这里因为横向一次性取4个float所以改为8,32 thread总数仍然为256
    dim3 grid(m / M_NUM_PER_BLOCK, n / N_NUM_PER_BLOCK);
    // cuda_gemm<BLOCK, BLOCK><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cuda_gemm<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, NUM_PER_THREAD><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
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