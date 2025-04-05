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

/*

   ← s tile loop (K dimension)
    ┌────────────────────────────┐
    │  load A_tile to smem       │
    │  load B_tile to smem       │
    │  __syncthreads()           │
    │                            │
    │  ← k loop over tile depth  │
    │   ┌─────────────────────┐  │
    │   │  load A col to reg  │  │
    │   │  load B row to reg  │  │
    │   │  temp += A × Bᵀ     │  │ ← outer product
    │   └─────────────────────┘  │
    │                            │
    │  __syncthreads()           │
    └────────────────────────────┘

→ Write back temp[2][2] to global C

*/

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// A: [M][K]
// B: [K][N]
// C: [M][N]
template <unsigned int M_NUM_PER_BLOCK, 
        unsigned int N_NUM_PER_BLOCK, 
        unsigned int K_NUM_PER_BLOCK, 
        unsigned int M_NUM_PER_THREAD,
        unsigned int N_NUM_PER_THREAD,
        unsigned int K_NUM_PER_THREAD>
__global__ void cuda_gemm(float* A_ptr, float* B_ptr, float* C_ptr, const int M, const int N, const int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // int tid = ty * blockDim.x + tx; // 重排
    // int ctx = tid % 16;
    // int cty = tid / 16;
    float* A_ptr_start = A_ptr + blockIdx.y * M_NUM_PER_BLOCK * K;
    float* B_ptr_start = B_ptr + blockIdx.x * N_NUM_PER_BLOCK;

    __shared__ float a_shared[M_NUM_PER_BLOCK][K_NUM_PER_BLOCK];
    __shared__ float b_shared[K_NUM_PER_BLOCK][N_NUM_PER_BLOCK];

    // constexpr int REG_NUM = NUM_PER_THREAD / 2;
    float a_reg[M_NUM_PER_THREAD] = {0.f};
    float b_reg[N_NUM_PER_THREAD] = {0.f};
    float a_load_reg[K_NUM_PER_THREAD] = {0.f}; // 用于做转秩的寄存器
    float temp[M_NUM_PER_THREAD][N_NUM_PER_THREAD] = {0.f}; // 这里是因为外积会生成一个M_NUM_PER_THREAD * N_NUM_PER_THREAD的矩阵，然后累加这个矩阵才能得到最后的结果
    

    for(int s = 0; s < K; s += K_NUM_PER_BLOCK) {
    // 先从global mem 搬运到shared mem
    /*
    【配置 A】
  tile size = 64×64 = 4096 float
  thread count = 256
  每 thread 只能一次加载 4 → 不够 → ❗要用 for 加载多次

【配置 B】
  tile size = 32×32 = 1024 float
  thread count = 256
  每 thread 加载 4 → 总量刚好够 → ✅ 不需要 for

    */        
        for(int i = 0; i < M_NUM_PER_THREAD; i++) {
            // 先把4个数从global mem中移动到寄存器中
            FETCH_FLOAT4(a_load_reg[0]) = 
                FETCH_FLOAT4(A_ptr_start[K * (ty * M_NUM_PER_THREAD + i) + tx * K_NUM_PER_THREAD + s]);
            // 然后将register中的数据存进 shared mem
            // 注意这里转秩了，所以原来的y索引变为了x索引
            a_shared[tx * K_NUM_PER_THREAD][ty * M_NUM_PER_THREAD + i] = a_load_reg[0];
            a_shared[tx * K_NUM_PER_THREAD + 1][ty * M_NUM_PER_THREAD + i] = a_load_reg[1];
            a_shared[tx * K_NUM_PER_THREAD + 2][ty * M_NUM_PER_THREAD + i] = a_load_reg[2];
            a_shared[tx * K_NUM_PER_THREAD + 3][ty * M_NUM_PER_THREAD + i] = a_load_reg[3];

        }
        for(int i = 0; i < K_NUM_PER_THREAD; i++) {
            FETCH_FLOAT4(b_shared[ty * K_NUM_PER_THREAD + i][tx * N_NUM_PER_THREAD]) = 
                FETCH_FLOAT4(B_ptr_start[N * (ty * K_NUM_PER_THREAD + i + s) + tx * N_NUM_PER_THREAD]);
        }
        __syncthreads();
 
        for(int k = 0; k < K_NUM_PER_BLOCK; k++) {
            // a_reg[0] = a_shared[ty * M_NUM_PER_THREAD][k];
            // a_reg[1] = a_shared[ty * M_NUM_PER_THREAD + 1][k];
            // a_reg[2] = a_shared[ty * M_NUM_PER_THREAD + 2][k];
            // a_reg[3] = a_shared[ty * M_NUM_PER_THREAD + 3][k];
            FETCH_FLOAT4(a_reg[0]) = FETCH_FLOAT4(a_shared[k][ty * N_NUM_PER_THREAD]);
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(b_shared[k][tx * N_NUM_PER_THREAD]);
            for(int i = 0; i < M_NUM_PER_THREAD; i++) {
                for(int j = 0; j < N_NUM_PER_THREAD; j++) {
                    temp[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    float* C_ptr_start = C_ptr + blockIdx.y * M_NUM_PER_BLOCK * N + blockIdx.x * N_NUM_PER_BLOCK; 
    // for(int i = 0; i < M_NUM_PER_THREAD; i++) {
    //     for(int j = 0; j < N_NUM_PER_THREAD; j++) {
    //         C_ptr_start[N * (ty * M_NUM_PER_THREAD + i) + tx * N_NUM_PER_THREAD + j] = temp[i][j];
    //     }
    // }
    for(int i = 0; i < M_NUM_PER_THREAD; i++) {
        FETCH_FLOAT4(C_ptr_start[N * (ty * M_NUM_PER_THREAD + i) + tx * N_NUM_PER_THREAD]) = FETCH_FLOAT4(temp[i][0]);
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

    constexpr int M_NUM_PER_BLOCK = 64;
    constexpr int N_NUM_PER_BLOCK = 64;
    constexpr int K_NUM_PER_BLOCK = 64;
    constexpr int NUM_PER_THREAD = 16; // 每个线程负责多少个数字
    constexpr int M_NUM_PER_THREAD = 4; // 用于寄存器大小的分配
    constexpr int N_NUM_PER_THREAD = 4; // 用于寄存器大小的分配
    constexpr int K_NUM_PER_THREAD = 4; // thread 在k维度上的stride
    dim3 block(16, 16); 
    dim3 grid(m / M_NUM_PER_BLOCK, n / N_NUM_PER_BLOCK);
    // cuda_gemm<BLOCK, BLOCK><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cuda_gemm<M_NUM_PER_BLOCK, N_NUM_PER_BLOCK, K_NUM_PER_BLOCK, M_NUM_PER_THREAD, N_NUM_PER_THREAD, K_NUM_PER_THREAD><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
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