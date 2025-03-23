#include<cstdio>
#include <iostream>
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

float compare_matrices(int m, int n, float *a, float *b) {
    //    printf("\n---result----\n");
    //    print_matrix(m, n, a, lda);
    //    printf("\n-------\n");
    //    print_matrix(m, n, b, ldb);
    //    printf("\n-------\n");
    int i, j;
    float max_diff = 0.0, diff;
    int printed = 0;
  
    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
        diff = abs(A(i, j) - B(i, j));
        max_diff = (diff > max_diff ? diff : max_diff);
        if (0 == printed)
          if (max_diff > 0.5f || max_diff < -0.5f) {
            printf("\n error: i %d  j %d diff %f  got %f  expect %f ", i, j, max_diff, A(i, j), B(i, j));
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
            C_ptr[m * M + n] = temp;
        }
    }
}
// A: [M][K]
// B: [K][N]
// C: [M][N]
__global__ void cuda_gemm(float* A_ptr, float* B_ptr, float* C_ptr, const int M, const int N, const int K) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= N || y >= M) return; 
    float* A_ptr_start = A_ptr + blockDim.y * blockIdx.y * K; 
    // A矩阵的初始位置加上 y方向上走了多少步，乘每一步跨越的元素个数也就是K 
    float* B_ptr_start = B_ptr + blockDim.x * blockIdx.x;
    // B矩阵的初始位置加上x方向上走了多少步
    float temp = 0.f;
    for(int k = 0; k < K; k++) {
        temp += A_ptr_start[threadIdx.y * K + k] * B_ptr_start[k * N + threadIdx.x];
        // A_ptr_start[threadIdx.y * K + k] 在block内部在y方向上走了多少步，乘每一步跨越的元素个数，加上x方向的偏移量k
        // 访问 A 中第 threadIdx.y 行，第 k 列的元素
        // B_ptr_start[k * N + threadIdx.x] 每个 k 对应的是 B 的第 k 行，每一行有 N 个元素
        // 所以 k*N + threadIdx.x 表示访问第 k 行第 threadIdx.x 列的元素，即 B[k][threadIdx.x]
        // 在 block 内部在 x 方向上走了多少步，加上跨越的之前所有元素 k*N
    }

    C_ptr[x + y * N] = temp;
    //纵向每走一步跨越 N 个元素，加上 x 方向的偏移量
}

int main(){
    printf("Hello, SGEMM!\n");

    int m = 16;
    int n = 16;
    int k = 16;

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

    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, m, n, k);


    // std::cout << &matrix_C_host_cpu_calc << std::endl;

    constexpr int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((m + BLOCK - 1) / BLOCK, (n + BLOCK - 1) / BLOCK);
    cuda_gemm<<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, m, n, k);
    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);

    float diff = compare_matrices(m, n, matrix_C_host_gpu_calc, matrix_C_host_cpu_calc);

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