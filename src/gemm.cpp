/**
 * @file gemm.cpp
 * @brief Demonstrates DGEMM (double-precision general matrix-matrix multiplication)
 *        on CPU (using BLAS) and GPU (using hipBLAS) with performance comparison.
 *
 * Example output (measured on 1 AMD MI300A APU):
 * \code
 * ==================== Results ====================
 * CPU DGEMM time: 66171.6 ms
 * GPU hipBLAS DGEMM time: 1940.62 ms
 * Maximum |C_cpu - C_gpu| = 2.20098e-10
 * \endcode
 *
 * Hardware and Software Environment:
 * - ROCm 7.1.1
 * - OpenBLAS 0.3.20
 *
 * This example initializes two large square matrices with random values, computes
 * their product on the CPU using a BLAS library (e.g. OpenBLAS), and on the GPU using hipBLAS.
 * It then compares CPU and GPU results to verify correctness and measures execution time.
 *
 * Demonstrates:
 * - Memory allocation and initialization on host (CPU)
 * - CPU BLAS DGEMM computation
 * - GPU memory allocation and data transfer
 * - GPU DGEMM computation using hipBLAS
 * - Performance measurement using high-resolution timers
 * - Validation by computing the maximum absolute difference between CPU and GPU results
 *
 * @note Requires HIP and hipBLAS installation, and a CPU BLAS library (e.g., OpenBLAS or MKL).
 *
 * @author Marco Zank
 * @date 2025-12-12
 */

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <execution>
#include <numeric>
#include <chrono>
#include <cmath>

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

/**
 * @brief External CPU BLAS DGEMM routine.
 *
 * Computes C = alpha * A * B + beta * C
 *
 * @param transa  'N' for no transpose, 'T' for transpose of matrix A
 * @param transb  'N' for no transpose, 'T' for transpose of matrix B
 * @param m       Number of rows of matrices A and C
 * @param n       Number of columns of matrices B and C
 * @param k       Number of columns of A and rows of B
 * @param alpha   Scalar multiplier for the matrix product
 * @param A       Pointer to matrix A in column-major order
 * @param lda     Leading dimension of A
 * @param B       Pointer to matrix B in column-major order
 * @param ldb     Leading dimension of B
 * @param beta    Scalar multiplier for matrix C
 * @param C       Pointer to matrix C in column-major order (input and output)
 * @param ldc     Leading dimension of C
 */
extern "C" {
    void dgemm_(const char* transa, const char* transb,
                const int* m, const int* n, const int* k,
                const double* alpha,
                const double* A, const int* lda,
                const double* B, const int* ldb,
                const double* beta,
                double* C, const int* ldc);
}

/**
 * @brief Macro to check HIP runtime API errors.
 *
 * Prints the error message and exits if the HIP function fails.
 */
#define HIP_CHECK(status)                                         \
    {                                                             \
        hipError_t err = status;                                  \
        if (err != hipSuccess) {                                  \
            std::cerr << "HIP Error: " << hipGetErrorString(err)  \
                      << " at line " << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                              \
        }                                                         \
    }

/**
 * @brief Macro to check hipBLAS API errors.
 *
 * Prints an error message and exits if the hipBLAS function fails.
 */
#define HIPBLAS_CHECK(status)                                     \
    {                                                             \
        hipblasStatus_t err = status;                             \
        if (err != HIPBLAS_STATUS_SUCCESS) {                      \
            std::cerr << "hipBLAS Error at line " << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                              \
        }                                                         \
    }

/**
 * @brief Main function demonstrating CPU and GPU DGEMM.
 *
 * Performs the following steps:
 * 1. Allocates and initializes matrices on the host.
 * 2. Performs CPU DGEMM using BLAS.
 * 3. Allocates GPU memory and copies matrices.
 * 4. Performs GPU DGEMM using hipBLAS.
 * 5. Transfers results back to host.
 * 6. Validates GPU result against CPU result.
 * 7. Reports execution times and maximum absolute error.
 *
 * @return int Returns 0 on successful execution.
 */
int main() {
    // -------------------------
    // Matrix size and memory
    // -------------------------
    /**
     * @brief Dimension of square matrices.
     */
    constexpr int N = 32768;

    /**
     * @brief Total memory in bytes for one N x N double matrix.
     */
    const size_t MATRIX_BYTES = N * N * sizeof(double);

    // -------------------------
    // Host matrices
    // -------------------------
    std::vector<double> h_matrixA(N * N);         /**< Input matrix A on host */
    std::vector<double> h_matrixB(N * N);         /**< Input matrix B on host */
    std::vector<double> h_matrixC_cpu(N * N, 0.0); /**< Output matrix C computed on CPU */
    std::vector<double> h_matrixC_gpu(N * N, 0.0); /**< Output matrix C computed on GPU */

    // -------------------------
    // Initialize matrices with random values in parallel
    // -------------------------
    std::random_device rd;

    /**
     * @brief Lambda function to fill a double value with a random number [0,1].
     *
     * Uses thread-local generators to allow parallel execution safely.
     */
    auto fill_random = [&](double& value) {
        thread_local std::mt19937 gen(rd() + std::hash<std::thread::id>{}(std::this_thread::get_id()));
        thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
        value = dist(gen);
    };

    // Fill matrices in parallel
    std::for_each(std::execution::par, h_matrixA.begin(), h_matrixA.end(), fill_random);
    std::for_each(std::execution::par, h_matrixB.begin(), h_matrixB.end(), fill_random);

    const double alpha = 1.0; /**< Scalar multiplier for matrix product */
    const double beta  = 0.0; /**< Scalar multiplier for existing C */

    // ============================================================
    // CPU DGEMM (using BLAS)
    // ============================================================
    /**
     * @brief Compute matrix multiplication on CPU: C_cpu = alpha * A * B + beta * C
     *
     * Uses the external dgemm_ BLAS function.
     * Measures execution time.
     */
    auto cpu_start = std::chrono::high_resolution_clock::now();
    dgemm_("N", "N", &N, &N, &N,
           &alpha,
           h_matrixA.data(), &N,
           h_matrixB.data(), &N,
           &beta,
           h_matrixC_cpu.data(), &N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ============================================================
    // Allocate GPU memory
    // ============================================================
    /**
     * @brief Allocate GPU device memory for matrices A, B, and C
     *        and copy host matrices A and B to GPU.
     */
    double *d_matrixA = nullptr, *d_matrixB = nullptr, *d_matrixC = nullptr;

    HIP_CHECK(hipMalloc(&d_matrixA, MATRIX_BYTES));
    HIP_CHECK(hipMalloc(&d_matrixB, MATRIX_BYTES));
    HIP_CHECK(hipMalloc(&d_matrixC, MATRIX_BYTES));

    HIP_CHECK(hipMemcpy(d_matrixA, h_matrixA.data(), MATRIX_BYTES, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_matrixB, h_matrixB.data(), MATRIX_BYTES, hipMemcpyHostToDevice));

    // ============================================================
    // GPU DGEMM (hipBLAS)
    // ============================================================
    /**
     * @brief Compute matrix multiplication on GPU using hipBLAS:
     *        C_gpu = alpha * A * B + beta * C
     *
     * Creates a hipBLAS handle, performs DGEMM, and measures execution time.
     */
    hipblasHandle_t handle;
    HIPBLAS_CHECK(hipblasCreate(&handle));

    HIP_CHECK(hipDeviceSynchronize());
    auto gpu_start = std::chrono::high_resolution_clock::now();

    HIPBLAS_CHECK(hipblasDgemm(handle,
                               HIPBLAS_OP_N, HIPBLAS_OP_N,
                               N, N, N,
                               &alpha,
                               d_matrixA, N,
                               d_matrixB, N,
                               &beta,
                               d_matrixC, N));

    HIP_CHECK(hipDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    HIP_CHECK(hipMemcpy(h_matrixC_gpu.data(), d_matrixC, MATRIX_BYTES, hipMemcpyDeviceToHost));

    // Cleanup GPU resources
    HIPBLAS_CHECK(hipblasDestroy(handle));
    HIP_CHECK(hipFree(d_matrixA));
    HIP_CHECK(hipFree(d_matrixB));
    HIP_CHECK(hipFree(d_matrixC));

    // ============================================================
    // Compute maximum absolute difference between CPU and GPU results
    // ============================================================
    /**
     * @brief Validate GPU computation by comparing to CPU result.
     *
     * Computes max(|C_cpu[i] - C_gpu[i]|) over all elements.
     */
    double max_abs_diff = std::transform_reduce(
        std::execution::par,
        h_matrixC_cpu.begin(), h_matrixC_cpu.end(),
        h_matrixC_gpu.begin(),
        0.0,
        [](double x, double y) { return std::max(x, y); },
        [](double a, double b) { return std::abs(a - b); }
    );

    // ============================================================
    // Print performance and validation results
    // ============================================================
    std::cout << "==================== Results ====================\n";
    std::cout << "CPU DGEMM time: " << cpu_time_ms << " ms\n";
    std::cout << "GPU hipBLAS DGEMM time: " << gpu_time_ms << " ms\n";
    std::cout << "Maximum |C_cpu - C_gpu| = " << max_abs_diff << "\n";

    return EXIT_SUCCESS;
}
