/**
 * @file vectorreduction.cpp
 * @brief Demonstrates parallel reduction (sum) of a large vector
 *        on CPU and GPU using HIP.
 *
 * Example output (measured on 1 node with 4 AMD MI300A APUs):
 * \code
 * ==================== Results ====================
 * sum_CPU: 1.07374e+09, time: 49.8159 ms
 * sum_GPU: 1.07374e+09, time: 4.55739 ms
 * |sum_CPU - sum_GPU| = 0
 * \endcode
 *
 * Hardware and Software Environment:
 * - ROCm 7.1.1
 *
 * This example initializes a large vector of doubles, computes its sum
 * on the CPU using parallel STL algorithms, and on the GPU using a
 * custom HIP reduction kernel. It then compares CPU and GPU results
 * for correctness and measures execution times.
 *
 * Demonstrates:
 * - Memory allocation and initialization on host (CPU)
 * - CPU reduction using parallel STL
 * - GPU memory allocation and data transfer
 * - GPU block reduction kernel launch
 * - Iterative GPU reduction until final sum
 * - Performance measurement using high-resolution timers
 * - Validation by computing the maximum absolute difference between CPU and GPU results
 *
 * @note Requires HIP installation.
 *
 * @author Marco Zank
 * @date 2025-12-15
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <execution>
#include <chrono>
#include <cmath>

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
 * @brief Size of the vector to reduce.
 */
constexpr size_t ARRAY_SIZE = 1ULL << 30;

/**
 * @brief Number of threads per GPU block.
 */
constexpr uint32_t THREADS_PER_BLOCK = 256;

/**
 * @brief GPU kernel performing block-level reduction.
 *
 * Each block reduces 2*blockDim.x elements from the input vector
 * into a single sum per block stored in the output array.
 *
 * @param input Pointer to input vector on GPU.
 * @param output Pointer to output array storing partial sums.
 * @param size Number of elements in input vector.
 */
static __global__ void block_reduce(const double* input,
                             double* output,
                             size_t size)
{
    __shared__ double sdata[THREADS_PER_BLOCK];

    const unsigned int tid = threadIdx.x;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x * 2 + tid;

    double sum = 0.0;
    if (idx < size) sum += input[idx];
    if (idx + blockDim.x < size) sum += input[idx + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

/**
 * @brief Main function demonstrating CPU and GPU reduction.
 *
 * Performs the following steps:
 * 1. Allocates and initializes vector on host.
 * 2. Computes sum on CPU using parallel STL.
 * 3. Allocates GPU memory and copies vector.
 * 4. Computes sum on GPU using block reduction kernel.
 * 5. Iteratively reduces partial sums until final GPU result.
 * 6. Validates GPU result against CPU result.
 * 7. Reports execution times and maximum absolute error.
 *
 * @return int Returns 0 on successful execution.
 */
int main()
{
    // -------------------------
    // Host data allocation
    // -------------------------
    std::vector<double> h_data(ARRAY_SIZE, 1.0); /**< Host vector initialized to 1.0 */

    // -------------------------
    // CPU reduction
    // -------------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    const double cpu_sum =
        std::reduce(std::execution::par,
                    h_data.begin(), h_data.end(), 0.0);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    const double cpu_time_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // -------------------------
    // GPU memory allocation
    // -------------------------
    double* d_data = nullptr;
    double* d_partial = nullptr;

    HIP_CHECK(hipMalloc(&d_data, ARRAY_SIZE * sizeof(double)));

    const size_t blocks = (ARRAY_SIZE + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);
    HIP_CHECK(hipMalloc(&d_partial, blocks * sizeof(double)));
    HIP_CHECK(hipMemcpy(d_data, h_data.data(),
                        ARRAY_SIZE * sizeof(double),
                        hipMemcpyHostToDevice));

    // -------------------------
    // GPU reduction
    // -------------------------
    HIP_CHECK(hipDeviceSynchronize());
    auto gpu_start = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(
        block_reduce,
        dim3(static_cast<uint32_t>(blocks)),
        dim3(THREADS_PER_BLOCK),
        0, 0,
        d_data, d_partial, ARRAY_SIZE);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    size_t s = blocks;
    while (s > 1) {
        const size_t next_blocks = (s + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);

        hipLaunchKernelGGL(
            block_reduce,
            dim3(static_cast<uint32_t>(next_blocks)),
            dim3(THREADS_PER_BLOCK),
            0, 0,
            d_partial, d_partial, s);

        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        s = next_blocks;
    }

    double gpu_sum = 0.0;
    HIP_CHECK(hipMemcpy(&gpu_sum, d_partial, sizeof(double), hipMemcpyDeviceToHost));

    auto gpu_end = std::chrono::high_resolution_clock::now();
    const double gpu_time_ms =
        std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    // -------------------------
    // Validation
    // -------------------------
    const double max_abs_diff = std::abs(cpu_sum - gpu_sum);

    // -------------------------
    // Print results
    // -------------------------
    std::cout << "==================== Results ====================\n";
    std::cout << "sum_CPU: " << cpu_sum
              << ", time: " << cpu_time_ms << " ms\n";
    std::cout << "sum_GPU: " << gpu_sum
              << ", time: " << gpu_time_ms << " ms\n";
    std::cout << "|sum_CPU - sum_GPU| = " << max_abs_diff << "\n";

    // -------------------------
    // Cleanup GPU resources
    // -------------------------
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_partial));

    return 0;
}
