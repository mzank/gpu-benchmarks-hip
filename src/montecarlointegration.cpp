/**
 * @file montecarlointegration.cpp
 * @brief Demonstrates Monte Carlo integration on CPU and GPU using HIP.
 * 
 * Example output (measured on 1 node with 4 AMD MI300A APUs):
 * \code
 * GPU config: 14592 blocks x 256 threads
 * GPU result: -0.00378359 in 0.0204081 s
 * CPU result: -0.00378631 in 0.683242 s
 * \endcode
 *
 * Hardware and Software Environment:
 * - ROCm 7.1.1
 *
 * This example estimates a three-dimensional integral of a smooth,
 * oscillatory function over the unit cube [0,1]^3 using Monte Carlo
 * sampling on both CPU and GPU.
 *
 * The GPU implementation uses a HIP kernel with hipRAND for
 * random number generation, while the CPU implementation uses
 * C++ standard library facilities with parallel execution.
 *
 * Demonstrates:
 * - Monte Carlo integration using random sampling
 * - Parallel CPU execution with C++17 parallel algorithms
 * - GPU kernel launch with HIP
 * - Random number generation on GPU using hipRAND
 * - Atomic accumulation of partial results on the GPU
 * - Performance comparison between CPU and GPU implementations
 *
 * @note Requires HIP and hipRAND.
 *
 * @author Marco Zank
 * @date 2025-12-18
 */

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

#include <iostream>
#include <vector>
#include <numeric>
#include <execution>
#include <random>
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

// ============================================================
// Constants
// ============================================================

/**
 * @brief Total number of Monte Carlo samples.
 */
constexpr std::size_t N = 1'000'000'000;  // 1 billion samples

/**
 * @brief Number of threads per GPU block.
 */
constexpr int THREADS_PER_BLOCK = 256;

/**
 * @brief Number of samples processed per CPU iteration.
 *
 * Used to amortize random number generation overhead.
 */
constexpr std::size_t CPU_SAMPLES_PER_ITER = 16;

// ============================================================
// Integrand definition
// ============================================================

/**
 * @brief Function to be integrated over the unit cube [0,1]^3.
 *
 * This function is evaluated on both CPU and GPU.
 *
 * @param x First coordinate in [0,1]
 * @param y Second coordinate in [0,1]
 * @param z Third coordinate in [0,1]
 * @return Value of the integrand at (x, y, z)
 */
__host__ __device__ inline double f(double x, double y, double z) {
    return std::exp(-(x * x + y * y + z * z))
         * std::sin(5.0 * x)
         * std::cos(5.0 * y)
         * std::sin(5.0 * z);
}

// ============================================================
// CPU Monte Carlo implementation
// ============================================================

/**
 * @brief Monte Carlo integration on the CPU.
 *
 * Uses C++17 parallel algorithms to distribute work across
 * CPU threads. Each thread maintains its own random number
 * generator.
 *
 * @param num_samples Total number of Monte Carlo samples
 * @param samples_per_iter Number of samples computed per iteration
 * @return Estimated integral value
 */
static double monteCarloCPU(std::size_t num_samples,
                            std::size_t samples_per_iter) {
    const std::size_t num_chunks =
        (num_samples + samples_per_iter - 1) / samples_per_iter;

    std::vector<std::size_t> chunks(num_chunks);
    std::iota(chunks.begin(), chunks.end(), 0);

    double sum = std::transform_reduce(
        std::execution::par,
        chunks.begin(), chunks.end(),
        0.0,
        std::plus<double>(),
        [samples_per_iter](std::size_t) {
            thread_local std::mt19937 rng(std::random_device{}());
            thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);

            double local_sum = 0.0;
            for (std::size_t i = 0; i < samples_per_iter; ++i) {
                local_sum += f(dist(rng), dist(rng), dist(rng));
            }
            return local_sum;
        }
    );

    return sum / static_cast<double>(num_samples);
}

// ============================================================
// GPU Monte Carlo kernel
// ============================================================

/**
 * @brief GPU kernel for Monte Carlo integration.
 *
 * Each thread generates random samples using hipRAND and
 * accumulates a partial sum, which is added atomically
 * to the global result.
 *
 * @param result Pointer to global result accumulator
 * @param num_samples Total number of Monte Carlo samples
 * @param seed Random seed for hipRAND
 */
static __global__ void monteCarloGPU(double* result,
                                     std::size_t num_samples,
                                     unsigned long long seed) {
    const std::size_t idx =
        blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride =
        gridDim.x * blockDim.x;

    hiprandStatePhilox4_32_10_t state;
    hiprand_init(seed, idx, 0, &state);

    double local_sum = 0.0;
    for (std::size_t i = idx; i < num_samples; i += stride) {
        local_sum += f(
            hiprand_uniform_double(&state),
            hiprand_uniform_double(&state),
            hiprand_uniform_double(&state)
        );
    }

    atomicAdd(result, local_sum);
}

// ============================================================
// Main function
// ============================================================

/**
 * @brief Main function performing CPU and GPU Monte Carlo integration.
 *
 * Execution steps:
 * 1. Allocate and initialize GPU memory.
 * 2. Launch GPU Monte Carlo kernel and measure execution time.
 * 3. Copy GPU result back to host and normalize.
 * 4. Perform CPU Monte Carlo integration and measure execution time.
 * 5. Print results and timing information.
 *
 * @return int Returns 0 on successful execution.
 */
int main() {
    // ============================================================
    // GPU Monte Carlo
    // ============================================================

    double* d_result = nullptr;
    HIP_CHECK(hipMalloc(&d_result, sizeof(double)));
    HIP_CHECK(hipMemset(d_result, 0, sizeof(double)));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    const std::size_t threads = THREADS_PER_BLOCK;
    const std::size_t blocks = std::min(
        (N + threads - 1) / threads,
        static_cast<std::size_t>(prop.multiProcessorCount) * 64
    );

    std::cout << "GPU config: "
              << blocks << " blocks x "
              << threads << " threads\n";

    auto gpu_start = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(
        monteCarloGPU,
        dim3(static_cast<unsigned int>(blocks)),
        dim3(static_cast<unsigned int>(threads)),
        0, 0,
        d_result, N, 1234ULL
    );

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    double gpu_sum = 0.0;
    HIP_CHECK(hipMemcpy(&gpu_sum, d_result,
                        sizeof(double),
                        hipMemcpyDeviceToHost));

    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time_s =
        std::chrono::duration<double>(gpu_end - gpu_start).count();

    std::cout << "GPU result: "
              << gpu_sum / static_cast<double>(N)
              << " in " << gpu_time_s << " s\n";

    HIP_CHECK(hipFree(d_result));

    // ============================================================
    // CPU Monte Carlo
    // ============================================================

    auto cpu_start = std::chrono::high_resolution_clock::now();
    double cpu_result = monteCarloCPU(N, CPU_SAMPLES_PER_ITER);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_time_s =
        std::chrono::duration<double>(cpu_end - cpu_start).count();

    std::cout << "CPU result: "
              << cpu_result
              << " in " << cpu_time_s << " s\n";

    return EXIT_SUCCESS;
}
