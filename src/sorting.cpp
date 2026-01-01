/**
 * @file sorting.cpp
 * @brief Demonstrates large-scale integer sorting on CPU and GPU using HIP.
 *
 * Example output (measured on 1 AMD MI300A APU):
 * \code
 * Results match: YES
 * CPU parallel sort time: 3181.42 ms
 * GPU hipCUB sort time:  41.4023 ms
 * \endcode
 *
 * Hardware and Software Environment:
 * - ROCm 7.1.1
 *
 * This example generates a large array of random integers directly
 * on the GPU using hipRAND, sorts the data on the GPU using hipCUB’s
 * radix sort, and compares the result with a parallel CPU sort
 * using C++17 execution policies.
 *
 * Demonstrates:
 * - Random number generation on the GPU using hipRAND
 * - In-place radix sort on the GPU using hipCUB
 * - Parallel sorting on the CPU with C++17
 * - Data transfers between host and device
 * - Performance comparison between CPU and GPU sorting
 *
 * @note Requires HIP, hipRAND, and hipCUB.
 *
 * @author Marco Zank
 * @date 2026-01-02
 */

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <hiprand/hiprand.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
#include <chrono>
#include <cstdlib>

// ============================================================
// Error checking macros
// ============================================================

/**
 * @brief Macro to check HIP runtime API errors.
 *
 * Prints the error message and exits if the HIP function fails.
 */
#define HIP_CHECK(status)                                         \
    {                                                             \
        hipError_t err = status;                                  \
        if (err != hipSuccess) {                                  \
            std::cerr << "HIP Error: "                             \
                      << hipGetErrorString(err)                   \
                      << " at line " << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                              \
        }                                                         \
    }

/**
 * @brief Macro to check hipRAND API errors.
 *
 * Prints the error code and exits if the hipRAND function fails.
 */
#define HIPRAND_CHECK(status)                                     \
    {                                                             \
        hiprandStatus_t err = status;                             \
        if (err != HIPRAND_STATUS_SUCCESS) {                      \
            std::cerr << "hipRAND Error code " << err             \
                      << " at line " << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                              \
        }                                                         \
    }

// ============================================================
// Constants
// ============================================================

/**
 * @brief Number of elements to be sorted.
 *
 * Approximately one billion integers.
 */
constexpr std::size_t N = 1ULL << 30;

// ============================================================
// Main function
// ============================================================

/**
 * @brief Main function performing CPU and GPU sorting.
 *
 * Execution steps:
 * 1. Allocate GPU memory.
 * 2. Generate random integers on the GPU using hipRAND.
 * 3. Copy data to the host for CPU sorting.
 * 4. Sort data on the GPU using hipCUB radix sort.
 * 5. Sort data on the CPU using C++17 parallel sort.
 * 6. Verify correctness and print timing results.
 *
 * @return int Returns EXIT_SUCCESS on successful execution.
 */
int main() {
    // ============================================================
    // Allocate GPU memory
    // ============================================================

    int* d_data = nullptr;
    HIP_CHECK(hipMalloc(&d_data, N * sizeof(int)));

    // ============================================================
    // Generate random integers on GPU using hipRAND
    // ============================================================

    hiprandGenerator_t generator;
    HIPRAND_CHECK(
        hiprandCreateGenerator(&generator, HIPRAND_RNG_PSEUDO_DEFAULT)
    );
    HIPRAND_CHECK(
        hiprandSetPseudoRandomGeneratorSeed(generator, 12345ULL)
    );

    HIPRAND_CHECK(
        hiprandGenerate(
            generator,
            reinterpret_cast<unsigned int*>(d_data),
            N
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    // ============================================================
    // Copy data to host for CPU sorting
    // ============================================================

    std::vector<int> h_cpu(N);
    HIP_CHECK(
        hipMemcpy(
            h_cpu.data(),
            d_data,
            N * sizeof(int),
            hipMemcpyDeviceToHost
        )
    );

    // ============================================================
    // GPU radix sort using hipCUB
    // ============================================================

    void* d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;

    // Query temporary storage size
    HIP_CHECK(
        hipcub::DeviceRadixSort::SortKeys(
            d_temp_storage,
            temp_storage_bytes,
            d_data,
            d_data,
            N
        )
    );

    HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes));

    auto gpu_start = std::chrono::high_resolution_clock::now();

    HIP_CHECK(
        hipcub::DeviceRadixSort::SortKeys(
            d_temp_storage,
            temp_storage_bytes,
            d_data,
            d_data,
            N
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time_ms =
        std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    // Copy GPU-sorted data back to host
    std::vector<int> h_gpu(N);
    HIP_CHECK(
        hipMemcpy(
            h_gpu.data(),
            d_data,
            N * sizeof(int),
            hipMemcpyDeviceToHost
        )
    );

    // ============================================================
    // CPU parallel sort
    // ============================================================

    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::sort(std::execution::par, h_cpu.begin(), h_cpu.end());
    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_time_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ============================================================
    // Verify correctness and print results
    // ============================================================

    bool correct =
        std::equal(h_cpu.begin(), h_cpu.end(), h_gpu.begin());

    std::cout << "Results match: "
              << (correct ? "YES" : "NO") << "\n";
    std::cout << "CPU parallel sort time: "
              << cpu_time_ms << " ms\n";
    std::cout << "GPU hipCUB sort time:  "
              << gpu_time_ms << " ms\n";

    // ============================================================
    // Cleanup
    // ============================================================

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_temp_storage));

    return EXIT_SUCCESS;
}
