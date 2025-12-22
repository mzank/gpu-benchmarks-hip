/**
 * @file fftpoisson3d.cpp
 * @brief Solves a 3D Poisson equation using FFTs on CPU (FFTW) and GPU (hipFFT),
 *        and compares performance and accuracy.
 * 
 * Example output (measured on 1 AMD MI300A APU) for ./fftpoisson3d 1024 1024 1024
 * \code
 * Running FFT Poisson solver with grid: 1024 x 1024 x 1024 = 1073741824
 * GPU warm-up completed.
 * GPU run 1 time = 0.330952 s
 * GPU run 2 time = 0.334203 s
 * GPU run 3 time = 0.334891 s
 * GPU run 4 time = 0.337424 s
 * GPU run 5 time = 0.33906 s
 * CPU: No FFTW wisdom found, plans will be measured.
 * CPU warm-up completed.
 * CPU run 1 time = 11.9273 s
 * CPU run 2 time = 12.7428 s
 * CPU run 3 time = 13.5741 s
 * CPU run 4 time = 14.2809 s
 * CPU run 5 time = 13.7809 s
 * FFTW wisdom saved to fftpoisson3d_fftw_wisdom_1024_1024_1024.dat.
 * 
 * ================== GPU vs CPU Comparison ==================
 * Solver | Avg Time (s) |         L2 Error |        Max Error
 * -------|--------------|------------------|-----------------
 * GPU    |     0.335306 |     8.755893e-15 |     5.573320e-14
 * CPU    |    13.261201 |     8.710855e-15 |     5.484502e-14
 * ===========================================================
 * \endcode
 *
 * This program computes the solution of a periodic 3D Poisson problem
 * by transforming the right-hand side into Fourier space, dividing by
 * the squared wave number, and transforming back.
 *
 * Both CPU and GPU implementations are provided:
 * - CPU: FFTW with multithreading and optional wisdom
 * - GPU: hipFFT with custom HIP kernels
 *
 * The numerical solution is compared against a known analytical solution.
 * L2 and maximum error norms are reported along with timing results.
 *
 * Hardware and Software Environment:
 * - ROCm 7.1.1
 * - FFTW3 3.3.10 (with threads support)
 *
 * @note Periodic boundary conditions are assumed.
 *
 * @author Marco Zank
 * @date 2025-12-22
 */

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <fftw3.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include <execution>
#include <chrono>
#include <iomanip>
#include <thread>
#include <sstream>
#include <string>

// ============================================================
// Error checking macros
// ============================================================

/**
 * @brief Macro to check HIP runtime API errors.
 *
 * Prints an error message and exits if a HIP call fails.
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
 * @brief Macro to check hipFFT API errors.
 *
 * Prints an error message and exits if a hipFFT call fails.
 */
#define HIPFFT_CHECK(status)                                      \
    {                                                             \
        hipfftResult err = status;                                \
        if (err != HIPFFT_SUCCESS) {                              \
            std::cerr << "hipFFT Error: " << err                  \
                      << " at line " << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                              \
        }                                                         \
    }

// ============================================================
// Constants
// ============================================================

/** @brief Number of timing runs for averaging */
constexpr size_t N_RUNS = 5;

/** @brief Mathematical constant π */
constexpr double PI = 3.14159265358979323846;

/** @brief Domain length (periodic in all directions) */
constexpr double L = 2.0 * PI;

// ============================================================
// Exact solution and right-hand side
// ============================================================

/**
 * @brief Exact analytical solution u(x,y,z).
 */
__host__ __device__ inline
double exactSolution(double x, double y, double z)
{
    const double phi =
          0.7  * std::cos(x)
        + 0.5  * std::cos(2.0 * y)
        + 0.3  * std::cos(3.0 * z)
        + 0.2  * std::sin(x + y)
        + 0.1  * std::sin(y + z)
        + 0.05 * std::cos(28.0 * x)
        + 0.05 * std::sin(27.0 * (y + z));

    return std::exp(phi);
}

/**
 * @brief Right-hand side f(x,y,z) = -Δu.
 */
__host__ __device__ inline
double rhsFunction(double x, double y, double z)
{
    const double phi =
          0.7  * std::cos(x)
        + 0.5  * std::cos(2.0 * y)
        + 0.3  * std::cos(3.0 * z)
        + 0.2  * std::sin(x + y)
        + 0.1  * std::sin(y + z)
        + 0.05 * std::cos(28.0 * x)
        + 0.05 * std::sin(27.0 * (y + z));

    const double phix =
          -0.7 * std::sin(x)
        + 0.2 * std::cos(x + y)
        - 0.05 * 28.0 * std::sin(28.0 * x);

    const double phiy =
          -1.0 * std::sin(2.0 * y)
        + 0.2 * std::cos(x + y)
        + 0.1 * std::cos(y + z)
        + 0.05 * 27.0 * std::cos(27.0 * (y + z));

    const double phiz =
          -0.9 * std::sin(3.0 * z)
        + 0.1 * std::cos(y + z)
        + 0.05 * 27.0 * std::cos(27.0 * (y + z));

    const double phixx =
          -0.7 * std::cos(x)
        - 0.2 * std::sin(x + y)
        - 0.05 * 28.0 * 28.0 * std::cos(28.0 * x);

    const double phiyy =
          -2.0 * std::cos(2.0 * y)
        - 0.2 * std::sin(x + y)
        - 0.1 * std::sin(y + z)
        - 0.05 * 27.0 * 27.0 * std::sin(27.0 * (y + z));

    const double phizz =
          -2.7 * std::cos(3.0 * z)
        - 0.1 * std::sin(y + z)
        - 0.05 * 27.0 * 27.0 * std::sin(27.0 * (y + z));

    const double lap_phi = phixx + phiyy + phizz;
    const double grad_phi_sq = phix * phix + phiy * phiy + phiz * phiz;

    return -(lap_phi + grad_phi_sq) * std::exp(phi);
}

// ============================================================
// GPU kernels
// ============================================================

/**
 * @brief Initializes the RHS on the GPU.
 */
static __global__
void initRhsKernel(hipfftDoubleComplex* d_data,
                   size_t Nx, size_t Ny, size_t Nz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < Nx && j < Ny && k < Nz) {
        size_t idx = (i * Ny + j) * Nz + k;

        double x = L * static_cast<double>(i) / static_cast<double>(Nx);
        double y = L * static_cast<double>(j) / static_cast<double>(Ny);
        double z = L * static_cast<double>(k) / static_cast<double>(Nz);

        d_data[idx].x = rhsFunction(x, y, z);
        d_data[idx].y = 0.0;
    }
}

/**
 * @brief Fourier-space Poisson solver kernel.
 */
static __global__
void poissonFourierKernel(hipfftDoubleComplex* d_data,
                          size_t Nx, size_t Ny, size_t Nz)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < Nx && j < Ny && k < Nz) {
        size_t idx = (i * Ny + j) * Nz + k;

        int ki = (i <= Nx / 2) ? int(i) : int(i) - int(Nx);
        int kj = (j <= Ny / 2) ? int(j) : int(j) - int(Ny);
        int kk = (k <= Nz / 2) ? int(k) : int(k) - int(Nz);

        double k2 = double(ki * ki + kj * kj + kk * kk);

        if (k2 > 0.0) {
            d_data[idx].x /= k2;
            d_data[idx].y /= k2;
        } else {
            d_data[idx].x = 0.0;
            d_data[idx].y = 0.0;
        }
    }
}

/**
 * @brief Normalizes inverse FFT output.
 */
static __global__
void normalizeKernel(hipfftDoubleComplex* d_data,
                     size_t N, double scale)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx].x *= scale;
        d_data[idx].y *= scale;
    }
}

// ============================================================
// GPU Poisson solver
// ============================================================

/**
 * @brief Solves the Poisson problem on the GPU.
 */
static void poissonSolverGpu(size_t Nx, size_t Ny, size_t Nz,
                             hipfftHandle fftPlan,
                             hipfftDoubleComplex* d_solution)
{
    const size_t N = Nx * Ny * Nz;

    dim3 threads(8, 8, 8);
    dim3 blocks(static_cast<unsigned int>((Nx + threads.x - 1) / threads.x),
            static_cast<unsigned int>((Ny + threads.y - 1) / threads.y),
            static_cast<unsigned int>((Nz + threads.z - 1) / threads.z));

    initRhsKernel<<<blocks, threads>>>(d_solution, Nx, Ny, Nz);
    HIP_CHECK(hipDeviceSynchronize());

    HIPFFT_CHECK(hipfftExecZ2Z(fftPlan, d_solution, d_solution, HIPFFT_FORWARD));
    HIP_CHECK(hipDeviceSynchronize());

    poissonFourierKernel<<<blocks, threads>>>(d_solution, Nx, Ny, Nz);
    HIP_CHECK(hipDeviceSynchronize());

    HIPFFT_CHECK(hipfftExecZ2Z(fftPlan, d_solution, d_solution, HIPFFT_BACKWARD));
    HIP_CHECK(hipDeviceSynchronize());

    const unsigned int threads1D = 256;
    const unsigned int blocks1D = static_cast<unsigned int>((N + threads1D - 1) / threads1D);

    normalizeKernel<<<blocks1D, threads1D>>>(d_solution, N, 1.0 / double(N));
    HIP_CHECK(hipDeviceSynchronize());
}

// ============================================================
// CPU Poisson solver
// ============================================================

/**
 * @brief Solves the Poisson problem on the CPU using FFTW.
 */
static void poissonSolverCpu(size_t Nx, size_t Ny, size_t Nz,
                             fftw_complex* h_data)
{
    const size_t N = Nx * Ny * Nz;

    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par, indices.begin(), indices.end(),
        [&](size_t idx) {
            size_t i = idx / (Ny * Nz);
            size_t j = (idx / Nz) % Ny;
            size_t k = idx % Nz;

            double x = L * static_cast<double>(i) / static_cast<double>(Nx);
            double y = L * static_cast<double>(j) / static_cast<double>(Ny);
            double z = L * static_cast<double>(k) / static_cast<double>(Nz);

            h_data[idx][0] = rhsFunction(x, y, z);
            h_data[idx][1] = 0.0;
        });

    fftw_plan forwardPlan =
        fftw_plan_dft_3d(int(Nx), int(Ny), int(Nz),
                         h_data, h_data,
                         FFTW_FORWARD, FFTW_MEASURE);

    fftw_plan backwardPlan =
        fftw_plan_dft_3d(int(Nx), int(Ny), int(Nz),
                         h_data, h_data,
                         FFTW_BACKWARD, FFTW_MEASURE);

    fftw_execute(forwardPlan);

    std::for_each(std::execution::par, indices.begin(), indices.end(),
        [&](size_t idx) {
            size_t i = idx / (Ny * Nz);
            size_t j = (idx / Nz) % Ny;
            size_t k = idx % Nz;

            int ki = (i <= Nx / 2) ? int(i) : int(i) - int(Nx);
            int kj = (j <= Ny / 2) ? int(j) : int(j) - int(Ny);
            int kk = (k <= Nz / 2) ? int(k) : int(k) - int(Nz);

            double k2 = double(ki * ki + kj * kj + kk * kk);

            if (k2 > 0.0) {
                h_data[idx][0] /= k2;
                h_data[idx][1] /= k2;
            } else {
                h_data[idx][0] = 0.0;
                h_data[idx][1] = 0.0;
            }
        });

    fftw_execute(backwardPlan);

    const double scale = 1.0 / double(N);
    std::for_each(std::execution::par, indices.begin(), indices.end(),
        [&](size_t idx) {
            h_data[idx][0] *= scale;
            h_data[idx][1] *= scale;
        });

    fftw_destroy_plan(forwardPlan);
    fftw_destroy_plan(backwardPlan);
}

// ============================================================
// Main program
// ============================================================

/**
 * @brief Entry point of the 3D FFT-based Poisson solver.
 */
int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " Nx Ny Nz\n";
        return EXIT_FAILURE;
    }

    const size_t Nx = std::stoul(argv[1]);
    const size_t Ny = std::stoul(argv[2]);
    const size_t Nz = std::stoul(argv[3]);

    if (Nx <= 0 || Ny <= 0 || Nz <= 0) {
        std::cerr << "Error: Nx, Ny, Nz must be positive integers.\n";
        return EXIT_FAILURE;
    }

    const size_t N = static_cast<size_t>(Nx) * Ny * Nz;

    std::cout << "Running FFT Poisson solver with grid: "
          << Nx << " x " << Ny << " x " << Nz << " = " << N << "\n";

    // ---------------- GPU SETUP ----------------
    hipfftHandle plan_gpu;
    HIPFFT_CHECK(hipfftPlan3d(&plan_gpu, (int)Nx, (int)Ny, (int)Nz, HIPFFT_Z2Z));

    hipfftDoubleComplex* d_output;
    HIP_CHECK(hipMalloc(&d_output, N * sizeof(hipfftDoubleComplex)));

    // ---------------- GPU WARM-UP ----------------
    poissonSolverGpu(Nx, Ny, Nz, plan_gpu, d_output);
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "GPU warm-up completed.\n";

    double total_gpu_time = 0.0;
    for (size_t run = 0; run < N_RUNS; ++run) {
        auto t0_gpu = std::chrono::high_resolution_clock::now();
        poissonSolverGpu(Nx, Ny, Nz, plan_gpu, d_output);
        HIP_CHECK(hipDeviceSynchronize());
        auto t1_gpu = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> dt_gpu = t1_gpu - t0_gpu;
        total_gpu_time += dt_gpu.count();
        std::cout << "GPU run " << run+1 << " time = " << dt_gpu.count() << " s\n";
    }

    double avg_gpu_time = total_gpu_time / N_RUNS;

    // Copy GPU result to host in vector<double>
    std::vector<double> h_data(N);
    { // Use the scope to free h_temp.
        std::vector<hipfftDoubleComplex> h_temp(N);
        HIP_CHECK(hipMemcpy(h_temp.data(), d_output, N * sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost));

        std::transform(std::execution::par, h_temp.begin(), h_temp.end(), h_data.begin(),
                    [](const hipfftDoubleComplex& c){ return c.x; });
    }

    HIP_CHECK(hipFree(d_output));
    HIPFFT_CHECK(hipfftDestroy(plan_gpu));

    // ---------------- CPU SETUP ----------------
    fftw_init_threads();
    fftw_plan_with_nthreads(int(std::thread::hardware_concurrency()));

    std::ostringstream wisdom_name;
    wisdom_name << "fftpoisson3d_fftw_wisdom_"
                << Nx << "_" << Ny << "_" << Nz << ".dat";
    std::string wisdom_file = wisdom_name.str();

    bool wisdom_loaded = fftw_import_wisdom_from_filename(wisdom_file.c_str());
    if (wisdom_loaded)
        std::cout << "CPU: FFTW wisdom loaded.\n";
    else
        std::cout << "CPU: No FFTW wisdom found, plans will be measured.\n";

    fftw_complex* cpu_data = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    // ---------------- CPU WARM-UP ----------------
    poissonSolverCpu(Nx, Ny, Nz, cpu_data);
    std::cout << "CPU warm-up completed.\n";

    double total_cpu_time = 0.0;
    for (size_t run = 0; run < N_RUNS; ++run) {
        auto t0_cpu = std::chrono::high_resolution_clock::now();
        poissonSolverCpu(Nx, Ny, Nz, cpu_data);
        auto t1_cpu = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> dt_cpu = t1_cpu - t0_cpu;
        total_cpu_time += dt_cpu.count();
        std::cout << "CPU run " << run+1 << " time = " << dt_cpu.count() << " s\n";
    }

    double avg_cpu_time = total_cpu_time / N_RUNS;

    // Copy CPU solution real part into vector<double> using parallel STL
    std::vector<double> h_cpu_data(N);
    std::transform(std::execution::par, cpu_data, cpu_data + N, h_cpu_data.begin(),
                   [](const fftw_complex& c){ return c[0]; });

    if (!wisdom_loaded) {
        fftw_export_wisdom_to_filename(wisdom_file.c_str());
        std::cout << "FFTW wisdom saved to " << wisdom_file << ".\n";
    }

    fftw_cleanup_threads();
    fftw_free(cpu_data);

    // ---------------- ERROR CALCULATION ----------------
    std::vector<double> exact_u(N), gpu_errors(N), cpu_errors(N);
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    
    // ---------------- PRECOMPUTE EXACT SOLUTION ----------------
    std::transform(std::execution::par, indices.begin(), indices.end(), exact_u.begin(),
        [Nx, Ny, Nz](size_t idx){
            size_t i = idx / (Ny * Nz);
            size_t j = (idx / Nz) % Ny;
            size_t k = idx % Nz;

            return exactSolution(
                L * static_cast<double>(i) / static_cast<double>(Nx),
                L * static_cast<double>(j) / static_cast<double>(Ny),
                L * static_cast<double>(k) / static_cast<double>(Nz)
            );
        });

    // ---------------- MEAN OF EXACT SOLUTION ----------------
    double mean_exact =
        std::transform_reduce(
            std::execution::par,
            exact_u.begin(), exact_u.end(),
            0.0,
            std::plus<>(),
            [](double v){ return v; }
        ) / static_cast<double>(N);


    // ---------------- ZERO-MEAN EXACT SOLUTION ----------------
    std::for_each(std::execution::par, exact_u.begin(), exact_u.end(),
        [mean_exact](double& v){
            v -= mean_exact;
        });

    std::transform(std::execution::par, indices.begin(), indices.end(), gpu_errors.begin(),
        [&h_data, &exact_u](size_t idx){
            return std::abs(h_data[idx] - exact_u[idx]);
        });

    std::transform(std::execution::par, indices.begin(), indices.end(), cpu_errors.begin(),
        [&h_cpu_data, &exact_u](size_t idx){
            return std::abs(h_cpu_data[idx] - exact_u[idx]);
        });

    double gpu_l2 = std::transform_reduce(std::execution::par, gpu_errors.begin(), gpu_errors.end(), 0.0, std::plus<>(), [](double e){ return e*e; });
    double cpu_l2 = std::transform_reduce(std::execution::par, cpu_errors.begin(), cpu_errors.end(), 0.0, std::plus<>(), [](double e){ return e*e; });

    double gpu_max = *std::max_element(std::execution::par, gpu_errors.begin(), gpu_errors.end());
    double cpu_max = *std::max_element(std::execution::par, cpu_errors.begin(), cpu_errors.end());

    // ---------------- PRINT COMPARISON ----------------
    std::cout << "\n================== GPU vs CPU Comparison ==================\n";
    std::cout << std::left 
              << std::setw(6) << "Solver" << " | "
              << std::right << std::setw(12) << "Avg Time (s)" << " | "
              << std::setw(16) << "L2 Error" << " | "
              << std::setw(16) << "Max Error" << "\n";

    std::cout << "-------|--------------|------------------|-----------------\n";

    std::cout << std::left << std::setw(6) << "GPU" << " | "
              << std::right << std::setw(12) << std::fixed << std::setprecision(6) << avg_gpu_time << " | "
              << std::setw(16) << std::scientific << std::setprecision(6) << std::sqrt(gpu_l2/(double)N) << " | "
              << std::setw(16) << std::scientific << std::setprecision(6) << gpu_max << "\n";

    std::cout << std::left << std::setw(6) << "CPU" << " | "
              << std::right << std::setw(12) << std::fixed << std::setprecision(6) << avg_cpu_time << " | "
              << std::setw(16) << std::scientific << std::setprecision(6) << std::sqrt(cpu_l2/double(N)) << " | "
              << std::setw(16) << std::scientific << std::setprecision(6) << cpu_max << "\n";

    std::cout << "===========================================================\n";

    return EXIT_SUCCESS;
}
