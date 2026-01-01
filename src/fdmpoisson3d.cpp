/**
 * @file fdmpoisson3d.cpp
 * @brief Solves the 3D Poisson equation using finite differences and rocALUTION
 *        with a refinement study (SA-AMG + CG solver).
 *
 * Example output (measured on 1 AMD MI300A APU) for ./fdmpoisson3d 3
 * \code
 * Number of CPU cores: 48
 * Host thread affinity policy - thread mapping on every core
 * Number of HIP devices in the system: 1
 * rocALUTION ver 4.0.1-b0adf82
 * rocALUTION platform is initialized
 * Accelerator backend: HIP
 * OpenMP threads: 48
 * rocBLAS ver 5.1.1.f322e9ab61
 * rocSPARSE ver 4.1.0-f322e9ab61
 * ------------------------------------------------
 * Selected HIP device: 0
 * Device name: AMD Instinct MI300A
 * totalGlobalMem: 131072 MByte
 * clockRate: 2100000
 * compute capability: 9.4
 * ------------------------------------------------
 * MPI is not initialized
 * Refinement study (Poisson 3D, SAAMG + CG)
 * ----------------------------------------------------------------------------------------------------------------------
 * Level |   Nx=Ny=Nz   |    DoF     | CG iters | GPU Solver time [s] | CPU Solver time [s] |    L2 error   |  Linf error
 * ----------------------------------------------------------------------------------------------------------------------
 *     0 |           64 |     238328 |       21 |               0.447 |               0.640 |     9.818e-02 |   6.310e-01
 *     1 |          128 |    2000376 |       24 |               0.162 |               2.163 |     1.914e-02 |   1.204e-01
 *     2 |          256 |   16387064 |       29 |               0.940 |              18.137 |     4.488e-03 |   2.831e-02
 *     3 |          512 |  132651000 |       34 |               7.675 |             150.312 |     1.101e-03 |   7.020e-03
 * ----------------------------------------------------------------------------------------------------------------------
 * \endcode
 *
 * Hardware and Software Environment:
 * - ROCm 7.1.1
 * - rocALUTION 4.0.1
 * 
 * @section run How to Run
 *
 * To run the solver, provide the maximal level of refinement as one command-line argument:
 * \code
 * ./fdmpoisson3d level_max
 * \endcode
 * Example:
 * \code
 * ./fdmpoisson3d 3
 * \endcode
 * 
 * Demonstrates:
 * - Building a 3D FDM Poisson matrix with homogeneous Dirichlet BCs
 * - Building RHS and exact solution vectors
 * - Solving using rocALUTION with SA-AMG preconditioned CG
 * - Computing L2 and Linf errors
 * - Performance measurement for different refinement levels on CPU and GPU
 *
 * @author Marco Zank
 * @date 2026-01-01
 */

#include <rocalution/rocalution.hpp>
#include <rocalution/solvers/multigrid/smoothed_amg.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <execution>
#include <numeric>
#include <chrono>
#include <iomanip>

using namespace rocalution;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
constexpr double PI = 3.14159265358979323846; /**< π */
constexpr double DOMAIN_LENGTH = 2.0 * PI;    /**< Physical domain length */

// -----------------------------------------------------------------------------
// Exact solution and RHS (manufactured solution)
// -----------------------------------------------------------------------------
/**
 * @brief Exact solution for manufactured problem.
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @param z z-coordinate
 * @return double Exact solution u(x,y,z)
 */
__host__ __device__ inline
double exactSolution(double x, double y, double z)
{
    return std::sin(x) * std::sin(y) * std::sin(z) * std::cos(x * y * z);
}

/**
 * @brief Right-hand side function f(x,y,z) for manufactured solution.
 * 
 * @param x x-coordinate
 * @param y y-coordinate
 * @param z z-coordinate
 * @return double RHS value
 */
__host__ __device__ inline
double rhsFunction(double x, double y, double z)
{
    const double sx = std::sin(x), sy = std::sin(y), sz = std::sin(z);
    const double cx = std::cos(x), cy = std::cos(y), cz = std::cos(z);

    const double g = sx * sy * sz;
    const double gx = cx * sy * sz;
    const double gy = sx * cy * sz;
    const double gz = sx * sy * cz;
    const double lap_g = -3.0 * g;  // Δg = -3g

    const double xyz = x * y * z;
    const double h   = std::cos(xyz);
    const double sh  = std::sin(xyz);

    const double hx = -y * z * sh;
    const double hy = -x * z * sh;
    const double hz = -x * y * sh;

    const double hxx = -(y * y * z * z) * h;
    const double hyy = -(x * x * z * z) * h;
    const double hzz = -(x * x * y * y) * h;
    const double lap_h = hxx + hyy + hzz;

    const double grad_g_dot_grad_h = gx * hx + gy * hy + gz * hz;

    return -(h * lap_g + g * lap_h + 2.0 * grad_g_dot_grad_h);
}

// -----------------------------------------------------------------------------
// 3D index helpers
// -----------------------------------------------------------------------------
/**
 * @brief Compute linear index for 3D array flattened in row-major order.
 * 
 * @param i x-index
 * @param j y-index
 * @param k z-index
 * @param Nx_i number of interior points in x
 * @param Ny_i number of interior points in y
 * @return size_t linear index
 */
inline size_t idx3D(size_t i, size_t j, size_t k,
                    size_t Nx_i, size_t Ny_i)
{
    return k * (Nx_i * Ny_i) + j * Nx_i + i;
}

// -----------------------------------------------------------------------------
// Build 3D FDM Poisson matrix (Dirichlet BC)
// -----------------------------------------------------------------------------
/**
 * @brief Builds CSR representation of the 3D FDM Poisson matrix.
 * 
 * @param Nx Total grid points in x
 * @param Ny Total grid points in y
 * @param Nz Total grid points in z
 * @param row_offset Output row offsets (CSR)
 * @param col Output column indices (CSR)
 * @param val Output values (CSR)
 */
static void buildFDM3DPoisson(size_t Nx, size_t Ny, size_t Nz,
                              std::vector<int>& row_offset,
                              std::vector<int>& col,
                              std::vector<double>& val)
{
    const size_t Nx_i = Nx - 2;
    const size_t Ny_i = Ny - 2;
    const size_t Nz_i = Nz - 2;
    const size_t N = Nx_i * Ny_i * Nz_i;

    const double hx2 = 1.0 / std::pow(DOMAIN_LENGTH / static_cast<double>(Nx - 1), 2);
    const double hy2 = 1.0 / std::pow(DOMAIN_LENGTH / static_cast<double>(Ny - 1), 2);
    const double hz2 = 1.0 / std::pow(DOMAIN_LENGTH / static_cast<double>(Nz - 1), 2);

    // Phase 1: compute nnz per row
    std::vector<int> row_nnz(N);
    for (size_t k = 0; k < Nz_i; ++k)
        for (size_t j = 0; j < Ny_i; ++j)
            for (size_t i = 0; i < Nx_i; ++i)
            {
                int nnz = 1;
                if (i>0) ++nnz; if (i<Nx_i-1) ++nnz;
                if (j>0) ++nnz; if (j<Ny_i-1) ++nnz;
                if (k>0) ++nnz; if (k<Nz_i-1) ++nnz;
                row_nnz[idx3D(i,j,k,Nx_i,Ny_i)] = nnz;
            }

    // Prefix sum -> row_offset
    row_offset.resize(N + 1);
    row_offset[0] = 0;
    std::partial_sum(row_nnz.begin(), row_nnz.end(), row_offset.begin() + 1);

    const size_t nnz_total = static_cast<size_t>(row_offset[N]);
    col.resize(nnz_total);
    val.resize(nnz_total);

    // Phase 2: fill CSR entries in parallel
    std::vector<size_t> rows(N);
    std::iota(rows.begin(), rows.end(), 0);

    std::for_each(std::execution::par, rows.begin(), rows.end(),
                  [&](size_t row)
    {
        const size_t k = row / (Nx_i * Ny_i);
        const size_t j = (row / Nx_i) % Ny_i;
        const size_t i = row % Nx_i;

        size_t p = static_cast<size_t>(row_offset[row]);

        col[p] = static_cast<int>(row);
        val[p++] = 2.0 * (hx2 + hy2 + hz2);

        if (i>0)        { col[p] = static_cast<int>(idx3D(i-1,j,k,Nx_i,Ny_i)); val[p++] = -hx2; }
        if (i<Nx_i-1)   { col[p] = static_cast<int>(idx3D(i+1,j,k,Nx_i,Ny_i)); val[p++] = -hx2; }
        if (j>0)        { col[p] = static_cast<int>(idx3D(i,j-1,k,Nx_i,Ny_i)); val[p++] = -hy2; }
        if (j<Ny_i-1)   { col[p] = static_cast<int>(idx3D(i,j+1,k,Nx_i,Ny_i)); val[p++] = -hy2; }
        if (k>0)        { col[p] = static_cast<int>(idx3D(i,j,k-1,Nx_i,Ny_i)); val[p++] = -hz2; }
        if (k<Nz_i-1)   { col[p] = static_cast<int>(idx3D(i,j,k+1,Nx_i,Ny_i)); val[p++] = -hz2; }
    });
}

// -----------------------------------------------------------------------------
// Build RHS or exact solution vector
// -----------------------------------------------------------------------------
/**
 * @brief Builds a vector from a function sampled on the 3D grid interior.
 * 
 * @tparam Func Function type taking (x,y,z) -> double
 * @param Nx Grid points in x
 * @param Ny Grid points in y
 * @param Nz Grid points in z
 * @param vec Output vector
 * @param func Function to evaluate
 */
template <typename Func>
static void build3DVector(size_t Nx, size_t Ny, size_t Nz,
                          std::vector<double>& vec,
                          Func func)
{
    const size_t Nx_i = Nx - 2;
    const size_t Ny_i = Ny - 2;
    const size_t Nz_i = Nz - 2;
    const size_t N = Nx_i * Ny_i * Nz_i;

    const double hx = DOMAIN_LENGTH / static_cast<double>(Nx - 1);
    const double hy = DOMAIN_LENGTH / static_cast<double>(Ny - 1);
    const double hz = DOMAIN_LENGTH / static_cast<double>(Nz - 1);

    vec.resize(N);
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    std::transform(std::execution::par,
                   indices.begin(), indices.end(),
                   vec.begin(),
                   [&](size_t id)
                   {
                       const size_t i = id % Nx_i;
                       const size_t j = (id / Nx_i) % Ny_i;
                       const size_t k = id / (Nx_i * Ny_i);
                       return func(static_cast<double>(i+1)*hx,
                                   static_cast<double>(j+1)*hy,
                                   static_cast<double>(k+1)*hz);
                   });
}

// -----------------------------------------------------------------------------
// Compute L2 and Linf errors
// -----------------------------------------------------------------------------
/**
 * @brief Compute L2 and Linf errors between solution vector and exact solution.
 * 
 * @param x Computed solution vector
 * @param u_exact Exact solution vector
 * @return std::pair<double,double> L2 and Linf error
 */
static std::pair<double,double> computeErrorL2Linf(const std::vector<double>& x,
                                                   const std::vector<double>& u_exact)
{
    const size_t N = x.size();

    double l2 = std::sqrt(
        std::transform_reduce(
            std::execution::par,
            x.begin(), x.end(),
            u_exact.begin(),
            0.0,
            std::plus<>(),
            [](double xi, double ui){ double e = xi - ui; return e*e; }
        ) / static_cast<double>(N)
    );

    double linf = std::transform_reduce(
        std::execution::par,
        x.begin(), x.end(),
        u_exact.begin(),
        0.0,
        [](double a, double b){ return std::max(a,b); },
        [](double xi, double ui){ return std::abs(xi-ui); }
    );

    return {l2, linf};
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " level_max\n";
        return EXIT_FAILURE;
    }

    size_t level_max = 0;
    try {
        long long temp_level = std::stoll(argv[1]);
        if (temp_level < 0) throw std::invalid_argument("negative level");
        level_max = static_cast<size_t>(temp_level);
    } catch (...) {
        std::cerr << "Invalid input: level_max must be a non-negative integer.\n";
        return EXIT_FAILURE;
    }

    init_rocalution();
    info_rocalution();

    std::cout << "Refinement study (Poisson 3D, SAAMG + CG)\n";
    std::cout << "----------------------------------------------------------------------------------------------------------------------\n";
    std::cout << "Level |   Nx=Ny=Nz   |    DoF     | CG iters | GPU Solver time [s] | CPU Solver time [s] |    L2 error   |  Linf error\n";
    std::cout << "----------------------------------------------------------------------------------------------------------------------\n";

    for (size_t level = 0; level <= level_max; ++level)
    {
        const size_t N = 64 * (1 << level);
        const size_t Nx = N, Ny = N, Nz = N;
        const size_t N_interior = (Nx-2)*(Ny-2)*(Nz-2);

        // Build matrix and RHS
        std::vector<int> row_offset, col;
        std::vector<double> val;
        buildFDM3DPoisson(Nx, Ny, Nz, row_offset, col, val);

        std::vector<double> h_b;
        build3DVector(Nx, Ny, Nz, h_b, rhsFunction);

        // rocALUTION objects
        LocalMatrix<double> A;
        LocalVector<double> x, b;

        A.CopyFromHostCSR(row_offset.data(), col.data(), val.data(),
                          "FDM_Poisson_3D",
                          static_cast<int>(val.size()),
                          static_cast<int>(N_interior),
                          static_cast<int>(N_interior));

        x.Allocate("x", static_cast<int>(N_interior));
        b.Allocate("b", static_cast<int>(N_interior));
        b.CopyFromHostData(h_b.data());

        std::vector<double> h_u_exact;
        build3DVector(Nx, Ny, Nz, h_u_exact, exactSolution);

        // CPU-only solve
        x.Zeros();        // reset solution vector

        CG<LocalMatrix<double>, LocalVector<double>, double> solver_cpu;
        SAAMG<LocalMatrix<double>, LocalVector<double>, double> precond_cpu;
        precond_cpu.SetCoarseningStrategy(PMIS);
        precond_cpu.Verbose(0);
        solver_cpu.SetPreconditioner(precond_cpu);
        solver_cpu.SetOperator(A);
        solver_cpu.Init(1e-8, 1e-12, 1e+6, 1000);
        solver_cpu.Verbose(0);

        auto t_start_cpu = std::chrono::high_resolution_clock::now();
        solver_cpu.Build();
        solver_cpu.Solve(b, &x);
        auto t_end_cpu = std::chrono::high_resolution_clock::now();
        double solver_time_cpu = std::chrono::duration<double>(t_end_cpu - t_start_cpu).count();
        int cg_iters_cpu = solver_cpu.GetIterationCount();
        solver_cpu.Clear();

        std::vector<double> h_x_cpu(N_interior);
        x.CopyToData(h_x_cpu.data());
        auto [l2_cpu, linf_cpu] = computeErrorL2Linf(h_x_cpu, h_u_exact);

        // GPU solve
        x.Zeros();        // reset solution vector
        A.MoveToAccelerator();
        x.MoveToAccelerator();
        b.MoveToAccelerator();

        CG<LocalMatrix<double>, LocalVector<double>, double> solver_gpu;
        SAAMG<LocalMatrix<double>, LocalVector<double>, double> precond_gpu;
        precond_gpu.SetCoarseningStrategy(PMIS);
        precond_gpu.Verbose(0);
        solver_gpu.SetPreconditioner(precond_gpu);
        solver_gpu.SetOperator(A);
        solver_gpu.Init(1e-8, 1e-12, 1e+6, 1000);
        solver_gpu.Verbose(0);

        auto t_start_gpu = std::chrono::high_resolution_clock::now();
        solver_gpu.Build();
        solver_gpu.Solve(b, &x);
        auto t_end_gpu = std::chrono::high_resolution_clock::now();
        double solver_time_gpu = std::chrono::duration<double>(t_end_gpu - t_start_gpu).count();
        int cg_iters_gpu = solver_gpu.GetIterationCount();
        solver_gpu.Clear();

        x.MoveToHost();
        std::vector<double> h_x_gpu(N_interior);
        x.CopyToData(h_x_gpu.data());

        auto [l2_gpu, linf_gpu] = computeErrorL2Linf(h_x_gpu, h_u_exact);

        if (cg_iters_cpu != cg_iters_gpu) {
            std::cout << "Mismatch at level " << level
                    << ": CG iterations CPU=" << cg_iters_cpu
                    << ", GPU=" << cg_iters_gpu << "\n";
        }

        if (std::abs(l2_cpu - l2_gpu) > 1e-12) {
            std::cout << "Mismatch at level " << level
                    << ": L2 error CPU=" << std::scientific << l2_cpu
                    << ", GPU=" << l2_gpu << "\n";
        }

        if (std::abs(linf_cpu - linf_gpu) > 1e-12) {
            std::cout << "Mismatch at level " << level
                    << ": Linf error CPU=" << std::scientific << linf_cpu
                    << ", GPU=" << linf_gpu << "\n";
        }

        // Output both GPU and CPU times and errors from GPU
        std::cout << std::setw(5) << level << " | "
                << std::setw(12) << Nx << " | "
                << std::setw(10) << N_interior << " | "
                << std::setw(8) << cg_iters_gpu << " | "
                << std::fixed << std::setprecision(3)
                << std::setw(19) << solver_time_gpu << " | "
                << std::setw(19) << solver_time_cpu << " | "
                << std::scientific << std::setprecision(3)
                << std::setw(13) << l2_gpu << " | "
                << std::setw(11) << linf_gpu << "\n";

        A.Clear(); x.Clear(); b.Clear();
    }

    std::cout << "----------------------------------------------------------------------------------------------------------------------\n";
    stop_rocalution();
    return EXIT_SUCCESS;
}
