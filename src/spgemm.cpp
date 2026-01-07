/**
 * @file spgemm.cpp
 * @brief Demonstrates SpGEMM (sparse general matrix-matrix multiplication)
 *        on GPU using hipSPARSE with CSR matrices.
 *
 * Example output (measured on 1 AMD MI300A APU):
 * \code
 * Matrix A: 10000000 x 10000000 with nnz = 100000000
 * Matrix B: 10000000 x 10000000 with nnz = 100000000
 * Matrix C: 10000000 x 10000000 with nnz = 999994750
 * First few entries of C:
 * C[0] = 25.9424 (col 21866)
 * C[1] = 24.7442 (col 51201)
 * C[2] = 22.9642 (col 298749)
 * C[3] = 0.909916 (col 379633)
 * C[4] = 37.8533 (col 383441)
 * C[5] = 27.9135 (col 393337)
 * C[6] = 25.0793 (col 406619)
 * C[7] = 15.1923 (col 426698)
 * C[8] = 28.3851 (col 576559)
 * C[9] = 18.1633 (col 703616)
 * SpGEMM completed successfully.
 * \endcode
 *
 * Hardware and Software Environment:
 * - ROCm 7.1.1
 *
 * Demonstrates:
 * - Generating large random sparse CSR matrices on host
 * - Allocating and copying CSR matrices to GPU
 * - Using hipSPARSE SpGEMM routines for sparse matrix multiplication
 * - Querying and allocating output CSR matrix
 * - Copying results back to host for inspection
 *
 * @note Requires HIP and hipSPARSE.
 *
 * @author Marco Zank
 * @date 2026-01-07
 */

#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>

#include <iostream>
#include <vector>
#include <random>

/**
 * @brief Macro to check HIP runtime API errors.
 *
 * Prints the error message and exits if the HIP function fails.
 */
#define HIP_CHECK(err) \
    if (err != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }

/**
 * @brief Macro to check hipSPARSE API errors.
 *
 * Prints the error message and exits if the hipSPARSE function fails.
 */
#define HIPSPARSE_CHECK(err) \
    if (err != HIPSPARSE_STATUS_SUCCESS) { \
        std::cerr << "hipSPARSE error at line " << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }

/**
 * @brief Main function demonstrating large SpGEMM using hipSPARSE.
 *
 * Steps:
 * 1. Generate large random sparse matrices A and B in CSR format on host.
 * 2. Allocate GPU memory and copy A and B.
 * 3. Create hipSPARSE CSR descriptors.
 * 4. Perform SpGEMM (work estimation, compute, copy) on GPU.
 * 5. Query output matrix C, allocate memory, and copy first few entries to host.
 * 6. Cleanup GPU resources.
 *
 * @return int Returns EXIT_SUCCESS on success.
 */
int main()
{
    // ------------------------------------------------------------
    // Create hipSPARSE handle
    // ------------------------------------------------------------
    hipsparseHandle_t handle;
    HIPSPARSE_CHECK(hipsparseCreate(&handle));

    // ------------------------------------------------------------
    // Matrix dimensions
    // ------------------------------------------------------------
    constexpr size_t A_rows = 10'000'000; /**< Number of rows of A */
    constexpr size_t A_cols = 10'000'000; /**< Number of columns of A */
    constexpr size_t B_rows = A_cols;     /**< Number of rows of B */
    constexpr size_t B_cols = 10'000'000; /**< Number of columns of B */

    constexpr size_t nnzA = 100'000'000; /**< Non-zero elements in A */
    constexpr size_t nnzB = 100'000'000; /**< Non-zero elements in B */

    // ------------------------------------------------------------
    // Host CSR memory allocation
    // ------------------------------------------------------------
    std::vector<int> hA_rp(A_rows + 1, 0); /**< Row pointers of A */
    std::vector<int> hA_ci(nnzA);          /**< Column indices of A */
    std::vector<double> hA_v(nnzA);        /**< Values of A */

    std::vector<int> hB_rp(B_rows + 1, 0); /**< Row pointers of B */
    std::vector<int> hB_ci(nnzB);          /**< Column indices of B */
    std::vector<double> hB_v(nnzB);        /**< Values of B */

    // Random number generation
    std::mt19937 rng(123); 
    std::uniform_int_distribution<size_t> col_dist(0, A_cols - 1);
    std::uniform_real_distribution<double> val_dist(0.1, 10.0);

    // ------------------------------------------------------------
    // Generate random sparse CSR matrix A
    // ------------------------------------------------------------
    for (size_t row = 0; row < A_rows; ++row)
    {
        size_t row_nnz = nnzA / A_rows;
        hA_rp[row + 1] = hA_rp[row] + static_cast<int>(row_nnz);
        for (int i = hA_rp[row]; i < hA_rp[row + 1]; ++i)
        {
            size_t i_size_t = static_cast<size_t>(i);
            hA_ci[i_size_t] = static_cast<int>(col_dist(rng));
            hA_v[i_size_t]  = val_dist(rng);
        }
    }

    // ------------------------------------------------------------
    // Generate random sparse CSR matrix B
    // ------------------------------------------------------------
    for (size_t row = 0; row < B_rows; ++row)
    {
        size_t row_nnz = nnzB / B_rows;
        hB_rp[row + 1] = hB_rp[row] + static_cast<int>(row_nnz);
        for (int i = hB_rp[row]; i < hB_rp[row + 1]; ++i)
        {
            size_t i_size_t = static_cast<size_t>(i);
            hB_ci[i_size_t] = static_cast<int>(col_dist(rng));
            hB_v[i_size_t]  = val_dist(rng);
        }
    }

    // ------------------------------------------------------------
    // Device memory allocation
    // ------------------------------------------------------------
    int *dA_rp = nullptr, *dA_ci = nullptr, *dB_rp = nullptr, *dB_ci = nullptr;
    double *dA_v = nullptr, *dB_v = nullptr;

    HIP_CHECK(hipMalloc(&dA_rp, (A_rows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&dA_ci, nnzA * sizeof(int)));
    HIP_CHECK(hipMalloc(&dA_v,  nnzA * sizeof(double)));

    HIP_CHECK(hipMalloc(&dB_rp, (B_rows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&dB_ci, nnzB * sizeof(int)));
    HIP_CHECK(hipMalloc(&dB_v,  nnzB * sizeof(double)));

    HIP_CHECK(hipMemcpy(dA_rp, hA_rp.data(), (A_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dA_ci, hA_ci.data(), nnzA * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dA_v,  hA_v.data(),  nnzA * sizeof(double), hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(dB_rp, hB_rp.data(), (B_rows + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB_ci, hB_ci.data(), nnzB * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB_v,  hB_v.data(),  nnzB * sizeof(double), hipMemcpyHostToDevice));

    // ------------------------------------------------------------
    // Create CSR descriptors
    // ------------------------------------------------------------
    hipsparseSpMatDescr_t matA, matB, matC;
    HIPSPARSE_CHECK(hipsparseCreateCsr(&matA,
        static_cast<int64_t>(A_rows),
        static_cast<int64_t>(A_cols),
        static_cast<int64_t>(nnzA),
        dA_rp, dA_ci, dA_v,
        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F));

    HIPSPARSE_CHECK(hipsparseCreateCsr(&matB,
        static_cast<int64_t>(B_rows),
        static_cast<int64_t>(B_cols),
        static_cast<int64_t>(nnzB),
        dB_rp, dB_ci, dB_v,
        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F));

    HIPSPARSE_CHECK(hipsparseCreateCsr(&matC,
        static_cast<int64_t>(A_rows),
        static_cast<int64_t>(B_cols),
        0, nullptr, nullptr, nullptr,
        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F));

    // ------------------------------------------------------------
    // SpGEMM parameters and descriptor
    // ------------------------------------------------------------
    double alpha = 1.0, beta = 0.0;
    hipsparseSpGEMMDescr_t spgemmDesc;
    HIPSPARSE_CHECK(hipsparseSpGEMM_createDescr(&spgemmDesc));

    // ------------------------------------------------------------
    // Step 1: Work estimation
    // ------------------------------------------------------------
    size_t bufferSize1 = 0, bufferSize2 = 0;
    void *dBuffer1 = nullptr, *dBuffer2 = nullptr;

    HIPSPARSE_CHECK(hipsparseSpGEMM_workEstimation(handle,
        HIPSPARSE_OPERATION_NON_TRANSPOSE,
        HIPSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        HIP_R_64F, HIPSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize1, nullptr));

    HIP_CHECK(hipMalloc(&dBuffer1, bufferSize1));
    HIPSPARSE_CHECK(hipsparseSpGEMM_workEstimation(handle,
        HIPSPARSE_OPERATION_NON_TRANSPOSE,
        HIPSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        HIP_R_64F, HIPSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize1, dBuffer1));

    // ------------------------------------------------------------
    // Step 2: Compute
    // ------------------------------------------------------------
    HIPSPARSE_CHECK(hipsparseSpGEMM_compute(handle,
        HIPSPARSE_OPERATION_NON_TRANSPOSE,
        HIPSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        HIP_R_64F, HIPSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize2, nullptr));

    HIP_CHECK(hipMalloc(&dBuffer2, bufferSize2));
    HIPSPARSE_CHECK(hipsparseSpGEMM_compute(handle,
        HIPSPARSE_OPERATION_NON_TRANSPOSE,
        HIPSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        HIP_R_64F, HIPSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize2, dBuffer2));

    // ------------------------------------------------------------
    // Step 3: Query nnz of C
    // ------------------------------------------------------------
    int64_t C_rows_int, C_cols_int, nnzC_int;
    HIPSPARSE_CHECK(hipsparseSpMatGetSize(matC, &C_rows_int, &C_cols_int, &nnzC_int));

    size_t C_rows = static_cast<size_t>(C_rows_int);
    size_t C_cols = static_cast<size_t>(C_cols_int);
    size_t nnzC   = static_cast<size_t>(nnzC_int);

    std::cout << "Matrix A: " << A_rows << " x " << A_cols 
              << " with nnz = " << nnzA << "\n";
    std::cout << "Matrix B: " << B_rows << " x " << B_cols 
              << " with nnz = " << nnzB << "\n";
    std::cout << "Matrix C: " << C_rows << " x " << C_cols 
              << " with nnz = " << nnzC << "\n";

    // ------------------------------------------------------------
    // Step 4: Allocate C
    // ------------------------------------------------------------
    int *dC_rp = nullptr, *dC_ci = nullptr;
    double *dC_v = nullptr;

    HIP_CHECK(hipMalloc(&dC_rp, (C_rows + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&dC_ci, nnzC * sizeof(int)));
    HIP_CHECK(hipMalloc(&dC_v,  nnzC * sizeof(double)));
    HIPSPARSE_CHECK(hipsparseCsrSetPointers(matC, dC_rp, dC_ci, dC_v));

    // ------------------------------------------------------------
    // Step 5: Copy result
    // ------------------------------------------------------------
    HIPSPARSE_CHECK(hipsparseSpGEMM_copy(handle,
        HIPSPARSE_OPERATION_NON_TRANSPOSE,
        HIPSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        HIP_R_64F, HIPSPARSE_SPGEMM_DEFAULT,
        spgemmDesc));

    // ------------------------------------------------------------
    // Copy first 10 entries to host
    // ------------------------------------------------------------
    size_t print_nnz = std::min<size_t>(nnzC, 10);
    std::vector<int> hC_ci(print_nnz);
    std::vector<double> hC_v(print_nnz);

    HIP_CHECK(hipMemcpy(hC_ci.data(), dC_ci, print_nnz * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hC_v.data(),  dC_v,  print_nnz * sizeof(double), hipMemcpyDeviceToHost));

    std::cout << "First few entries of C:\n";
    for (size_t i = 0; i < print_nnz; ++i)
        std::cout << "C[" << i << "] = " << hC_v[i]
                  << " (col " << hC_ci[i] << ")\n";

    // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------
    HIP_CHECK(hipFree(dA_rp)); 
    HIP_CHECK(hipFree(dA_ci)); 
    HIP_CHECK(hipFree(dA_v));
    HIP_CHECK(hipFree(dB_rp));
    HIP_CHECK(hipFree(dB_ci));
    HIP_CHECK(hipFree(dB_v));
    HIP_CHECK(hipFree(dC_rp)); 
    HIP_CHECK(hipFree(dC_ci));
    HIP_CHECK(hipFree(dC_v));
    HIP_CHECK(hipFree(dBuffer1));
    HIP_CHECK(hipFree(dBuffer2));

    HIPSPARSE_CHECK(hipsparseDestroySpMat(matA));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matB));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matC));
    HIPSPARSE_CHECK(hipsparseSpGEMM_destroyDescr(spgemmDesc));
    HIPSPARSE_CHECK(hipsparseDestroy(handle));

    std::cout << "SpGEMM completed successfully.\n";
    return EXIT_SUCCESS;
}
