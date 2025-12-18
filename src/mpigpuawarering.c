/**
 * @file mpigpuawarering.c
 * @brief Measure GPU-to-GPU ring bandwidth using HIP and GPU-aware MPI
 *
 * This example demonstrates:
 * - GPU memory allocation
 * - Direct MPI communication on GPU buffers (GPU-aware MPI)
 * - MPI ring communication
 * - Performance measurement
 * - Verification of first element
 *
 * End-to-end timing includes:
 *   GPU MPI send/recv only
 *
 * NUMA library usage is optional. If available, it improves CPU memory locality.
 * If not available, the code will still run correctly.
 * 
 * Example output (measured on 1 node with 4 AMD MI300A APUs):
 * \code
 * [hostname:PID] Rank 0 bound to package[0][core:0-23]
 * [hostname:PID] Rank 1 bound to package[1][core:24-47]
 * [hostname:PID] Rank 2 bound to package[2][core:48-71]
 * [hostname:PID] Rank 3 bound to package[3][core:72-95]
 *
 * Msg size (MB) | Rank 0 BW (GB/s) | Send[0] | Recv[0] | Rank 1 BW (GB/s) | Send[0] | Recv[0] | Rank 2 BW (GB/s) | Send[0] | Recv[0] | Rank 3 BW (GB/s) | Send[0] | Recv[0] |
 *         67.11 |            37.46 |    1.00 |    4.00 |            37.47 |    2.00 |    1.00 |            37.49 |    3.00 |    2.00 |            37.49 |    4.00 |    3.00 |
 *        134.22 |           158.22 |    1.00 |    4.00 |           158.03 |    2.00 |    1.00 |           158.43 |    3.00 |    2.00 |           158.46 |    4.00 |    3.00 |
 *        268.44 |           170.04 |    1.00 |    4.00 |           170.04 |    2.00 |    1.00 |           170.23 |    3.00 |    2.00 |           170.24 |    4.00 |    3.00 |
 *        536.87 |           169.34 |    1.00 |    4.00 |           168.98 |    2.00 |    1.00 |           169.48 |    3.00 |    2.00 |           169.49 |    4.00 |    3.00 |
 *       1073.74 |           169.64 |    1.00 |    4.00 |           169.61 |    2.00 |    1.00 |           169.83 |    3.00 |    2.00 |           169.83 |    4.00 |    3.00 |
 *       2147.48 |           170.03 |    1.00 |    4.00 |           170.00 |    2.00 |    1.00 |           169.71 |    3.00 |    2.00 |           169.71 |    4.00 |    3.00 |
 *       4294.97 |           171.27 |    1.00 |    4.00 |           171.24 |    2.00 |    1.00 |           171.04 |    3.00 |    2.00 |           171.04 |    4.00 |    3.00 |
 *       8589.93 |           171.50 |    1.00 |    4.00 |           171.44 |    2.00 |    1.00 |           171.25 |    3.00 |    2.00 |           171.25 |    4.00 |    3.00 |
 * \endcode
 *
 * Hardware and Software Environment:
 * - ROCm 7.1.1
 * - openMPI 5.0.7-ucc1.4.4-ucx1.18.1 (GPU-aware)
 *
 * @author Marco Zank
 * @date 2025-12-15
 */

#ifdef USE_NUMA
#define _GNU_SOURCE       /* Needed for sched_getcpu() */
#include <sched.h>        /* For sched_getcpu() */
#include <numa.h>         /* For NUMA allocation and node binding */
#endif

#include <mpi.h>
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

/* ------------------------------------------------------------- */
/* Configuration                                                 */
/* ------------------------------------------------------------- */
#define MIN_MSG_SIZE  (1LL << 26)   /* 64 MB */
#define MAX_MSG_SIZE  (1LL << 33)   /* 8 GB */
#define N_REPEAT      10            /* Number of repetitions per message size */

/* ------------------------------------------------------------- */
/* HIP error checking macro                                      */
/* ------------------------------------------------------------- */
#define HIP_CHECK(call)                                                \
    do {                                                               \
        hipError_t err = (call);                                       \
        if (err != hipSuccess) {                                       \
            fprintf(stderr,                                            \
                    "HIP error %s at %s:%d\n",                         \
                    hipGetErrorString(err), __FILE__, __LINE__);       \
            MPI_Abort(MPI_COMM_WORLD, -1);                             \
        }                                                              \
    } while (0)

/* ------------------------------------------------------------- */
/* Memory allocation check macro                                 */
/* ------------------------------------------------------------- */
#define CHECK_ALLOC(ptr)                                               \
    do {                                                               \
        if (!(ptr)) {                                                  \
            fprintf(stderr, "Allocation failed at %s:%d\n",            \
                    __FILE__, __LINE__);                               \
            MPI_Abort(MPI_COMM_WORLD, -1);                             \
        }                                                              \
    } while (0)

/* ------------------------------------------------------------- */
/* Main program                                                  */
/* ------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    /* ------------------------- */
    /* Initialize MPI            */
    /* ------------------------- */
    MPI_Init(&argc, &argv);

    /* ------------------------- */
    /* Optional NUMA and CPU affinity */
    /* ------------------------- */
#ifdef USE_NUMA
    int cpu  = sched_getcpu();
    int node = numa_node_of_cpu(cpu);
    numa_run_on_node(node);     /* Bind process to NUMA node */
    numa_set_localalloc();      /* Allocate future memory on local node */
#endif

    /* ------------------------- */
    /* MPI info                  */
    /* ------------------------- */
    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* ------------------------- */
    /* Node-local communicator   */
    /* ------------------------- */
    MPI_Comm host_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD,
                        MPI_COMM_TYPE_SHARED,
                        0, MPI_INFO_NULL,
                        &host_comm);

    int host_rank = 0;
    MPI_Comm_rank(host_comm, &host_rank);

    /* ------------------------- */
    /* HIP device selection      */
    /* ------------------------- */
    int num_devices = 0;
    HIP_CHECK(hipGetDeviceCount(&num_devices));
    HIP_CHECK(hipSetDevice(host_rank % num_devices));

    const int next = (world_rank + 1) % world_size;
    const int prev = (world_rank - 1 + world_size) % world_size;

    /* ------------------------- */
    /* Print header              */
    /* ------------------------- */
    if (world_rank == 0) {
        printf("\nMsg size (MB) |");
        for (int r = 0; r < world_size; r++) {
            printf(" Rank %d BW (GB/s) | Send[0] | Recv[0] |", r);
        }
        printf("\n");
    }

    /* ------------------------- */
    /* Loop over message sizes   */
    /* ------------------------- */
    for (size_t msg_size = MIN_MSG_SIZE;
         msg_size <= MAX_MSG_SIZE;
         msg_size <<= 1) {

        const size_t count = msg_size / sizeof(double);
        if (count > INT_MAX) {
            fprintf(stderr, "Message too large for MPI count (%zu elements)\n", count);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        const int mpi_count = (int)count;

        /* ------------------------- */
        /* Allocate GPU device buffers */
        /* ------------------------- */
        double *d_send = NULL;
        double *d_recv = NULL;
        HIP_CHECK(hipMalloc((void**)&d_send, msg_size));
        HIP_CHECK(hipMalloc((void**)&d_recv, msg_size));

        /* ------------------------- */
        /* Initialize host send buffer */
        /* ------------------------- */
        double *h_init = malloc(msg_size);
        CHECK_ALLOC(h_init);
        for (size_t i = 0; i < count; i++) {
            h_init[i] = (double)(world_rank + 1);
        }

        HIP_CHECK(hipMemcpy(d_send, h_init, msg_size, hipMemcpyHostToDevice));
        free(h_init);

        HIP_CHECK(hipDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);

        /* ------------------------- */
        /* Timed ring communication */
        /* ------------------------- */
        double total_time = 0.0;
        MPI_Request reqs[2];

        for (int rep = 0; rep < N_REPEAT; rep++) {

            HIP_CHECK(hipDeviceSynchronize());
            double t0 = MPI_Wtime();

            MPI_Irecv(d_recv, mpi_count, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(d_send, mpi_count, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[1]);
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

            HIP_CHECK(hipDeviceSynchronize());
            total_time += MPI_Wtime() - t0;
        }

        /* ------------------------- */
        /* Verification of first element */
        /* ------------------------- */
        double send0 = 0.0, recv0 = 0.0;
        HIP_CHECK(hipMemcpy(&send0, d_send, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&recv0, d_recv, sizeof(double), hipMemcpyDeviceToHost));

        /* ------------------------- */
        /* Compute bandwidth (GB/s)  */
        /* ------------------------- */
        const double avg_time = total_time / N_REPEAT;
        const double bw_GBps = (2.0 * (double)msg_size / avg_time) * 1.0e-9;

        /* ------------------------- */
        /* Gather results to rank 0   */
        /* ------------------------- */
        double *all_bw    = NULL;
        double *all_send0 = NULL;
        double *all_recv0 = NULL;

        if (world_rank == 0) {
            const size_t n = (size_t)world_size;
            all_bw    = malloc(n * sizeof(double));
            all_send0 = malloc(n * sizeof(double));
            all_recv0 = malloc(n * sizeof(double));
            CHECK_ALLOC(all_bw);
            CHECK_ALLOC(all_send0);
            CHECK_ALLOC(all_recv0);
        }

        MPI_Gather(&bw_GBps, 1, MPI_DOUBLE, all_bw, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&send0, 1, MPI_DOUBLE, all_send0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&recv0, 1, MPI_DOUBLE, all_recv0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* ------------------------- */
        /* Print results on rank 0    */
        /* ------------------------- */
        if (world_rank == 0) {
            printf("%13.2f |", (double)msg_size * 1.0e-6);
            for (int r = 0; r < world_size; r++) {
                printf(" %16.2f | %7.2f | %7.2f |",
                       all_bw[r], all_send0[r], all_recv0[r]);
            }
            printf("\n");

            free(all_bw);
            free(all_send0);
            free(all_recv0);
        }

        /* ------------------------- */
        /* Cleanup buffers          */
        /* ------------------------- */
        HIP_CHECK(hipFree(d_send));
        HIP_CHECK(hipFree(d_recv));
    }

    MPI_Comm_free(&host_comm);
    MPI_Finalize();
    return 0;
}
