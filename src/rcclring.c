/**
 * @file rcclring.c
 * @brief Measure GPU-to-GPU ring bandwidth using HIP, RCCL, and MPI
 *
 * This example demonstrates:
 * - GPU memory allocation
 * - RCCL ring communication
 * - MPI coordination for multi-GPU across nodes
 * - Performance measurement
 * - Verification of first element
 *
 * End-to-end timing includes:
 *   RCCL send/recv only
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
 *         67.11 |           164.67 |    1.00 |    4.00 |           165.22 |    2.00 |    1.00 |           163.56 |    3.00 |    2.00 |           162.95 |    4.00 |    3.00 |
 *        134.22 |           169.64 |    1.00 |    4.00 |           170.57 |    2.00 |    1.00 |           170.66 |    3.00 |    2.00 |           165.90 |    4.00 |    3.00 |
 *        268.44 |           173.46 |    1.00 |    4.00 |           173.64 |    2.00 |    1.00 |           171.43 |    3.00 |    2.00 |           173.24 |    4.00 |    3.00 |
 *        536.87 |           176.19 |    1.00 |    4.00 |           176.19 |    2.00 |    1.00 |           175.56 |    3.00 |    2.00 |           175.86 |    4.00 |    3.00 |
 *       1073.74 |           176.41 |    1.00 |    4.00 |           176.42 |    2.00 |    1.00 |           177.33 |    3.00 |    2.00 |           177.37 |    4.00 |    3.00 |
 *       2147.48 |           177.82 |    1.00 |    4.00 |           177.79 |    2.00 |    1.00 |           178.33 |    3.00 |    2.00 |           178.28 |    4.00 |    3.00 |
 *       4294.97 |           171.75 |    1.00 |    4.00 |           172.09 |    2.00 |    1.00 |           172.43 |    3.00 |    2.00 |           172.43 |    4.00 |    3.00 |
 *       8589.93 |           171.53 |    1.00 |    4.00 |           171.23 |    2.00 |    1.00 |           171.01 |    3.00 |    2.00 |           171.01 |    4.00 |    3.00 |
 * \endcode
 *
 * Hardware and Software Environment:
 * - ROCm 7.1.1
 * - openMPI 5.0.7-ucc1.4.4-ucx1.18.1
 * - RCCL 2.27.7
 *
 * @author Marco Zank
 * @date 2025-12-16
 */

#ifdef USE_NUMA
#define _GNU_SOURCE       /* Needed for sched_getcpu() */
#include <sched.h>        /* For sched_getcpu() */
#include <numa.h>         /* For NUMA allocation and node binding */
#endif

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <mpi.h>

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

/* ------------------------------------------------------------- */
/* Configuration                                                 */
/* ------------------------------------------------------------- */
#define MIN_MSG_SIZE  (1LL << 26)   /* 64 MB */
#define MAX_MSG_SIZE  (1LL << 33)   /* 8 GB */
#define N_REPEAT      10            /* Number of repetitions per message size */
#define N_WARMUP      2             /* Number of warm-up iterations */

/* ------------------------------------------------------------- */
/* HIP and RCCL error checking macros                             */
/* ------------------------------------------------------------- */
#define HIP_CHECK(call)                                                \
    do {                                                               \
        hipError_t err = (call);                                       \
        if (err != hipSuccess) {                                       \
            fprintf(stderr, "HIP error %s at %s:%d\n",                 \
                    hipGetErrorString(err), __FILE__, __LINE__);       \
            MPI_Abort(MPI_COMM_WORLD, -1);                             \
        }                                                              \
    } while (0)

#define RCCL_CHECK(call)                                               \
    do {                                                               \
        ncclResult_t res = (call);                                     \
        if (res != ncclSuccess) {                                      \
            fprintf(stderr, "RCCL error %s at %s:%d\n",                \
                    ncclGetErrorString(res), __FILE__, __LINE__);      \
            MPI_Abort(MPI_COMM_WORLD, -1);                             \
        }                                                              \
    } while (0)

/* ------------------------------------------------------------- */
/* Memory allocation check macro                                   */
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

    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
    /* HIP device selection      */
    /* ------------------------- */
    int num_devices = 0;
    HIP_CHECK(hipGetDeviceCount(&num_devices));
    HIP_CHECK(hipSetDevice(world_rank % num_devices));

    /* ------------------------- */
    /* RCCL initialization       */
    /* ------------------------- */
    ncclUniqueId id;
    if (world_rank == 0)
        RCCL_CHECK(ncclGetUniqueId(&id));

    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm;
    RCCL_CHECK(ncclCommInitRank(&comm, world_size, id, world_rank));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    const int next = (world_rank + 1) % world_size;
    const int prev = (world_rank - 1 + world_size) % world_size;

    /* ------------------------- */
    /* Print header              */
    /* ------------------------- */
    if (world_rank == 0) {
        printf("\nMsg size (MB) |");
        for (int r = 0; r < world_size; r++)
            printf(" Rank %d BW (GB/s) | Send[0] | Recv[0] |", r);
        printf("\n");
    }

    /* ------------------------- */
    /* Loop over message sizes   */
    /* ------------------------- */
    for (size_t msg_size = MIN_MSG_SIZE;
         msg_size <= MAX_MSG_SIZE;
         msg_size <<= 1)
    {
        const size_t count = msg_size / sizeof(double);
        if (count > INT_MAX) {
            fprintf(stderr, "Message too large for MPI count (%zu elements)\n", count);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        /* ------------------------- */
        /* Allocate GPU buffers      */
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
        for (size_t i = 0; i < count; i++)
            h_init[i] = (double)(world_rank + 1);

        HIP_CHECK(hipMemcpy(d_send, h_init, msg_size, hipMemcpyHostToDevice));
        free(h_init);

        HIP_CHECK(hipDeviceSynchronize());

        /* ------------------------- */
        /* Warm-up iterations        */
        /* ------------------------- */
        for (int i = 0; i < N_WARMUP; i++) {
            RCCL_CHECK(ncclGroupStart());
            RCCL_CHECK(ncclRecv(d_recv, count, ncclDouble, prev, comm, stream));
            RCCL_CHECK(ncclSend(d_send, count, ncclDouble, next, comm, stream));
            RCCL_CHECK(ncclGroupEnd());
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        /* ------------------------- */
        /* Timed iterations          */
        /* ------------------------- */
        float total_ms = 0.0f;
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for (int rep = 0; rep < N_REPEAT; rep++) {
            HIP_CHECK(hipEventRecord(start, stream));

            RCCL_CHECK(ncclGroupStart());
            RCCL_CHECK(ncclRecv(d_recv, count, ncclDouble, prev, comm, stream));
            RCCL_CHECK(ncclSend(d_send, count, ncclDouble, next, comm, stream));
            RCCL_CHECK(ncclGroupEnd());

            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
            total_ms += ms;
        }

        HIP_CHECK(hipDeviceSynchronize());

        /* ------------------------- */
        /* Verification of first element */
        /* ------------------------- */
        double send0 = 0.0, recv0 = 0.0;
        HIP_CHECK(hipMemcpy(&send0, d_send, sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&recv0, d_recv, sizeof(double), hipMemcpyDeviceToHost));

        /* ------------------------- */
        /* Compute bandwidth (GB/s)  */
        /* ------------------------- */
        const double avg_s = ((double)total_ms / N_REPEAT) * 1.0e-3;
        const double bw_GBps = (2.0 * (double)msg_size / avg_s) * 1.0e-9;

        /* ------------------------- */
        /* Gather results to rank 0  */
        /* ------------------------- */
        double *all_bw = NULL;
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
        MPI_Gather(&send0,   1, MPI_DOUBLE, all_send0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&recv0,   1, MPI_DOUBLE, all_recv0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* ------------------------- */
        /* Print results on rank 0   */
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
        /* Cleanup buffers           */
        /* ------------------------- */
        HIP_CHECK(hipFree(d_send));
        HIP_CHECK(hipFree(d_recv));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    /* ------------------------- */
    /* Cleanup RCCL and HIP      */
    /* ------------------------- */
    RCCL_CHECK(ncclCommDestroy(comm));
    HIP_CHECK(hipStreamDestroy(stream));

    MPI_Finalize();
    return EXIT_SUCCESS;
}
