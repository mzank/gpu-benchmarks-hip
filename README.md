# HIP

This project demonstrates GPU-accelerated computations using **HIP** on AMD GPUs.  
It contains:

1. **DGEMM** (`gemm.cpp`) – Double-precision general matrix-matrix multiplication using CPU BLAS and GPU hipBLAS.  
2. **Vector Reduction** (`vectorreduction.cpp`) – Sum reduction of a large vector using CPU parallel STL and a GPU HIP kernel.
3. **MPI GPU Ring with CPU-based MPI (pure C)** (`mpigpuring.c`) – Measures GPU-to-GPU ring bandwidth using HIP and CPU-based MPI (non-GPU-aware).

---

## Hardware and Software Environment

All example results were obtained on **1 node with 4 AMD MI300A APUs** with the following software stack:

- ROCm 7.1.1  (tested, should work on other ROCm-supported AMD GPUs)
- openMPI 5.0.7-ucc1.4.4-ucx1.18.1   
- OpenBLAS 0.3.20  

The code should work on other ROCm-supported AMD GPUs, although performance and numerical results may vary.

---

## Requirements

- AMD GPU supported by ROCm
- ROCm (e.g. 7.1.1 compatible)
- HIP and hipBLAS
- BLAS library (e.g. OpenBLAS)
- MPI library (e.g. OpenMPI) capable of binding to NUMA nodes
- GNU Make
- C++17-compatible compiler (e.g. `hipcc`)

> **Note:** A NUMA library (`libnuma`) is **optional**. It can improve CPU memory locality on multi-socket systems, but OpenMPI’s `--bind-to numa` is sufficient for most setups.

---

## Build Instructions

Build the examples using the provided **Makefile**:

```bash
# Build both examples
make

# Or build individually
make build/gemm
make build/vectorreduction
make build/mpigpuring
```

This will create the binaries in the build/ directory.

---

## How to Run

After building, you can run the programs as follows:

### Run DGEMM example
```bash
./build/gemm
```

### Run Vector Reduction example
```bash
./build/vectorreduction
```

### Run MPI GPU Ring example with CPU-based MPI
```bash
export HSA_ENABLE_SDMA=1  # Enable asynchronous DMA for GPU-to-GPU transfers
mpirun -np 4 --bind-to numa --map-by numa --report-bindings ./build/mpigpuring
```

Program outputs shown below are also saved under the `output/` directory
(e.g. `output/gemm_output.txt`).

---

## Example Outputs

DGEMM (gemm.cpp)
```yaml
==================== Results ====================
CPU DGEMM time: 29002.7 ms
GPU hipBLAS DGEMM time: 2170.61 ms
Maximum |C_cpu - C_gpu| = 2.30102e-10
```

Vector Reduction (vectorreduction.cpp)
```yaml
==================== Results ====================
CPU sum: 1.07374e+09, time: 43.1988 ms
GPU sum: 1.07374e+09, time: 4.56007 ms
Maximum |CPU - GPU| difference: 0
```

MPI GPU Ring with CPU-based MPI (mpigpuring.c)
```yaml
mpirun -np 4 --bind-to numa --map-by numa --report-bindings ./mpigpuring
Msg size (MB) | Rank 0 BW (GB/s) | Send[0] | Recv[0] | Rank 1 BW (GB/s) | Send[0] | Recv[0] | Rank 2 BW (GB/s) | Send[0] | Recv[0] | Rank 3 BW (GB/s) | Send[0] | Recv[0] |
        67.11 |            11.76 |    1.00 |    4.00 |            11.82 |    2.00 |    1.00 |            11.82 |    3.00 |    2.00 |            11.76 |    4.00 |    3.00 |
       134.22 |            11.68 |    1.00 |    4.00 |            11.76 |    2.00 |    1.00 |            11.77 |    3.00 |    2.00 |            11.69 |    4.00 |    3.00 |
       268.44 |            11.93 |    1.00 |    4.00 |            12.02 |    2.00 |    1.00 |            12.02 |    3.00 |    2.00 |            11.93 |    4.00 |    3.00 |
       536.87 |            12.04 |    1.00 |    4.00 |            12.11 |    2.00 |    1.00 |            12.11 |    3.00 |    2.00 |            12.04 |    4.00 |    3.00 |
      1073.74 |            12.08 |    1.00 |    4.00 |            12.07 |    2.00 |    1.00 |            12.07 |    3.00 |    2.00 |            12.08 |    4.00 |    3.00 |
      2147.48 |            12.13 |    1.00 |    4.00 |            12.11 |    2.00 |    1.00 |            12.11 |    3.00 |    2.00 |            12.13 |    4.00 |    3.00 |
      4294.97 |            12.15 |    1.00 |    4.00 |            12.24 |    2.00 |    1.00 |            12.23 |    3.00 |    2.00 |            12.15 |    4.00 |    3.00 |
      8589.93 |            12.15 |    1.00 |    4.00 |            12.23 |    2.00 |    1.00 |            12.23 |    3.00 |    2.00 |            12.15 |    4.00 |    3.00 |
```

---

## Doxygen Documentation

Detailed API and workflow documentation is available as HTML:
```bash
docs/html/index.html
```

To generate it yourself (requires `doxygen` installed):
```bash
doxygen Doxyfile
```

This documentation includes:
- Function descriptions
- GPU kernel explanations
- Example outputs
- Compilation and runtime notes

---

## Third-Party Software and Trademarks

This project is a research and demonstration code showcasing GPU-accelerated
computations using the AMD ROCm platform.

It depends on the following third-party software:

- **HIP** and **hipBLAS** (AMD ROCm)
- **OpenBLAS**
- **OpenMPI**

These components are licensed under their respective open-source licenses and
are not covered by this project's Apache License 2.0. Users are responsible for
complying with the license terms of each dependency.

HIP, ROCm, and AMD are trademarks of Advanced Micro Devices, Inc.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

You can also view the license here: https://www.apache.org/licenses/LICENSE-2.0

---
