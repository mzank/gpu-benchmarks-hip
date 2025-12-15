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

### NUMA Hardware Configuration

The system used for testing has 4 NUMA nodes with the following configuration (output from `numactl --hardware`):
```yaml
available: 4 nodes (0-3)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119
node 0 size: 128193 MB
node 0 free: 126397 MB
node 1 cpus: 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
node 1 size: 128719 MB
node 1 free: 127384 MB
node 2 cpus: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167
node 2 size: 128719 MB
node 2 free: 126290 MB
node 3 cpus: 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191
node 3 size: 128661 MB
node 3 free: 127346 MB
node distances:
node   0   1   2   3 
  0:  10  32  32  32 
  1:  32  10  32  32 
  2:  32  32  10  32 
  3:  32  32  32  10 
```
> **Note:** This NUMA configuration was used to guide CPU affinity in the MPI GPU ring examples. For reproducibility, the same output is saved in `output/numa_info.txt`.

---

### GPU Topology

The ROCm GPU topology on this system (output from `rocm-smi --showtopo`) is:
```yaml
============================ ROCm System Management Interface ============================

================================ Weight between two GPUs =================================
       GPU0         GPU1         GPU2         GPU3         
GPU0   0            15           15           15           
GPU1   15           0            15           15           
GPU2   15           15           0            15           
GPU3   15           15           15           0            

================================= Hops between two GPUs ==================================
       GPU0         GPU1         GPU2         GPU3         
GPU0   0            1            1            1            
GPU1   1            0            1            1            
GPU2   1            1            0            1            
GPU3   1            1            1            0            

=============================== Link Type between two GPUs ===============================
       GPU0         GPU1         GPU2         GPU3         
GPU0   0            XGMI         XGMI         XGMI         
GPU1   XGMI         0            XGMI         XGMI         
GPU2   XGMI         XGMI         0            XGMI         
GPU3   XGMI         XGMI         XGMI         0            

======================================= Numa Nodes =======================================
GPU[0]		: (Topology) Numa Node: 0
GPU[0]		: (Topology) Numa Affinity: 0
GPU[1]		: (Topology) Numa Node: 1
GPU[1]		: (Topology) Numa Affinity: 1
GPU[2]		: (Topology) Numa Node: 2
GPU[2]		: (Topology) Numa Affinity: 2
GPU[3]		: (Topology) Numa Node: 3
GPU[3]		: (Topology) Numa Affinity: 3
================================== End of ROCm SMI Log ===================================
```
> **Note:** This GPU topology output is saved in `output/gpu_topology.txt`.

---

## Requirements

- AMD GPU supported by ROCm
- ROCm (e.g. 7.1.1 compatible)
- HIP and hipBLAS
- BLAS library (e.g. OpenBLAS)
- MPI library (e.g. OpenMPI) capable of binding to NUMA nodes
- GNU Make
- C++17-compatible compiler (e.g. `hipcc`)

> **Note:** A NUMA library (`libnuma`) is **optional**. It can improve CPU memory locality on multi-socket systems, but OpenMPI’s `--bind-to numa` is sufficient for most setups. The Makefile may link -lnuma; if your system does not have NUMA, you can remove it.

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
(e.g. `output/gemm_output.txt`, `output/vector_output.txt`, `output/mpirun_output.txt`, `output/numa_info.txt`).

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
