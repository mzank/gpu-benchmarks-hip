# HIP

This project demonstrates **GPU-accelerated computations using HIP** on AMD GPUs.  
It includes the following examples:

1. **DGEMM** (`gemm.cpp`) – Double-precision general matrix-matrix multiplication using CPU BLAS and GPU hipBLAS.  
1. **Vector Reduction** (`vectorreduction.cpp`) – Sum reduction of a large vector using CPU parallel STL and a GPU HIP kernel.
1. **Large-Scale Integer Sorting (CPU + GPU)** (`sorting.cpp`) – Sorts ~1 billion integers using hipRAND for GPU random number generation, hipCUB radix sort on GPU, and C++17 parallel STL sort on CPU, with performance comparison.
1. **MPI GPU Ring (CPU-based MPI, pure C)** (`mpigpuring.c`) – Measures GPU-to-GPU ring bandwidth using HIP and CPU-based (non-GPU-aware) MPI.
1. **MPI GPU Ring (GPU-aware MPI, pure C)** (`mpigpuawarering.c`) – Measures GPU-to-GPU ring bandwidth using HIP and GPU-aware MPI with direct device-buffer communication.
1. **RCCL GPU Ring (pure C)** (`rcclring.c`) – Measures GPU-to-GPU ring bandwidth using HIP, RCCL, and CPU-based MPI.
1. **Monte Carlo Integration (CPU + GPU)** (`montecarlointegration.cpp`) – Estimates a 3D integral using Monte Carlo sampling on CPU (C++17 parallel STL) and GPU (HIP + hipRAND), with performance comparison.
1. **3D FFT Poisson Solver (CPU + GPU)** (`fftpoisson3d.cpp`) – Solves a periodic 3D Poisson equation using FFTs on CPU (FFTW) and GPU (hipFFT), compares performance and numerical accuracy.
1. **3D FDM Poisson Solver (CPU + GPU)** (`fdmpoisson3d.cpp`) – Solves a 3D Poisson equation on a cube with homogeneous Dirichlet boundary conditions using finite differences. Uses rocALUTION with SA-AMG preconditioned CG and performs a refinement study, comparing solver time on CPU/GPU and numerical errors (L2 and Linf) across grid levels.

---

## Hardware and Software Environment

All example results were obtained on a single node with **4 AMD MI300A APUs**, where the examples without MPI utilized only 1 AMD MI300A APU.

> **Note:** Each MI300A integrates Zen 4 CPU cores and a CDNA3 GPU sharing HBM memory. NUMA locality therefore plays a critical role in both CPU and GPU performance.

### Software Stack

- ROCm 7.1.1 (tested, should work on other ROCm-supported AMD GPUs)
- OpenMPI 5.0.7 (with UCC 1.4.4, UCX 1.18.1, ROCm support)
- OpenBLAS 0.3.20
- RCCL 2.27.7
- FFTW3 3.3.10 (with threads support)
- rocALUTION 4.0.1

> **Note:** FFTW should be built with threads support (`--enable-threads`) for optimal CPU performance.

The code should work on other ROCm-supported AMD GPUs, though performance and numerical results may vary.

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
- ROCm (e.g. 7.1.1)
- HIP, hipBLAS, hipCUB, hipFFT, hipRAND and rocALUTION
- BLAS library (e.g. OpenBLAS)
- MPI library (e.g. OpenMPI) with NUMA binding support
- RCCL (e.g. 2.27.7)
- FFTW3 (e.g. 3.3.10 with threads support)
- GNU Make
- C++17-compatible compiler for HIP/C++ sources (e.g. `hipcc`)
- C11-compatible compiler for pure C MPI examples (e.g. `hipcc`)

> **Optional:** `libnuma` can improve CPU memory locality on multi-socket systems. OpenMPI’s `--bind-to numa` is sufficient for most setups. If `libnuma` is not installed, it can be removed from the linker flags in the Makefile.

---

## Build Instructions

Build the examples using the provided **Makefile**:

```bash
# Build all examples
make

# Or build individually
make build/gemm
make build/vectorreduction
make build/sorting
make build/mpigpuring
make build/mpigpuawarering
make build/rcclring
make build/montecarlointegration
make build/fftpoisson3d
make build/fdmpoisson3d
```

All binaries are generated in the `build/` directory.

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

### Run Sorting example
```bash
./build/sorting
```

### Run MPI GPU Ring example with CPU-based MPI
```bash
export HSA_ENABLE_SDMA=1  # enables GPU DMA for host-staged MPI
mpirun -np 4 --bind-to numa --map-by numa --report-bindings ./build/mpigpuring
```

> **Note:** This configuration uses **host-staged communication**.
`HSA_ENABLE_SDMA` controls the use of GPU copy engines; behavior depends on the MPI data path.

### Run MPI GPU Ring example with GPU-aware MPI
```bash
export HSA_ENABLE_SDMA=0  # disables GPU DMA for GPU-aware direct transfers
mpirun -np 4 -mca pml ucx --bind-to numa --map-by numa --report-bindings ./build/mpigpuawarering
```

> **Note:** This example requires a **GPU-aware MPI** build (e.g. OpenMPI with UCX and ROCm support). GPU device pointers are passed directly to `MPI_Isend`/`MPI_Irecv` without host staging.

### Run RCCL GPU Ring example
```bash
export HSA_NO_SCRATCH_RECLAIM=1  # keeps GPU scratch memory allocated between kernels
mpirun -np 4 --bind-to numa --map-by numa --report-bindings ./build/rcclring
```
> **Note:** This example uses RCCL to test collective communication patterns but runs entirely with CPU-based MPI. Setting `HSA_NO_SCRATCH_RECLAIM=1` ensures that GPU scratch (private) memory remains allocated across kernel launches, which improves performance stability and prevents memory allocation overhead in multi-GPU workloads.

### Run Monte Carlo Integration example
```bash
./build/montecarlointegration
```

### Run 3D FFT Poisson Solver
```bash
./build/fftpoisson3d 1024 1024 1024
```

> **Note:** The three arguments specify the grid dimensions `(Nx Ny Nz)`. Large grids require significant GPU and host memory.

#### FFTW Wisdom

The 3D FFT Poisson solver uses **FFTW’s wisdom mechanism** to speed up CPU FFT planning:

1. **First run**:  
   If no wisdom file exists for the specified grid size, FFTW will measure plans. This can take several seconds to minutes depending on the grid.  
2. **Wisdom file creation**:  
After the first run, FFTW writes a wisdom file named: `fftpoisson3d_fftw_wisdom_Nx_Ny_Nz.dat`
where `Nx`, `Ny`, `Nz` are the grid dimensions.
3. **Subsequent runs**:  
The program automatically loads the existing wisdom file, drastically reducing CPU FFT plan time.  

> **Tip:** Wisdom files are grid-size specific. Changing any of `Nx`, `Ny`, or `Nz` will require generating a new wisdom file.

### Run 3D FDM Poisson Solver
```bash
./build/fdmpoisson3d 3
```
> **Note:** The single argument `level_max` specifies the maximum refinement level. Each level doubles the grid size in each direction starting from 64 points. For example, `level_max=3` runs grids of size 64³, 128³, 256³, and 512³ points. The program reports CG iterations, solver time on CPU/GPU, and L2/Linf errors for each level.

---

Program outputs shown below are also saved under the `output/` directory
(e.g. `output/gemm_output.txt`, `output/numa_info.txt`, `output/gpu_topology.txt`).

---

## Example Outputs

DGEMM (gemm.cpp)
```yaml
==================== Results ====================
CPU DGEMM time: 66171.6 ms
GPU hipBLAS DGEMM time: 1940.62 ms
Maximum |C_cpu - C_gpu| = 2.20098e-10
```

Vector Reduction (vectorreduction.cpp)
```yaml
==================== Results ====================
sum_CPU: 1.07374e+09, time: 49.8159 ms
sum_GPU: 1.07374e+09, time: 4.55739 ms
|sum_CPU - sum_GPU| = 0
```

Large-Scale Sorting (sorting.cpp)
```yaml
Results match: YES
CPU parallel sort time: 3181.42 ms
GPU hipCUB sort time:  41.4023 ms
```

MPI GPU Ring with CPU-based MPI (mpigpuring.c)
```yaml
[hostname:PID] Rank 0 bound to package[0][core:0-23]
[hostname:PID] Rank 1 bound to package[1][core:24-47]
[hostname:PID] Rank 2 bound to package[2][core:48-71]
[hostname:PID] Rank 3 bound to package[3][core:72-95]

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

MPI GPU Ring with GPU-aware MPI (mpigpuawarering.c)
```yaml
[hostname:PID] Rank 0 bound to package[0][core:0-23]
[hostname:PID] Rank 1 bound to package[1][core:24-47]
[hostname:PID] Rank 2 bound to package[2][core:48-71]
[hostname:PID] Rank 3 bound to package[3][core:72-95]

Msg size (MB) | Rank 0 BW (GB/s) | Send[0] | Recv[0] | Rank 1 BW (GB/s) | Send[0] | Recv[0] | Rank 2 BW (GB/s) | Send[0] | Recv[0] | Rank 3 BW (GB/s) | Send[0] | Recv[0] |
        67.11 |            37.46 |    1.00 |    4.00 |            37.47 |    2.00 |    1.00 |            37.49 |    3.00 |    2.00 |            37.49 |    4.00 |    3.00 |
       134.22 |           158.22 |    1.00 |    4.00 |           158.03 |    2.00 |    1.00 |           158.43 |    3.00 |    2.00 |           158.46 |    4.00 |    3.00 |
       268.44 |           170.04 |    1.00 |    4.00 |           170.04 |    2.00 |    1.00 |           170.23 |    3.00 |    2.00 |           170.24 |    4.00 |    3.00 |
       536.87 |           169.34 |    1.00 |    4.00 |           168.98 |    2.00 |    1.00 |           169.48 |    3.00 |    2.00 |           169.49 |    4.00 |    3.00 |
      1073.74 |           169.64 |    1.00 |    4.00 |           169.61 |    2.00 |    1.00 |           169.83 |    3.00 |    2.00 |           169.83 |    4.00 |    3.00 |
      2147.48 |           170.03 |    1.00 |    4.00 |           170.00 |    2.00 |    1.00 |           169.71 |    3.00 |    2.00 |           169.71 |    4.00 |    3.00 |
      4294.97 |           171.27 |    1.00 |    4.00 |           171.24 |    2.00 |    1.00 |           171.04 |    3.00 |    2.00 |           171.04 |    4.00 |    3.00 |
      8589.93 |           171.50 |    1.00 |    4.00 |           171.44 |    2.00 |    1.00 |           171.25 |    3.00 |    2.00 |           171.25 |    4.00 |    3.00 |
```

RCCL GPU Ring (rcclring.c)
```yaml
[hostname:PID] Rank 0 bound to package[0][core:0-23]
[hostname:PID] Rank 1 bound to package[1][core:24-47]
[hostname:PID] Rank 2 bound to package[2][core:48-71]
[hostname:PID] Rank 3 bound to package[3][core:72-95]

Msg size (MB) | Rank 0 BW (GB/s) | Send[0] | Recv[0] | Rank 1 BW (GB/s) | Send[0] | Recv[0] | Rank 2 BW (GB/s) | Send[0] | Recv[0] | Rank 3 BW (GB/s) | Send[0] | Recv[0] |
        67.11 |           164.67 |    1.00 |    4.00 |           165.22 |    2.00 |    1.00 |           163.56 |    3.00 |    2.00 |           162.95 |    4.00 |    3.00 |
       134.22 |           169.64 |    1.00 |    4.00 |           170.57 |    2.00 |    1.00 |           170.66 |    3.00 |    2.00 |           165.90 |    4.00 |    3.00 |
       268.44 |           173.46 |    1.00 |    4.00 |           173.64 |    2.00 |    1.00 |           171.43 |    3.00 |    2.00 |           173.24 |    4.00 |    3.00 |
       536.87 |           176.19 |    1.00 |    4.00 |           176.19 |    2.00 |    1.00 |           175.56 |    3.00 |    2.00 |           175.86 |    4.00 |    3.00 |
      1073.74 |           176.41 |    1.00 |    4.00 |           176.42 |    2.00 |    1.00 |           177.33 |    3.00 |    2.00 |           177.37 |    4.00 |    3.00 |
      2147.48 |           177.82 |    1.00 |    4.00 |           177.79 |    2.00 |    1.00 |           178.33 |    3.00 |    2.00 |           178.28 |    4.00 |    3.00 |
      4294.97 |           171.75 |    1.00 |    4.00 |           172.09 |    2.00 |    1.00 |           172.43 |    3.00 |    2.00 |           172.43 |    4.00 |    3.00 |
      8589.93 |           171.53 |    1.00 |    4.00 |           171.23 |    2.00 |    1.00 |           171.01 |    3.00 |    2.00 |           171.01 |    4.00 |    3.00 |
```

Monte Carlo Integration (montecarlointegration.cpp)
```yaml
GPU config: 14592 blocks x 256 threads
GPU result: -0.00378359 in 0.0204081 s
CPU result: -0.00378631 in 0.683242 s
```

FFT Poisson Solver (fftpoisson3d.cpp) for `./build/fftpoisson3d 1024 1024 1024`
```yaml
Running FFT Poisson solver with grid: 1024 x 1024 x 1024 = 1073741824
GPU warm-up completed.
GPU run 1 time = 0.330952 s
GPU run 2 time = 0.334203 s
GPU run 3 time = 0.334891 s
GPU run 4 time = 0.337424 s
GPU run 5 time = 0.33906 s
CPU: No FFTW wisdom found, plans will be measured.
CPU warm-up completed.
CPU run 1 time = 11.9273 s
CPU run 2 time = 12.7428 s
CPU run 3 time = 13.5741 s
CPU run 4 time = 14.2809 s
CPU run 5 time = 13.7809 s
FFTW wisdom saved to fftpoisson3d_fftw_wisdom_1024_1024_1024.dat.

================== GPU vs CPU Comparison ==================
Solver | Avg Time (s) |         L2 Error |        Max Error
-------|--------------|------------------|-----------------
GPU    |     0.335306 |     8.755893e-15 |     5.573320e-14
CPU    |    13.261201 |     8.710855e-15 |     5.484502e-14
===========================================================
```

3D FDM Poisson Solver (fdmpoisson3d.cpp) for `./build/fdmpoisson3d 3`
```yaml
./fdmpoisson3d 3
Number of CPU cores: 48
Host thread affinity policy - thread mapping on every core
Number of HIP devices in the system: 1
rocALUTION ver 4.0.1-b0adf82
rocALUTION platform is initialized
Accelerator backend: HIP
OpenMP threads: 48
rocBLAS ver 5.1.1.f322e9ab61
rocSPARSE ver 4.1.0-f322e9ab61
------------------------------------------------
Selected HIP device: 0
Device name: AMD Instinct MI300A
totalGlobalMem: 131072 MByte
clockRate: 2100000
compute capability: 9.4
------------------------------------------------
MPI is not initialized
Refinement study (Poisson 3D, SAAMG + CG)
----------------------------------------------------------------------------------------------------------------------
Level |   Nx=Ny=Nz   |    DoF     | CG iters | GPU Solver time [s] | CPU Solver time [s] |    L2 error   |  Linf error
----------------------------------------------------------------------------------------------------------------------
    0 |           64 |     238328 |       21 |               0.447 |               0.640 |     9.818e-02 |   6.310e-01
    1 |          128 |    2000376 |       24 |               0.162 |               2.163 |     1.914e-02 |   1.204e-01
    2 |          256 |   16387064 |       29 |               0.940 |              18.137 |     4.488e-03 |   2.831e-02
    3 |          512 |  132651000 |       34 |               7.675 |             150.312 |     1.101e-03 |   7.020e-03
----------------------------------------------------------------------------------------------------------------------
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
- Function-level documentation
- GPU kernel explanations
- Example outputs
- Compilation and runtime notes

---

## Third-Party Software and Trademarks

This project is a research and demonstration code showcasing GPU-accelerated
computations using the AMD ROCm platform.

It depends on the following third-party software:

- **HIP**, **hipBLAS**, **hipCUB**, **hipFFT**, **hipRAND** and **rocALUTION** (AMD ROCm)
- **OpenBLAS**
- **OpenMPI**
- **FFTW**

These components are licensed under their respective open-source licenses and
are **not** covered by this project's Apache License 2.0. Users are responsible for
complying with the license terms of each dependency.

HIP, ROCm, and AMD are trademarks of Advanced Micro Devices, Inc.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

You can also view the license here: https://www.apache.org/licenses/LICENSE-2.0

---
