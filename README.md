# HIP

This project demonstrates GPU-accelerated computations using **HIP** on AMD GPUs.  
It contains two examples:

1. **DGEMM** (`gemm.cpp`) – Double-precision general matrix-matrix multiplication using CPU BLAS and GPU hipBLAS.  
2. **Vector Reduction** (`vectorreduction.cpp`) – Sum reduction of a large vector using CPU parallel STL and a GPU HIP kernel.

---

## Hardware and Software Environment

All example results were obtained on **1 node with 4 AMD MI300A APUs** with the following software stack:

- ROCm 7.1.1  
- openMPI 5.0.7-ucc1.4.4-ucx1.18.1   
- OpenBLAS 0.3.20  

---

## Build Instructions

Build the examples using the provided **Makefile**:

```bash
# Build both examples
make

# Or build individually
make build/gemm
make build/vectorreduction```
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

Each program prints:
- Computation times on CPU and GPU
- Validation results (maximum difference between CPU and GPU results)

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

---

## Doxygen Documentation

Detailed API and workflow documentation is available as HTML:
```bash
docs/html/index.html
```

To generate it yourself:
```bash
doxygen Doxyfile
```

This documentation includes:
- Function descriptions
- GPU kernel explanations
- Example outputs
- Compilation and runtime notes

---

## License

This project is licensed under the Apache License 2.0. See LICENSE for details.

You can also view the license here: https://www.apache.org/licenses/LICENSE-2.0

---
