# Compiler and flags
CXX = hipcc
CXXFLAGS = -std=c++17 -O3
CFLAGS = -std=c11 -O3

LDFLAGS_GEMM = -lhipblas -lopenblas -ltbb
LDFLAGS_VEC  = -ltbb
LDFLAGS_MPI  = -lmpi -lnuma
LDFLAGS_RCCL  = -lrccl -lmpi -lnuma
LDFLAGS_MC   = -lhiprand -ltbb -lm

# Source files
SRC_GEMM = src/gemm.cpp
SRC_VEC  = src/vectorreduction.cpp
SRC_MPI  = src/mpigpuring.c
SRC_MPI_AWARE = src/mpigpuawarering.c
SRC_RCCL = src/rcclring.c
SRC_MC   = src/montecarlointegration.cpp

# Output binaries
BUILD_DIR = build
GEMM_BIN = $(BUILD_DIR)/gemm
VEC_BIN  = $(BUILD_DIR)/vectorreduction
MPI_BIN  = $(BUILD_DIR)/mpigpuring
MPI_AWARE_BIN = $(BUILD_DIR)/mpigpuawarering
RCCL_BIN = $(BUILD_DIR)/rcclring
MC_BIN   = $(BUILD_DIR)/montecarlointegration

# Default target: build all examples
all: $(BUILD_DIR) $(GEMM_BIN) $(VEC_BIN) $(MPI_BIN) $(MPI_AWARE_BIN) $(RCCL_BIN) $(MC_BIN)

# Build gemm
$(GEMM_BIN): $(SRC_GEMM)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS_GEMM) -o $@

# Build vector reduction
$(VEC_BIN): $(SRC_VEC)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS_VEC) -o $@

# Build MPI GPU Ring with CPU-based MPI
$(MPI_BIN): $(SRC_MPI)
	$(CXX) $(CFLAGS) $^ $(LDFLAGS_MPI) -o $@

# Build MPI GPU Ring with GPU-aware MPI
$(MPI_AWARE_BIN): $(SRC_MPI_AWARE)
	$(CXX) $(CFLAGS) $^ $(LDFLAGS_MPI) -o $@

# Build RCCL GPU Ring
$(RCCL_BIN): $(SRC_RCCL)
	$(CXX) $(CFLAGS) $^ $(LDFLAGS_RCCL) -o $@

# Build Monte Carlo Integration (CPU + GPU)
$(MC_BIN): $(SRC_MC)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS_MC) -o $@

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean
