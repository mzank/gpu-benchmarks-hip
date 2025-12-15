# Compiler and flags
CXX = hipcc
CXXFLAGS = -std=c++17 -O3
CFLAGS = -std=c11 -O3          # Pure C flags

LDFLAGS_GEMM = -lhipblas -lopenblas -ltbb
LDFLAGS_VEC  = -ltbb
LDFLAGS_MPI  = -lmpi -lnuma

# Source files
SRC_GEMM = src/gemm.cpp
SRC_VEC  = src/vectorreduction.cpp
SRC_MPI  = src/mpigpuring.c

# Output binaries
BUILD_DIR = build
GEMM_BIN = $(BUILD_DIR)/gemm
VEC_BIN  = $(BUILD_DIR)/vectorreduction
MPI_BIN  = $(BUILD_DIR)/mpigpuring

# Default target: build all examples
all: $(BUILD_DIR) $(GEMM_BIN) $(VEC_BIN) $(MPI_BIN)

# Build gemm
$(GEMM_BIN): $(SRC_GEMM)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS_GEMM) -o $@

# Build vector reduction
$(VEC_BIN): $(SRC_VEC)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS_VEC) -o $@

# Build MPI GPU Ring with CPU-based MPI
$(MPI_BIN): $(SRC_MPI)
	$(CXX) $(CFLAGS) $^ $(LDFLAGS_MPI) -o $@

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean
