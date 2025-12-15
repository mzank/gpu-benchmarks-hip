# Compiler and flags
CXX = hipcc
CXXFLAGS = -std=c++17 -O3
LDFLAGS_GEMM = -lhipblas -lopenblas -ltbb
LDFLAGS_VEC = -ltbb

# Source files
SRC_GEMM = src/gemm.cpp
SRC_VEC  = src/vectorreduction.cpp

# Output binaries
BUILD_DIR = build
GEMM_BIN = $(BUILD_DIR)/gemm
VEC_BIN  = $(BUILD_DIR)/vectorreduction

# Default target: build both
all: $(BUILD_DIR) $(GEMM_BIN) $(VEC_BIN)

# Build gemm
$(GEMM_BIN): $(SRC_GEMM)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS_GEMM) -o $@

# Build vector reduction
$(VEC_BIN): $(SRC_VEC)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS_VEC) -o $@

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean
