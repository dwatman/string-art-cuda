NVCC  ?= nvcc

# Directories
SRC_DIR   := src
BUILD_DIR := build

# Files
C_SRC_FILES    = $(wildcard $(SRC_DIR)/*.c)
CUDA_SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES  = $(patsubst $(SRC_DIR)/%.c,  $(BUILD_DIR)/%.o, $(C_SRC_FILES))
OBJ_FILES += $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CUDA_SRC_FILES))

# Common includes and paths
INCLUDES   := -I.
LIBS       := -lm -lpng -lm -lGL -lglut -lGLU -lGLEW -lpthread -lX11
CUDA_LIBS  :=
C_FLAGS    := -pg
CUDA_FLAGS := -gencode arch=compute_86,code=\"sm_86,compute_86\" # for RTX 3080Ti at home

# Target rules
all: clearscreen string-gen

# Clear the screen first to make debugging easier
clearscreen:
	clear

# Executable
string-gen: $(OBJ_FILES)
	$(NVCC) $(CUDA_FLAGS) -o $@ $+ $(LIBS) $(CUDA_LIBS)

# C source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/settings.h | $(BUILD_DIR)
	$(NVCC) $(C_FLAGS) $(INCLUDES) -c $< -o $@

# CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(SRC_DIR)/settings.h | $(BUILD_DIR)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)/*.o string-gen

