# ============================================================================
# Standalone benchmark for the ANUGA shallow-water OpenMP/GPU kernels.
#
# Builds the real kernel sources (anuga/shallow_water/gpu/*.c) against a small
# C driver -- no meson, no Cython, no Python, no MPI.
#
#   make cpu        gcc,  host OpenMP           -> bin/bench_cpu
#   make gpu        nvc,  OpenMP target offload -> bin/bench_gpu
#   make ompcpu     nvc,  host OpenMP multicore -> bin/bench_ompcpu
#   make clanggpu   clang, OpenMP target (nvptx)-> bin/bench_clanggpu
#   make all        cpu + gpu
#   make verify     build cpu+gpu, run both, diff the final state
#   make clean
#
# Useful overrides:
#   make gpu GPU_ARCH=cc80        # default: autodetected from nvidia-smi
#   make cpu CC=clang
#   make cpu EXTRA_CFLAGS=-DNVTX_ENABLED
# ============================================================================

CONFIG        ?= cpu
ANUGA_ROOT    := $(abspath ..)
SW            := $(ANUGA_ROOT)/anuga/shallow_water
GPUDIR        := $(SW)/gpu
UTIL          := $(ANUGA_ROOT)/anuga/utilities

INCLUDES      := -I$(UTIL) -I$(SW) -I$(GPUDIR) -Isrc

# Autodetect compute capability (cc90 for H100, cc80 for A100, ...).
GPU_ARCH      ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                   | head -1 | tr -d '.' | sed 's/^/cc/')
ifeq ($(strip $(GPU_ARCH)),)
GPU_ARCH      := cc80
endif

WARN          := -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function

# ---------------------------------------------------------------- per-config
ifeq ($(CONFIG),cpu)
  # make predefines CC, and the nvhpc module exports CC=nvc, so ?= would never
  # fire.  Only an explicit `make cpu CC=...` should win here.
  ifneq ($(origin CC),command line)
    CC        := gcc
  endif
  CFLAGS      := -O3 -g -march=native -fopenmp -DCPU_ONLY_MODE $(WARN)
  LDFLAGS     := -fopenmp
  # gpu_device_helpers.h has `#pragma omp declare target`; without this gcc
  # emits offload vars (GPU_TINY) that never resolve in a host-only link.
  ifeq ($(findstring clang,$(notdir $(CC))),)
    CFLAGS    += -foffload=disable
    LDFLAGS   += -foffload=disable
  endif
endif
ifeq ($(CONFIG),gpu)
  CC          := nvc
  CFLAGS      := -O3 -g -mp=gpu,multicore -gpu=$(GPU_ARCH) -Minfo=accel
  LDFLAGS     := -mp=gpu,multicore -gpu=$(GPU_ARCH)
endif
ifeq ($(CONFIG),ompcpu)
  CC          := nvc
  CFLAGS      := -O3 -g -mp=multicore -DCPU_ONLY_MODE
  LDFLAGS     := -mp=multicore
endif
ifeq ($(CONFIG),clanggpu)
  CC          := clang
  SM_ARCH     := sm_$(patsubst cc%,%,$(GPU_ARCH))
  CFLAGS      := -O3 -g -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
                 -Xopenmp-target=nvptx64-nvidia-cuda -march=$(SM_ARCH) $(WARN)
  LDFLAGS     := -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
endif

CFLAGS        += $(EXTRA_CFLAGS)
LDFLAGS       += $(EXTRA_LDFLAGS)
LDLIBS        := -lm

# ---------------------------------------------------------------- sources
KERNEL_SRC := \
  $(GPUDIR)/core_kernels.c \
  $(GPUDIR)/gpu_domain_core.c \
  $(GPUDIR)/gpu_kernels.c \
  $(GPUDIR)/gpu_boundaries.c \
  $(GPUDIR)/gpu_halo.c \
  $(GPUDIR)/gpu_flop.c \
  $(GPUDIR)/gpu_rate_operator.c \
  $(GPUDIR)/gpu_inlet_operator.c \
  $(GPUDIR)/gpu_culvert_operator.c \
  $(GPUDIR)/gpu_max_quantities_operator.c

BENCH_SRC  := src/bench.c src/mesh.c src/setup.c src/snapshot.c src/gcc_offload_shim.c

OBJDIR     := build/$(CONFIG)
BIN        := bin/bench_$(CONFIG)
OBJ        := $(addprefix $(OBJDIR)/,$(notdir $(KERNEL_SRC:.c=.o)) $(notdir $(BENCH_SRC:.c=.o)))
DEP        := $(OBJ:.o=.d)

# ---------------------------------------------------------------- targets
.PHONY: all cpu gpu ompcpu clanggpu build clean verify run
all: cpu gpu

cpu ompcpu gpu clanggpu:
	@$(MAKE) --no-print-directory CONFIG=$@ build

build: $(BIN)
	@echo "built $(BIN)  [$(CC) $(CFLAGS)]"

$(BIN): $(OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(OBJDIR)/%.o: $(GPUDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(OBJDIR)/%.o: src/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(OBJDIR) bin:
	@mkdir -p $@

# Cross-check the GPU build against the CPU build on an identical run.
NX     ?= 200
NY     ?= 200
STEPS  ?= 50
CASE   ?= dam
verify: cpu gpu
	./bin/bench_cpu --nx $(NX) --ny $(NY) --steps $(STEPS) --case $(CASE) --save build/golden.bin
	./bin/bench_gpu --nx $(NX) --ny $(NY) --steps $(STEPS) --case $(CASE) --check build/golden.bin

run: $(BIN)
	./$(BIN) --nx $(NX) --ny $(NY) --steps $(STEPS) --case $(CASE) --phases

clean:
	rm -rf build bin

-include $(DEP)
