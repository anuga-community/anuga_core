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
  # The CUDA experiment is a PURE nvcc shared library, dlopen()ed at runtime:
  # every way of linking CUDA objects into this binary broke nvomp's offload
  # registration or ICEd the compiler.  See src/cuda_extrapolate.cu.
  SM_ARCH     := sm_$(patsubst cc%,%,$(GPU_ARCH))
  NVCC        ?= nvcc
  CUDA_SO     = $(OBJDIR)/cuextrap.so
endif
ifeq ($(CONFIG),ompcpu)
  CC          := nvc
  CFLAGS      := -O3 -g -mp=multicore -DCPU_ONLY_MODE
  LDFLAGS     := -mp=multicore
endif
# --- experimental targets for the AMD/Intel portability test (untested on
# --- real hardware yet; expect flag fixes on first contact) ---------------
ifeq ($(CONFIG),amdgpu)
  # ROCm: amdclang or mainline clang with AMDGPU offload.
  # AMD_ARCH: gfx90a (MI210/250), gfx942 (MI300); autodetected when possible.
  CC          ?= amdclang
  AMD_ARCH    ?= $(shell rocm_agent_enumerator 2>/dev/null | grep -v gfx000 | head -1)
  ifeq ($(strip $(AMD_ARCH)),)
    AMD_ARCH  := gfx90a
  endif
  CFLAGS      := -O3 -g -fopenmp --offload-arch=$(AMD_ARCH) $(WARN)
  LDFLAGS     := -fopenmp --offload-arch=$(AMD_ARCH)
endif
ifeq ($(CONFIG),intelgpu)
  # oneAPI: icx with SPIR-V offload (JIT for the present device).
  CC          := icx
  CFLAGS      := -O3 -g -fiopenmp -fopenmp-targets=spir64 $(WARN)
  LDFLAGS     := -fiopenmp -fopenmp-targets=spir64
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
LDLIBS        := -lm -ldl

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

# GEOM=fp32 builds the static-geometry arrays in single precision (an
# experiment: ~halves geometry traffic; NOT bit-compatible with fp64).
GEOM       ?= fp64
ifeq ($(GEOM),fp32)
  CFLAGS   += -DANUGA_GEOM_FP32
  GEOMSFX  := _fp32
endif

OBJDIR     := build/$(CONFIG)$(GEOMSFX)
BIN        := bin/bench_$(CONFIG)$(GEOMSFX)

# Objects depend on a stamp of the exact compile flags, so changing CFLAGS
# (or GPU_ARCH) forces a rebuild instead of silently mixing objects -- mixed
# flag sets have produced corrupted offload registries twice now.
FLAGSTAMP  := $(OBJDIR)/.flags.$(firstword $(shell echo '$(CC) $(CFLAGS)' | md5sum))
$(FLAGSTAMP): | $(OBJDIR)
	@rm -f $(OBJDIR)/.flags.* $(OBJDIR)/*.o $(OBJDIR)/*.d
	@touch $@
OBJ        := $(addprefix $(OBJDIR)/,$(notdir $(KERNEL_SRC:.c=.o)) $(notdir $(BENCH_SRC:.c=.o)))
DEP        := $(OBJ:.o=.d)

# ---------------------------------------------------------------- targets
.PHONY: all cpu gpu ompcpu clanggpu amdgpu intelgpu build clean verify run dist
all: cpu gpu

cpu ompcpu gpu clanggpu amdgpu intelgpu:
	@$(MAKE) --no-print-directory CONFIG=$@ build

build: $(BIN)
	@echo "built $(BIN)  [$(CC) $(CFLAGS)]"

$(BIN): $(OBJ) $(CUDA_SO) | bin
	$(CC) $(LDFLAGS) -o $@ $(OBJ) $(LDLIBS)

$(OBJDIR)/%.o: $(GPUDIR)/%.c $(FLAGSTAMP) | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(OBJDIR)/%.o: src/%.c $(FLAGSTAMP) | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(OBJDIR)/cuextrap.so: src/cuda_extrapolate.cu src/cuda_extrap.h | $(OBJDIR)
	$(NVCC) -O3 -arch=$(SM_ARCH) -shared -Xcompiler -fPIC $(INCLUDES) -o $@ $<

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

# Self-contained snapshot: everything needed to build and run the benchmark
# with nothing but make + a compiler -- the kernel sources are COPIED in, so
# the tarball is frozen at today's code (the in-tree build always compiles
# the live sources; prefer it whenever you have the repo).
DISTDIR := anuga-sw-miniapp
dist:
	rm -rf $(DISTDIR) $(DISTDIR).tar.gz
	mkdir -p $(DISTDIR)/kernels/gpu $(DISTDIR)/src $(DISTDIR)/tools
	cp $(GPUDIR)/*.c $(GPUDIR)/*.h $(DISTDIR)/kernels/gpu/
	cp $(SW)/sw_domain.h $(SW)/sw_domain_math.h $(DISTDIR)/kernels/
	cp $(UTIL)/anuga_typedefs.h $(DISTDIR)/kernels/
	cp src/*.c src/*.h src/*.cu $(DISTDIR)/src/
	cp tools/scaling_sweep.sh $(DISTDIR)/tools/
	cp README.md env.sh $(DISTDIR)/
	sed -e 's|^ANUGA_ROOT .*|ANUGA_ROOT := $$(abspath .)|' \
	    -e 's|$$(ANUGA_ROOT)/anuga/shallow_water/gpu|$$(ANUGA_ROOT)/kernels/gpu|' \
	    -e 's|$$(ANUGA_ROOT)/anuga/shallow_water|$$(ANUGA_ROOT)/kernels|' \
	    -e 's|$$(ANUGA_ROOT)/anuga/utilities|$$(ANUGA_ROOT)/kernels|' \
	    Makefile > $(DISTDIR)/Makefile
	git -C $(ANUGA_ROOT) rev-parse HEAD > $(DISTDIR)/KERNEL_COMMIT 2>/dev/null || true
	tar czf $(DISTDIR).tar.gz $(DISTDIR)
	rm -rf $(DISTDIR)
	@echo "wrote $(DISTDIR).tar.gz (kernels frozen at KERNEL_COMMIT)"

clean:
	rm -rf build bin $(DISTDIR) $(DISTDIR).tar.gz

-include $(DEP)
