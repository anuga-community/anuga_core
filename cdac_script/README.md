# JORGE's secret sauce 

Ok, buckle up. 

## convert.py 

Easy script to transform a tch to msh file thereby reducing storage. You can get a "help" by running the script without args:

```
(anuga_env_3.13) [jlv900@gadi-gpu-v100-0035 cdac_script]$ python convert.py

Convert .tsh (ASCII) mesh files to .msh (NetCDF binary) format

Binary .msh files load 10-20x faster than ASCII .tsh files.

Usage:
    python convert_tsh_to_msh.py input.tsh output.msh
    python convert_tsh_to_msh.py input.tsh  # Creates input.msh
```

For further use, make sure the msh file gets redirected as `${resolution}sqm.msh`

## partition_mahadani.py

This basically invokes metis to partition the domain and INITIALIZE it. It needs to be run from the `300_data` directory or the pahts inside the script
need to be ammended. 


```
usage: partition_mahanadi.py [-h] --mesh-size {100,300,900} --nprocs NPROCS [--output OUTPUT] [--chilka-stage CHILKA_STAGE] [--verbose]

Partition Mahanadi Delta mesh

options:
  -h, --help            show this help message and exit
  --mesh-size {100,300,900}
                        Mesh resolution: 100, 300, or 900 sqm
  --nprocs NPROCS       Number of partitions to create
  --output OUTPUT       Output directory for partition files
  --chilka-stage CHILKA_STAGE
                        Initial stage inside Chilka polygon (default: 0.15)
  --verbose             Verbose output
```

Things will get stored in a partitions/ directory (around 5.7 GB total for 900sqm).


## run_model_3_partitioned.py  

Loads the prepartitioned data from `partitions/` and gets going asap. IT has no nice CLI arguments, 
you'll have to go in an ponit to the partitions/ and also the grid size: 

```
MESH_SIZE = 900  # <-- CHANGE THIS for different mesh sizes
```

This is why it is nice to call things `${resolution}sqm.msh` so that I can just change that name and boom. 


## Building ANUGA GPU

Requirements:

- Preferrably the conda_env_3.13 environment, 3.12 was evil (memory leak)
- An nvidia-hpc-sdk install (tested 25.5 and 25.7)
- CUDA for GPU code generation
- Depending on the behaviour of your system `module load your-mpi` mpi4py seems to come with one, but be careful!

### Basic build (V100 - default)

```bash
pip install -e . --no-build-isolation -Csetup-args=-Dgpu=true
```

### Specifying GPU architecture

Use the `-Dgpu_arch` flag to target different GPUs:

```bash
# V100 (default - cc70)
pip install -e . --no-build-isolation -Csetup-args=-Dgpu=true

# A100 (cc80)
pip install -e . --no-build-isolation -Csetup-args=-Dgpu=true -Csetup-args=-Dgpu_arch=cc80

# H100 (cc90)
pip install -e . --no-build-isolation -Csetup-args=-Dgpu=true -Csetup-args=-Dgpu_arch=cc90
```

### Supported GPU architectures

| GPU | Architecture | Flag |
|-----|--------------|------|
| NVIDIA V100 | cc70 | `-Dgpu_arch=cc70` (default) |
| NVIDIA A100 | cc80 | `-Dgpu_arch=cc80` |
| NVIDIA H100 | cc90 | `-Dgpu_arch=cc90` |
| AMD MI100 | gfx908 | `-Dgpu_arch=gfx908` (requires clang/AOMP) |
| AMD MI210/MI250 | gfx90a | `-Dgpu_arch=gfx90a` (requires clang/AOMP) |
| AMD MI300X | gfx942 | `-Dgpu_arch=gfx942` (requires clang/AOMP) |

### Runtime settings

To ensure GPU execution: `export OMP_TARGET_OFFLOAD=mandatory` 

All unit tests pass, using `pytest --pyargs anuga`. It will be slow-ish because there's many of them.

*To enable the GPU code*: `domain.set_multiprocessor_mode(2)`

You should see something like:
```
GPU halo exchange initialized:
  Neighbors: 2
  Total send: 7662 elements
  Total recv: 7634 elements
GPU domain initialized: 4869676 elements
GPU Domain Info (rank 0/4):
  Elements: 4869676
  GPU initialized: 0
  GPU-aware MPI: 0
  Halo neighbors: 2
  Total send: 7662, Total recv: 7634
+==============================================================================+
| GPU interface initialized using OpenMP target offloading                    |
+==============================================================================+
GPU arrays mapped to device 0
```

GPU aware comms DO NOT WORK at the moment. I am tired (it is 23:11 as of writing)


## Multi GPU execution

The code is GPU aware by design, `mpirun -np X python run_my_sim.py` will use X GPUs. If you ask for more processes than there are GPUs you will 
oversubscribe and that could lead to performance degradation. Be wary. I have not tested this. 

To run on multi node: `mpirun -np 64 --bind-to core --map-by ppr:4:node python run_model_3_partitioned.py` 

For example that line would run a 16 node, 64 GPU job for the partitioned scheme. You would need a prepartitioned mesh on 64 processes.






