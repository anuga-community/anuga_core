"""mpi4py-free ctypes driver for the dump-based portable runtime.

Reads an .adm partition via the C loader (anuga_dump_load) and drives the evolve
loop. MPI is owned by the C library (anuga_mpi_init / MPI_COMM_WORLD from the
system launcher), so this imports NO mpi4py - it runs under any python3 launched
by the system mpirun, with no conda. The only dependency is the standard library
(ctypes); numpy is optional (used only to write results).

    mpirun -n N python3 standalone/python/anuga_sw_dump.py <dump_dir> <case> [out_dir]

reads  <dump_dir>/<case>_P<N>_<rank>.adm
writes <out_dir>/<case>_P<N>_<rank>.result   (int64 count, then doubles)
"""
import ctypes
import os
import struct
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from anuga_sw import _find_library  # reuse the library locator

c_void_p, c_int, c_long, c_double = ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.c_double


def _bind(lib):
    sig = {
        "anuga_mpi_init": ([], None),
        "anuga_mpi_finalize": ([], None),
        "anuga_comm_world_rank": ([], c_int),
        "anuga_comm_world_size": ([], c_int),
        "anuga_dump_load": ([ctypes.c_char_p], c_void_p),
        "anuga_dump_domain": ([c_void_p], c_void_p),
        "anuga_dump_finaltime": ([c_void_p], c_double),
        "anuga_dump_timestepping_method": ([c_void_p], c_int),
        "anuga_dump_num_elements": ([c_void_p], c_long),
        "anuga_dump_get_owned_stage": ([c_void_p, ctypes.POINTER(c_double)], c_long),
        "anuga_dump_free": ([c_void_p], None),
        "anuga_domain_use_comm_world": ([c_void_p], None),
        "anuga_domain_map_to_device": ([c_void_p], c_int),
        "anuga_sync_to_device": ([c_void_p], None),
        "anuga_sync_from_device": ([c_void_p], None),
        "anuga_exchange_ghosts": ([c_void_p], None),
        "anuga_evolve_one_euler_step": ([c_void_p, c_double, c_int], c_double),
        "anuga_evolve_one_rk2_step": ([c_void_p, c_double, c_int], c_double),
        "anuga_evolve_one_rk3_step": ([c_void_p, c_double, c_int], c_double),
        "anuga_evolve_one_ader2_step": ([c_void_p, c_double, c_int, c_double], c_double),
    }
    for name, (a, r) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = a, r
    return lib


def main():
    dumpdir = sys.argv[1] if len(sys.argv) > 1 else "."
    name = sys.argv[2] if len(sys.argv) > 2 else "dam_break"
    outdir = sys.argv[3] if len(sys.argv) > 3 else dumpdir

    lib = _bind(ctypes.CDLL(_find_library()))
    lib.anuga_mpi_init()
    rank = lib.anuga_comm_world_rank()
    size = lib.anuga_comm_world_size()

    path = os.path.join(dumpdir, f"{name}_P{size}_{rank}.adm").encode()
    ld = lib.anuga_dump_load(path)
    if not ld:
        sys.stderr.write(f"rank {rank}: failed to load {path.decode()}\n")
        lib.anuga_mpi_finalize()
        sys.exit(1)

    dom = lib.anuga_dump_domain(ld)
    lib.anuga_domain_use_comm_world(dom)
    if not lib.anuga_domain_map_to_device(dom):
        sys.stderr.write(f"rank {rank}: map_to_device failed\n")
        sys.exit(1)
    lib.anuga_sync_to_device(dom)

    T = lib.anuga_dump_finaltime(ld)
    method = lib.anuga_dump_timestepping_method(ld)

    t = 0.0
    while t < T - 1e-12:
        max_dt = T - t
        if method == 1:
            dt = lib.anuga_evolve_one_rk2_step(dom, max_dt, 1)
        elif method == 2:
            dt = lib.anuga_evolve_one_rk3_step(dom, max_dt, 1)
        elif method == 3:
            dt = lib.anuga_evolve_one_ader2_step(dom, max_dt, 1, 0.0)
        else:
            dt = lib.anuga_evolve_one_euler_step(dom, max_dt, 1)
            if size > 1:
                lib.anuga_exchange_ghosts(dom)
        t += dt

    lib.anuga_sync_from_device(dom)

    n = lib.anuga_dump_num_elements(ld)
    buf = (c_double * (n if n else 1))()
    m = lib.anuga_dump_get_owned_stage(ld, buf)

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{name}_P{size}_{rank}.result")
    with open(out, "wb") as f:
        f.write(struct.pack("<q", m))
        if m:
            f.write(ctypes.string_at(buf, m * 8))
    s = sum(buf[i] for i in range(m))
    print(f"rank {rank}/{size}: t={t:.6f} owned={m} stage_sum={s:.10g} -> {out}", flush=True)

    lib.anuga_dump_free(ld)
    lib.anuga_mpi_finalize()


if __name__ == "__main__":
    main()
