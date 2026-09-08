# Device-side initialisation for generated meshes (option C)

**STATUS 2026-09-03: Phases 0-2 landed and validated on V100 + CPU build**
(`--device-init`, src/setup_device.c).  dam/river bit-exact vs host-init;
dambumps/lake at 1 ulp (device exp vs glibc).  16M tris: setup 11.4 s ->
0.25 s, host peak 7.8 -> 1.2 GiB.  64M tris (29 GiB, near the V100 wall):
init 0.545 s.  Remaining: Phase 3 (Morton perm), meta-only mesh generation
(-32 B/tri host), option B for --mesh/--tiles.  Measurements in
standalone/README.md.

Goal: for `rectangular_cross` runs (every sweep, gate, and hero run), the host
never allocates, fills, or transfers any O(n) array. The domain is born on the
device. Setup drops from seconds-to-minutes to milliseconds, and host RAM
stops limiting how big a mesh one rank can hold.

Measured motivation (V100, nx=1000, 4M triangles):

    build 2.095 s (serial host) + map-to-device 1.237 s   vs   0.131 s timed loop
    host peak 525 B/tri vs device 488 B/tri  -> host holds a full copy
    at the H200 hero size (293.8M tris) this scales to ~4 min and ~154 GiB host

## Why C is cheaper here than it sounds

Three properties of the existing code do most of the work:

1. **Everything O(n) is analytic for the generated mesh.** Vertex coords,
   geometry (areas/normals/lengths/radii), and the case ICs (`bed_value`,
   `stage_value`) are pure per-triangle functions of (m, n, delta1, delta2,
   x0, y0, cell index). Even connectivity is closed-form: the cross pattern
   gives each cell 4 triangles (t=0 left, 1 bottom, 2 right, 3 top) whose
   neighbours follow arithmetically from (i, j, t). The `edgemap` hash exists
   only for `--mesh`/`--tiles`, which keep the host path.

2. **`map(to:)` on already-present data does not copy.** OpenMP semantics: if
   the array is already mapped, `enter data map(to:)` only bumps the refcount.
   So the standalone can pre-map every big array with `map(alloc:)`, fill it
   with device kernels, then call the *unmodified* production
   `gpu_domain_map_arrays()` — all its `map(to:)` clauses become no-ops.
   **Zero changes to `anuga/shallow_water/gpu/*.c`.**

3. **`breg()` uses calloc.** Untouched calloc pages are virtual only. If the
   device path never writes the host mirrors, host RSS stays near zero with
   no allocator changes. (`--save`/`--check` still work: `target update from`
   faults the pages in only when asked.)

Bonus: written against `gpu_omp_macros.h`, the same init source compiles in
CPU_ONLY_MODE to `omp parallel for` filling host arrays — i.e. option A
(parallel host init) falls out of C for free on the CPU build.

## Structure

New standalone-only file `src/setup_device.c` + `--device-init` flag in
bench.c (default off until validated; generated meshes only — `--mesh`,
`--tiles`, and MPI slab fall back to the host path with a notice).

    bench.c:  if (device_init && generated_mesh) {
                  bench_domain_build_scalars(...);      // scalars only, no O(n) fills
                  bench_device_premap(&B->GD);          // enter data map(alloc:) all big arrays
                  bench_device_init(&B->GD, P, m, n, perm);   // device kernels fill them
              }
              bench_domain_to_device(...);              // unchanged; map(to:) now no-ops

### Phase 0 — scaffolding (~2 h)

- `--device-init` flag; refuse politely with `--mesh`/`--tiles`/MPI for now.
- Split `bench_domain_build` so the scalar section is callable alone;
  the O(n) allocations stay (calloc = virtual), the fill loops are skipped.
- `bench_device_premap`: `enter data map(alloc:)` for the same array list
  `gpu_domain_map_arrays` maps (keep the two lists adjacent in the source
  with a comment binding them).
- Matching `map(release:)` on teardown so refcounts return to zero
  (each array will be present twice: once from premap, once from the
  production map). Verify with `libomptarget` present-table debug or a
  map/unmap unit run.

### Phase 1 — geometry + quantities kernels (~3 h)

One `omp target teams loop` over k for each fill, same expression order as
the host loops (copy the bodies verbatim; `bed_value`/`stage_value`/
`river_bed` get `declare target`):

- vertex coords straight from the closed form: cell c = k/4, t = k%4,
  i = c/n, j = c%n, corner/centre nodes as in `rectangular_cross_core`
  (keep `delta1*(double)(i+i_off)+x0` exactly — the slab bit-exactness
  contract depends on that expression).
- geometry: areas, normals, edgelengths, centroids, edge midpoints, radii.
- quantities: bed/stage/height at vertices/edges/centroids, friction,
  momenta = 0. Work/update/backup arrays: device memset (they are
  calloc-zero on the host path).

### Phase 2 — connectivity + boundary numbering (~1 day, the real work)

- neighbours / neighbour_edges / surrogate_neighbours / number_of_boundaries
  in closed form from (i, j, t): intra-cell edges pair through the centre
  vertex; inter-cell edges pair left<->right (i±1) and bottom<->top (j±1);
  domain-rim edges get -1. Derive the 4x3 table once on paper, encode it,
  and let the gate below prove it.
- Boundary numbering must reproduce the host scan exactly
  (`for k, for e: if neighbour < 0 -> -(++idx)`, bcells/bedges in that
  order). Do it with a **device prefix scan** over the 3n edge slots rather
  than a closed-form count: the scan survives any cell permutation (Morton)
  unchanged, the closed form does not. nb ≈ 2(m+n) is tiny; copy bcells/
  bedges back to host for `gpu_reflective_init` (small D2H, unavoidable,
  negligible).
- scatter `owned_edges` list: same scan pattern as the host compaction
  (boundary slot or larger-index side), device-side.

### Phase 3 — Morton / permuted order (~2 h)

`--order morton` permutes whole cells before the build. Keep the host qsort
(it is O(cells log cells) on 16 B entries — 1.2 GiB transient at hero size,
acceptable; a device sort is a later nicety) and ship only
`perm[new_cell] = old_cell` (8 B/cell = 2 B/tri). Every Phase 1/2 kernel
indexes through perm to find (i, j). `--order random` stays host-path.

### Phase 4 — validation + measurement (~3 h)

The acceptance ladder, in order:

1. `--device-init --save` vs host-init `--save`: 4 cases x {row, morton}
   x nx {200, 1000}. Target **bitwise**; record per-field results.
   Known risks to bitwise: FMA contraction differing between host and
   device codegen, device `exp()` (dambumps/lake bed) and `hypot()` (radii)
   vs glibc. Decision point if any field misses bitwise: either align the
   formula in BOTH paths and re-freeze goldens (one-line, e.g. hypot ->
   sqrt form), or accept and document a per-field roundoff tolerance.
   Do not silently widen the default gate.
2. Full correctness ladder with `--device-init`: lake-at-rest, scatter==cell,
   active-set==full, `make verify` (CPU golden vs GPU device-init).
3. Throughput unchanged: same 19-size sweep row at 2-3 sizes (same kernels,
   same data — any drift means the init wrote something different).
4. Measure the win: build_s / map_s / host_peak_bytes at nx 1000/2000/5000
   plus the V100 ceiling size, host vs device init. The sweep CSV already
   carries build_s, map_s, host_peak_bytes — the cross-vendor init
   comparison then falls out of the campaign for free.

## Scope exclusions (explicit)

- `--mesh FILE`, `--tiles`, MPI slab: host path unchanged. (The follow-up
  for these is option B — ship coords+connectivity ~100 B/tri, derive the
  rest on device. Same kernels as Phase 1, different input.)
- `--order random`, `--cuda-extrap`, rain-grid replay: host path.
- No changes to `anuga/shallow_water/gpu/*.c` (production sources). The
  premap trick is what makes that possible.

## Sequencing vs SCA27

Off the paper's critical path — setup time is outside every reported metric.
Do it AFTER the cross-vendor campaigns are fired (falls in the 14-20 Sep
window), because editing setup.c before the campaign risks perturbing the
goldens every cross-vendor claim rests on. If it lands cleanly it earns one
paragraph in §3 ("the mesh is born on the device; the same init kernels run
on all five devices") and an init-time column in the cross-vendor table.

Expected result at 4M on V100: setup 3.33 s -> ~0.15 s; host peak
~2.1 GiB -> ~tens of MiB. At the H200 hero size: ~4 min -> seconds, and the
~154 GiB host copy disappears.
