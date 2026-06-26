# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TileQR-MPI is an MPI-parallel implementation of **Tile QR factorization** (タイル QR 分解) — a tiled/blocked algorithm for QR decomposition of dense matrices in distributed-memory parallel computing. This is the numerical-linear-algebra problem the codebase is being built to solve; keep new code organized around tiled kernels (GEQRT / TSQRT / ORMQR / TSMQR-style operations) operating on a distributed 2D block-cyclic matrix.

## Current state

Working distributed Tile QR. The algorithm runs as an **SPMD owner-computes** loop — every rank walks the full task space (`GEQRT` / `LARFB` / `TSQRT` / `SSRFB`) and, for each task it owns (`A.owns(output tile)`), calls the corresponding **PLASMA core kernel** (real `double` arithmetic). Data movement uses **non-blocking `MPI_Isend`/`MPI_Irecv`** (into the per-node panel buffers): `GEQRT`→`LARFB` V/T along the process row, and `TSQRT`/`SSRFB` down the process column (sequentially per `i`) plus the `(i,k)` V/T row-distribution for `SSRFB`; see Architecture. Verified: a built-in **residual check** (`-DCHECK=ON`) reports `‖A − QR‖_F / ‖A‖_F ≈ 2e-16` across all process-grid shapes (1×1, 4×1, 2×2, 3×2, 2×3, 1×6, 3×3), for square/rectangular and non-tile-divisible inputs and varied `ts`/`ib`; the value is identical across grids (the factored result is distribution-independent). There is **no CTest harness yet** (the check is opt-in at build time).

Files:
- `tile_matrix.hpp` — header-only distributed-matrix layer (`namespace tileqr`). `ProcessGrid` (P×Q grid, row-major `rank = myrow*Q + mycol`) and `TileMatrix` (descA-equivalent descriptor + node-local tile storage and per-tile T factors in `std::unique_ptr<double[]>`). See "Architecture" below.
- `main.cpp` — builds a `ProcessGrid` + `TileMatrix`, fills tiles with deterministic data, and runs the factorization with PLASMA kernels + MPI.
- `residual_check.hpp` — `-DCHECK=ON` only: gathers the original and factored tiles to rank 0 and computes the backward error `‖A − QR‖/‖A‖` (reconstructs `Q·R` with PLASMA `dormqr`/`dtsmqr` in `PlasmaNoTrans`, reverse sweep).
- `CMakeLists.txt` — CMake build (requires MPI and PLASMA; see Build & run).

## Build & run

Out-of-source under `build/` (gitignored):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
mpirun -np 4 ./build/tileqr 8 6 2 1 2 2   # args: global_m global_n ts ib [P Q]
```

`tileqr` args: `global_m global_n` (global element dims), `ts` (tile size = `nb`), `ib` (inner block), optional `P Q` (process grid; default `P=nprocs, Q=1`). `P*Q` must equal the MPI process count. On a workstation with few cores, add `--oversubscribe` to `mpirun`.

**Dependencies**: MPI and **PLASMA** (expected at `/opt/PLASMA`; override with `-Dplasma_DIR=<dir>/lib/cmake/plasma`). The installed `plasmaConfig.cmake` is empty (defines no targets), so CMake includes `plasmaTargets.cmake` directly to get `plasma::plasma_core_blas`; that target carries the include dir and links **OpenBLAS** (LAPACKE/CBLAS) + libgomp. Only the standalone `plasma_core_d*` kernels are used — no `plasma_init`/runtime. At runtime the PLASMA and OpenBLAS dylibs must be resolvable (they use absolute paths / rpath from the install).

**Output is gated behind the `COUT` flag and off by default.** Configure with `-DCOUT=ON` to enable detailed per-task tracing via [icecream-cpp](https://github.com/renatoGarcia/icecream-cpp) (single header, fetched at configure time by CMake `FetchContent`, pinned to v1.0.0). The `IC(...)` macro prints variable names + values (`op`, indices `(i,j,k)`, local tile coords, tile sizes) to **stderr**. All output sites are wrapped in `#ifdef COUT`; without it the program runs silently. Build it in a separate dir, e.g. `cmake -S . -B build-cout -DCOUT=ON`.

**Residual check** is gated behind `-DCHECK=ON` (independent of `COUT`, off by default). When on, after the factorization the program prints one line per run: `CHECK: ... ||A - QR||_F / ||A||_F = <value>` (≈ machine epsilon when correct). Implemented in `residual_check.hpp` (gather to rank 0 + serial `Q·R` reconstruction). E.g. `cmake -S . -B build-check -DCHECK=ON`.

## Architecture

- **`ProcessGrid`** mirrors a BLACS context + `gridinfo`: holds `comm`, `P`/`Q`, `myrow`/`mycol`, `rank`. Shared across multiple matrices (A, plus the V/T factors later).
- **`TileMatrix`** mirrors ScaLAPACK `descA` *and* owns the node-local data. Distribution is 2D block-cyclic with **one tile = one block** (so `nb` plays the role of `MB_=NB_`). Tiles are square `nb×nb` (edge tiles smaller); tile interiors are **column-major** so LAPACK/BLAS kernels apply directly. Key maps: `owner(I,J)`, `owns(I,J)`, `local_row/col` (= `I/P`, `J/Q`, independent of `rsrc/csrc`), `tile_rows/cols` (edge sizes), `tile(I,J)`/`local_tile(il,jl)` (data pointers). Local tile counts come from `local_tile_count` (ScaLAPACK `numroc` with block width 1). It also holds the Tile-QR **T factors** (triangular Householder factors) in a second per-tile buffer `T`: `ib*nb` (= `ib*ts`) per tile, `ib*nb*mtloc*ntloc` total, column-major with leading dim `ldt=ib`, accessed via `tile_T(I,J)`/`local_tile_T(il,jl)`. `GEQRT`/`TSQRT` produce these; `LARFB`/`SSRFB` consume them. It also owns per-node **communication buffers** (separate from the distributed data; sized for a whole panel row/column so multiple transfers can be in flight for future async comm): `comm_top` (`nb*nb*nt`, borrowed top-row `(k,*)` tiles, slot by tile-column via `top_buf(j)`), `comm_V` (`nb*nb*mt`, received V, slot by tile-row via `V_buf(i)`), and `comm_T` (`ib*nb*mt`, received T, via `T_buf(i)`).
- **Task convention**: each kernel is `KERNEL(i, j, k)` — at panel step `k`, output tile `(i,j)`. The executing rank is the owner of the output tile; the loop guards every task with `if (A.owns(i,j))` (owner-computes).
- **Communication (node-level, non-blocking)**: at step `k` the panel lives in process row `pr = owner_row(k)`. The owner of `(k,k)` (column `pc = owner_col(k)`) runs `GEQRT`, then `MPI_Isend`s V (`tile(k,k)`, `nb*nb`) and T (`tile_T(k,k)`, `ib*nb`) **once** to **each other process column in row `pr` that holds some `(k,j)` with `j>k`**, and does its own `LARFB` while those are in flight (`Waitall` before the TSQRT loop, since `TSQRT` overwrites `tile(k,k)`). A consumer node (`mycol != pc`, owns some `(k,j)`) `MPI_Irecv`s V,T **once** into `A.V_buf(k)`/`A.T_buf(k)`, `Wait`s, then applies `LARFB` to all its `(k,j)`. Managed per node, not per tile.
- **`TSQRT`/`SSRFB` communication (column-direction, sequential per `i`, non-blocking)**: for each `i = k+1..mt-1` in order — (1) **`TSQRT(i,k)`**: `owner(k,k)` lends `(k,k)` to `owner(i,k)` ("down" the column); the borrower `Irecv`s into `A.top_buf(k)`, runs `TSQRT`, and `Isend`s the updated `(k,k)` back. (2) **`SSRFB` V/T row-distribution**: `owner(i,k)` `Isend`s `(i,k)`'s V/T along process row `owner_row(i)` to each consumer column **once per `i`** (consumers `Irecv` into `A.V_buf(i)`/`A.T_buf(i)`). (3) **`SSRFB(i,j,k)`**: `owner(k,j)` lends `(k,j)` to `owner(i,j)`, which `Irecv`s into `A.top_buf(j)`, runs `SSRFB`, and `Isend`s `(k,j)` back. Distinct MPI tags per message class (V/T, `(k,k)`, `(i,k)` V/T, `(k,j)`). The received/borrowed buffers feed the kernels directly (the `(k,*)` owner uses its own local tiles). **Request management**: receives are waited before the consuming kernel; multicast V/T sends overlap local compute and are `Waitall`'d before the source tile is reused; the `(k,*)`-owner does `Isend`→`Wait`→`Irecv`→`Wait` (same buffer, so the lend must finish before the writeback overwrites it); writeback `Isend`s are deferred and waited per `top_buf` slot before reuse (and flushed at the end of each `k`-step via `top_back[]`). `Vik`/`Tik` are only formed on row-`pr_i` nodes (avoids forming a pointer to an unowned tile).
- **Local kernels (PLASMA `core`, double)**: `GEQRT`→`plasma_core_dgeqrt(m,n,ib, A,lld, T,ldt, tau, work)`; `LARFB`→`plasma_core_dormqr(PlasmaLeft,PlasmaTrans, m,n,k,ib, V,lld, T,ldt, C,lld, work, ldwork=n)`; `TSQRT`→`plasma_core_dtsqrt(m,n,ib, A1=(k,k),lld, A2=(i,k),lld, T,ldt, tau, work)`; `SSRFB`→`plasma_core_dtsmqr(PlasmaLeft,PlasmaTrans, m1=nb,n1, m2,n2, k, ib, A1=(k,j),lld, A2=(i,j),lld, V=(i,k),lld, T,ldt, work, ldwork=ib)`. Per-tile dims are `tile_rows`/`tile_cols`; `k=min(tile_rows(k),tile_cols(k))` for `dormqr`. Workspaces (matching PLASMA's driver): `tau[nb]` and `work[ib*nb]`, reused across kernels. These are the bare core routines (no OpenMP wrapper), so they run sequentially on the calling rank.

## Intended toolchain

The `.gitignore` is pre-configured for the expected toolchain, so follow it rather than introducing a different one:

- **CMake** — build system (`CMakeFiles/`, `CMakeCache.txt`, `compile_commands.json`, generated `Makefile` are all ignored). Add a `CMakeLists.txt` and build out-of-source under `build/`.
- **vcpkg** — C++ dependency management (`vcpkg_installed/` is ignored).
- Object/library/executable artifacts and Fortran module files (`*.mod`, `*.smod`) are ignored — the latter implies linking against a Fortran BLAS/LAPACK for the per-tile dense kernels.

Expected dependencies for the algorithm: an **MPI** implementation (e.g. OpenMPI/MPICH) and a **BLAS/LAPACK** provider for the local tile kernels.
