# Poisson 2D (PETSc vs HYPRE) — Minimal Test

This bundle builds 5 executables from a shared core. It solves `-Δu=f` on `(0,1)^2` with Dirichlet 0 BCs using the manufactured solution `u*(x,y)=sin(pi x) sin(pi y)` so `f=2*pi^2*u*`. It reports L2/Linf error, iterations, and timing.


| Executable                      | Matrix / Grid storage                | Solver       | Preconditioner                      |
|---------------------------------|--------------------------------------|--------------|-------------------------------------|
| `poisson_petsc`                 | PETSc **AIJ (CSR)**                  | **CG**       | **GAMG**      (algebraic multigrid) |
| `poisson_hypre`                 | HYPRE **ParCSR**            (via IJ) | **PCG**      | **BoomerAMG** (algebraic multigrid) |
| `poisson_petsc_struct`          | PETSc **DMDA** (structured 2-D grid) | **CG**       | **PCMG**      (geometric multigrid) |
| `poisson_hypre_struct`          | HYPRE **Struct**   (5-point stencil) | **PFMG**     | *(none; PFMG is the solver)*        |
| `poisson_hypre_struct_bicgstab` | HYPRE **Struct**   (5-point stencil) | **BiCGSTAB** | **PFMG**   (strict, 1 V-cycle/iter) |

When to use which
- **Structured grids (uniform Poisson):** Prefer **HYPRE/Struct + PFMG** or **PETSc/DMDA + PCMG** → fastest and most memory-friendly.
- **Unstructured/irregular meshes or complex coefficients:** Use **ParCSR/IJ + AMG** (PETSc **GAMG** or HYPRE **BoomerAMG**) → robust without geometric info.
- **Nonsymmetric or mildly non-SPD systems on structured grids:** **BiCGSTAB + PFMG preconditioner** is a solid choice.
- **SPD with good preconditioner:** **CG** is typically the most efficient Krylov method.


## Prereqs

- CMake ≥ 3.16
- MPI toolchain (`mpicc/mpicxx`, `mpirun`)
- PETSc (built with MPI, see instructions below)
- HYPRE (built with MPI, see instructions below)

Replace `$HOME` with your preferred path in the installation/build commands.

### Install PETSc

```bash
# Install PETSC
cd $HOME
git clone -b release https://gitlab.com/petsc/petsc.git petsc
./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=0 --download-f2cblaslapack=1 --with-debugging=0 COPTFLAGS='-O2 -march=native -mtune=native' CXXOPTFLAGS='-O2 -march=native -mtune=native'
make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux-c-opt all
make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux-c-opt check
```

### Install HYPRE

```bash
cd $HOME
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHYPRE_ENABLE_MPI=ON -DHYPRE_ENABLE_OPENMP=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$HOME/hypre/install
cmake --build build -j
cmake --install build
```

## Build

```bash
cmake -S . -B build -DENABLE_PETSC=ON -DENABLE_HYPRE=ON -DENABLE_PETSC_STRUCT=ON -DENABLE_HYPRE_STRUCT=ON -DENABLE_HYPRE_STRUCT_BICGSTAB=ON -DPETSC_DIR=$HOME/petsc -DPETSC_ARCH=arch-linux-c-opt -DCMAKE_BUILD_TYPE=Release -DHYPRE_DIR=$HOME/hypre/install/lib/cmake/HYPRE
cmake --build build -j
```

## Run

```bash
# PETSc backend
mpirun -n 4 ./build/poisson_petsc --N 1024 --rtol 1e-8

# HYPRE backend
mpirun -n 4 ./build/poisson_hypre --N 1024 --rtol 1e-8

# PETSc DMDA + PCMG (pass MG options)
mpirun -n 4 ./build/poisson_petsc_struct --N 1024 --rtol 1e-8 -ksp_type cg -pc_type mg -pc_mg_levels 6 -pc_mg_galerkin pmat -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_coarse_ksp_type preonly -mg_coarse_pc_type lu

# HYPRE Struct + PFMG
mpirun -n 4 ./build/poisson_hypre_struct --N 1024 --rtol 1e-8

# HYPRE Struct + BiCGSTAB
mpirun -n 4 ./build/poisson_hypre_struct_bicgstab --N 1024 --rtol 1e-8


```

CLI flags:
- `--N`   : number of interior points per dimension (global unknowns = N x N)
- `--rtol`: relative tolerance (solver stopping target)


Output:
```bash
$ mpirun -n 4 ./build/poisson_petsc --N 1024 --rtol 1e-8
Backend: (unknown)
=== Poisson 2D N=1024 (n=1048576), ranks=4 ===
rtol target: 1.000e-08
Setup time:  0.043029 s
Solve time:  1.041107 s
Total time:  1.095812 s
Iters:       12
Rel. res:    1.507e-06
Errors:      L2 = 3.914343e-07, Linf = 7.843982e-07


$ mpirun -n 4 ./build/poisson_hypre --N 1024 --rtol 1e-8
Backend: (unknown)
=== Poisson 2D N=1024 (n=1048576), ranks=4 ===
rtol target: 1.000e-08
Setup time:  0.221744 s
Solve time:  0.455112 s
Total time:  0.688121 s
Iters:       19
Rel. res:    4.645e-09
Errors:      L2 = 3.914182e-07, Linf = 7.828374e-07


$ mpirun -n 4 ./build/poisson_petsc_struct --N 1024 --rtol 1e-8 -ksp_type cg -pc_type mg -pc_mg_levels 6 -pc_mg_galerkin pmat -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_coarse_ksp_type preonly -mg_coarse_pc_type lu
Backend: (unknown)
=== Poisson 2D N=1024 (n=1048576), ranks=4 ===
rtol target: 1.000e-08
Setup time:  0.042291 s
Solve time:  0.986335 s
Total time:  1.046358 s
Iters:       12
Rel. res:    1.507e-06
Errors:      L2 = 3.914343e-07, Linf = 7.843982e-07


$ mpirun -n 4 ./build/poisson_hypre_struct --N 1024 --rtol 1e-8
Backend: (unknown)
=== Poisson 2D N=1024 (n=1048576), ranks=4 ===
rtol target: 1.000e-08
Setup time:  0.015045 s
Solve time:  0.190085 s
Total time:  0.250992 s
Iters:       18
Rel. res:    7.488e-09
Errors:      L2 = 3.914120e-07, Linf = 7.828152e-07


$ mpirun -n 4 ./build/poisson_hypre_struct_bicgstab --N 1024 --rtol 1e-8
Backend: (unknown)
=== Poisson 2D N=1024 (n=1048576), ranks=4 ===
rtol target: 1.000e-08
Setup time:  0.013841 s
Solve time:  0.193523 s
Total time:  0.248246 s
Iters:       7
Rel. res:    5.320e-10
Errors:      L2 = 3.914182e-07, Linf = 7.828345e-07
```


Results:

- Solve time (lower is better) — N=1024, 4 ranks

| Solver                         | Time (s) | Bar                            |
|--------------------------------|---------:|--------------------------------|
| HYPRE (Struct + PFMG)          | 0.190085 | █████                          |
| HYPRE (Struct + BiCGSTAB)      | 0.193523 | ██████                         |
| HYPRE (ParCSR + BoomerAMG)     | 0.455112 | █████████████                  |
| PETSc (DMDA + PCMG)            | 0.986335 | ████████████████████████████   |
| PETSc (CSR + GAMG)             | 1.041107 | ██████████████████████████████ |

_Scaled to slowest (1.041107 s)_


## Notes

- PETSc honors runtime options via `KSPSetFromOptions` (e.g. `-ksp_monitor -pc_type gamg`).
- Both backends share matrix/RHS/error computations for a fair comparison.
- Switch to periodic BCs or variable coefficients by editing `src/core.cpp` only.
