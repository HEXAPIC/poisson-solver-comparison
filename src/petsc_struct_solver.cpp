#include "solver_iface.hpp"
#include <vector>
#include <mpi.h>

// PETSc (DMDA + KSP/PCMG)
#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>

struct PETScStructSolver final : ISolver {
  Stats solve(const CSR& /*unused*/, const std::vector<double>& b_local,
              int N, int r0, int r1, double rtol,
              std::vector<double>& u_local, MPI_Comm /*comm*/) override
  {
    Stats S{};

    // Make sure PETSc is initialized
    PetscBool petsc_init = PETSC_FALSE;
    PetscInitialized(&petsc_init);
    if (!petsc_init) { PetscInitialize(NULL, NULL, NULL, NULL); }

    // DMDA for an N x N interior grid, 5-point stencil (width=1), no physical boundaries (Dirichlet implied)
    DM da;
    DMDACreate2d(PETSC_COMM_WORLD,
                 DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                 DMDA_STENCIL_STAR, N, N,
                 PETSC_DECIDE, PETSC_DECIDE,
                 1, /* dof */
                 1, /* stencil width */
                 NULL, NULL, &da);
    DMSetUp(da);

    // Matrix & vectors
    Mat A; DMCreateMatrix(da, &A);  // preallocated for 5-pt stencil
    Vec b, x;
    DMCreateGlobalVector(da, &x);
    VecDuplicate(x, &b);
    VecSet(x, 0.0);

    // Assemble A via stencil
    DMDALocalInfo info; DMDAGetLocalInfo(da, &info);
    const PetscReal h = 1.0 / (N + 1);
    const PetscScalar invh2 = 1.0 / (h * h);

    double t0 = MPI_Wtime();
    for (PetscInt i = info.xs; i < info.xs + info.xm; ++i) {
      for (PetscInt j = info.ys; j < info.ys + info.ym; ++j) {
        MatStencil row; row.i = i; row.j = j;
        MatStencil cols[5]; PetscScalar vals[5]; PetscInt n = 0;

        // center
        cols[n].i = i; cols[n].j = j; vals[n] = 4.0 * invh2; ++n;
        // west/east
        if (i - 1 >= 0)       { cols[n].i = i - 1; cols[n].j = j; vals[n] = -invh2; ++n; }
        if (i + 1 <  info.mx) { cols[n].i = i + 1; cols[n].j = j; vals[n] = -invh2; ++n; }
        // south/north
        if (j - 1 >= 0)       { cols[n].i = i; cols[n].j = j - 1; vals[n] = -invh2; ++n; }
        if (j + 1 <  info.my) { cols[n].i = i; cols[n].j = j + 1; vals[n] = -invh2; ++n; }

        MatSetValuesStencil(A, 1, &row, n, cols, vals, INSERT_VALUES);
      }
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // Assemble b: insert our local (r0..r1) chunk by global ID g = i*N + j
    for (int g = r0; g < r1; ++g) {
      const PetscInt G = (PetscInt) g;
      const PetscScalar bv = (PetscScalar) b_local[(size_t)(g - r0)];
      VecSetValue(b, G, bv, INSERT_VALUES);
    }
    VecAssemblyBegin(b); VecAssemblyEnd(b);

    // KSP + PCMG (geometric multigrid)
    KSP ksp; PC pc;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPCG);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCMG);

    // Attach DM so PETSc can build the multigrid hierarchy automatically
    KSPSetDM(ksp, da);
    KSPSetDMActive(ksp, PETSC_FALSE);  // we assembled A ourselves

    KSPSetTolerances(ksp, rtol, 0.0, PETSC_DEFAULT, 1000);

    // Good defaults; allow command-line tuning
    // Example at run-time:
    //   -pc_mg_levels 6 -pc_mg_type multiplicative -pc_mg_galerkin pmat
    //   -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi
    //   -mg_coarse_ksp_type preonly -mg_coarse_pc_type lu
    KSPSetFromOptions(ksp);

    double t1 = MPI_Wtime();
    KSPSolve(ksp, b, x);
    double t2 = MPI_Wtime();

    // Pull solution in the same global order [r0, r1)
    const PetscInt nloc = (PetscInt)(r1 - r0);
    std::vector<PetscInt> gids((size_t)nloc);
    for (PetscInt k = 0; k < nloc; ++k) gids[(size_t)k] = (PetscInt)(r0 + (int)k);
    std::vector<PetscScalar> xvals((size_t)nloc, 0.0);
    VecGetValues(x, nloc, gids.data(), xvals.data());
    u_local.resize((size_t)nloc);
    for (PetscInt k = 0; k < nloc; ++k) u_local[(size_t)k] = (double)xvals[(size_t)k];

    // Stats
    PetscInt its = 0; KSPGetIterationNumber(ksp, &its);
    PetscReal rn = 0; KSPGetResidualNorm(ksp, &rn);
    S.iters = (int)its;
    S.rtol_achieved = (double)rn;   // absolute final residual norm
    S.t_setup = t1 - t0;
    S.t_solve = t2 - t1;

    // Cleanup
    KSPDestroy(&ksp);
    VecDestroy(&x); VecDestroy(&b);
    MatDestroy(&A);
    DMDestroy(&da);

    PetscBool fin = PETSC_FALSE;
    PetscFinalized(&fin);
    if (!fin) PetscFinalize();

    return S;
  }
};

ISolver* create_solver() { return new PETScStructSolver(); }
