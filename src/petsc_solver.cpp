#include "solver_iface.hpp"
#include <vector>
#include <mpi.h>

// Pull in explicit PETSc headers
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>

struct PETScSolver final : ISolver {
  Stats solve(const CSR &csr, const std::vector<double> &b_local,
              int N, int r0, int r1, double rtol,
              std::vector<double> &u_local, MPI_Comm /*comm*/) override {
    Stats S;

    // Initialize PETSc if needed
    PetscBool petsc_initialized = PETSC_FALSE;
    PetscInitialized(&petsc_initialized);
    if (!petsc_initialized) {
      PetscInitialize(NULL, NULL, NULL, NULL);
    }

    const PetscInt n_local  = (PetscInt)csr.nrows;
    const PetscInt n_global = (PetscInt)N * (PetscInt)N;

    Mat A; Vec b, x;

    // ---- Matrix creation (version-robust) ----
    // Use MatCreateAIJ which maps to seq/mpi internally; preallocate ~5 nnz/row.
    // d_nz = expected diag nnz/row, o_nz = expected offdiag nnz/row.
    MatCreateAIJ(PETSC_COMM_WORLD,
                 n_local, n_local, n_global, n_global,
                 5, NULL, 5, NULL, &A);

    // ---- Vectors ----
    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, n_local, n_global);
    VecSetFromOptions(b);

    VecDuplicate(b, &x);
    VecSet(x, 0.0);

    double t0 = MPI_Wtime();

    // Assemble matrix with global indices
    for (PetscInt r = 0; r < n_local; ++r) {
      const int s = csr.ia[r], e = csr.ia[r + 1];
      std::vector<PetscInt> cols(e - s);
      std::vector<PetscScalar> vals(e - s);
      for (int p = s; p < e; ++p) {
        cols[p - s] = (PetscInt)csr.ja[p];
        vals[p - s] = (PetscScalar)csr.a[p];
      }
      const PetscInt grow = (PetscInt)(r0 + (int)r);
      MatSetValues(A, 1, &grow, (PetscInt)(e - s), cols.data(), vals.data(), INSERT_VALUES);
    }
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // RHS assembly
    for (PetscInt r = 0; r < n_local; ++r) {
      const PetscInt grow = (PetscInt)(r0 + (int)r);
      const PetscScalar bv = (PetscScalar)b_local[(size_t)r];
      VecSetValue(b, grow, bv, INSERT_VALUES);
    }
    VecAssemblyBegin(b); VecAssemblyEnd(b);

    // KSP setup
    KSP ksp; PC pc;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPCG);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCGAMG);                  // PETSc AMG
    KSPSetTolerances(ksp, rtol, 0.0, PETSC_DEFAULT, 1000);
    KSPSetFromOptions(ksp);                 // allow -ksp_monitor etc.

    double t1 = MPI_Wtime();
    KSPSolve(ksp, b, x);
    double t2 = MPI_Wtime();

    // Pull solution back (using global IDs)
    u_local.resize((size_t)n_local);
    std::vector<PetscInt> gids((size_t)n_local);
    for (PetscInt r = 0; r < n_local; ++r) gids[(size_t)r] = (PetscInt)(r0 + (int)r);
    std::vector<PetscScalar> vals((size_t)n_local, 0.0);
    VecGetValues(x, n_local, gids.data(), vals.data());
    for (PetscInt r = 0; r < n_local; ++r) u_local[(size_t)r] = (double)vals[(size_t)r];

    // Stats
    PetscInt its; KSPGetIterationNumber(ksp, &its);
    PetscReal rn; KSPGetResidualNorm(ksp, &rn);
    S.iters = (int)its;
    S.rtol_achieved = (double)rn; // absolute residual norm
    S.t_setup = t1 - t0;
    S.t_solve = t2 - t1;

    // Cleanup
    KSPDestroy(&ksp);
    VecDestroy(&x); VecDestroy(&b); MatDestroy(&A);

    // Finalize PETSc only if it hasn't been finalized yet
    PetscBool fin = PETSC_FALSE;
    PetscFinalized(&fin);
    if (!fin) PetscFinalize();

    return S;
  }
};

ISolver* create_solver() { return new PETScSolver(); }
