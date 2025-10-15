#include "solver_iface.hpp"
#include <vector>
#include <mpi.h>
#include <cstdio>
#include <algorithm>

#include "HYPRE.h"
#include "HYPRE_utilities.h"   // HYPRE_Initialize, HYPRE_BigInt/HYPRE_Int
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

// -------- Hypre API compatibility shims (support old & new names) --------
#ifndef HYPRE_IJMatrixInitialize_v2
#define HYPRE_IJMatrixInitialize_v2(A, mem) HYPRE_IJMatrixInitialize(A)
#endif
#ifndef HYPRE_IJVectorInitialize_v2
#define HYPRE_IJVectorInitialize_v2(V, mem) HYPRE_IJVectorInitialize(V)
#endif
#ifndef HYPRE_MEMORY_HOST
#define HYPRE_MEMORY_HOST 0
#endif

#ifndef HYPRE_PCGSetRecomputeResidualPeriod
// older hypre uses ...ResidualP
#define HYPRE_PCGSetRecomputeResidualPeriod HYPRE_PCGSetRecomputeResidualP
#endif

#ifndef HYPRE_BoomerAMGSetAggNumPaths
// older hypre uses SetNumPaths
#define HYPRE_BoomerAMGSetAggNumPaths HYPRE_BoomerAMGSetNumPaths
#endif

// -------- init + error helpers --------
static inline void hypre_maybe_init()
{
  static bool inited = false;
  if (!inited) { HYPRE_Initialize(); inited = true; }
}

static inline void hypre_chk(const char* where, int ierr)
{
  if (!ierr) return;
  char msg[1024] = {0};
  HYPRE_DescribeError(ierr, msg);
  int r = -1; MPI_Comm_rank(MPI_COMM_WORLD, &r);
  std::fprintf(stderr, "[rank %d] HYPRE error %d in %s: %s\n", r, ierr, where, msg);
  MPI_Abort(MPI_COMM_WORLD, ierr);
}
#define HYPRE_CHK(call) hypre_chk(#call, (call))

struct HypreSolver final : ISolver {
  Stats solve(const CSR &csr, const std::vector<double> &b_local,
              int N, int r0, int r1, double rtol,
              std::vector<double> &u_local, MPI_Comm comm) override
  {
    hypre_maybe_init();

    Stats S;
    const HYPRE_BigInt n_global = (HYPRE_BigInt) N * (HYPRE_BigInt) N;
    const HYPRE_BigInt ilower   = (HYPRE_BigInt) r0;
    const HYPRE_BigInt iupper   = (HYPRE_BigInt) (r1 - 1);

    // If this rank owns no rows, exit early
    if (r1 <= r0) {
      u_local.clear();
      S.iters = 0; S.rtol_achieved = 0.0; S.t_setup = 0.0; S.t_solve = 0.0;
      return S;
    }

    // Match column partition to local rows (safe default for PARCSR)
    const HYPRE_BigInt jlower = ilower;
    const HYPRE_BigInt jupper = iupper;

    // ---------------- Matrix (IJ / PARCSR) ----------------
    HYPRE_IJMatrix A;
    HYPRE_CHK( HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &A) );
    HYPRE_CHK( HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR) );

    // Preallocate explicit diag/offd sizes per local row
    std::vector<HYPRE_Int> diag_sz((size_t)csr.nrows, 0), offd_sz((size_t)csr.nrows, 0);
    HYPRE_Int offd_total = 0;
    for (int r = 0; r < csr.nrows; ++r) {
      const int s = csr.ia[r], e = csr.ia[r + 1];
      for (int p = s; p < e; ++p) {
        const HYPRE_BigInt cj = (HYPRE_BigInt) csr.ja[p];
        if (cj >= jlower && cj <= jupper) ++diag_sz[(size_t)r];
        else                               ++offd_sz[(size_t)r];
      }
      offd_total += offd_sz[(size_t)r];
    }
    HYPRE_CHK( HYPRE_IJMatrixSetDiagOffdSizes(A, diag_sz.data(), offd_sz.data()) );
    // Be permissive with off-proc inserts (some builds want a nonzero allowance)
    HYPRE_CHK( HYPRE_IJMatrixSetMaxOffProcElmts(A, std::max<HYPRE_Int>(1, offd_total)) );

    // v2 initializer (falls back via shim if not available)
    HYPRE_CHK( HYPRE_IJMatrixInitialize_v2(A, HYPRE_MEMORY_HOST) );

    // Fill with global indices
    const double t0 = MPI_Wtime();
    for (int r = 0; r < csr.nrows; ++r) {
      const int s = csr.ia[r], e = csr.ia[r + 1];
      HYPRE_Int nnz = (HYPRE_Int)(e - s);     // non-const pointer expected
      const double* vals = csr.a.data() + s;

      std::vector<HYPRE_BigInt> cols((size_t)nnz);
      for (HYPRE_Int k = 0; k < nnz; ++k)
        cols[(size_t)k] = (HYPRE_BigInt) csr.ja[s + k];

      const HYPRE_BigInt grow = (HYPRE_BigInt)(r0 + r);
      HYPRE_CHK( HYPRE_IJMatrixSetValues(A, 1, &nnz, &grow, cols.data(), vals) );
    }
    HYPRE_CHK( HYPRE_IJMatrixAssemble(A) );

    HYPRE_ParCSRMatrix parA = nullptr;
    HYPRE_CHK( HYPRE_IJMatrixGetObject(A, (void**)&parA) );

    // ---------------- RHS and solution vectors ----------------
    HYPRE_IJVector b, x;
    HYPRE_CHK( HYPRE_IJVectorCreate(comm, ilower, iupper, &b) );
    HYPRE_CHK( HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR) );
    HYPRE_CHK( HYPRE_IJVectorInitialize_v2(b, HYPRE_MEMORY_HOST) );
    for (int r = 0; r < csr.nrows; ++r) {
      const HYPRE_BigInt grow = (HYPRE_BigInt)(r0 + r);
      const double v = b_local[(size_t)r];
      HYPRE_CHK( HYPRE_IJVectorSetValues(b, 1, &grow, &v) );
    }
    HYPRE_CHK( HYPRE_IJVectorAssemble(b) );
    HYPRE_ParVector parb = nullptr;
    HYPRE_CHK( HYPRE_IJVectorGetObject(b, (void**)&parb) );

    HYPRE_CHK( HYPRE_IJVectorCreate(comm, ilower, iupper, &x) );
    HYPRE_CHK( HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR) );
    HYPRE_CHK( HYPRE_IJVectorInitialize_v2(x, HYPRE_MEMORY_HOST) );
    for (int r = 0; r < csr.nrows; ++r) {
      const HYPRE_BigInt grow = (HYPRE_BigInt)(r0 + r);
      const double z = 0.0;
      HYPRE_CHK( HYPRE_IJVectorSetValues(x, 1, &grow, &z) );
    }
    HYPRE_CHK( HYPRE_IJVectorAssemble(x) );
    HYPRE_ParVector parx = nullptr;
    HYPRE_CHK( HYPRE_IJVectorGetObject(x, (void**)&parx) );

    // ---------------- Solver: PCG + BoomerAMG ----------------
    HYPRE_Solver pcg=nullptr, amg=nullptr;
    HYPRE_CHK( HYPRE_ParCSRPCGCreate(comm, &pcg) );
    HYPRE_CHK( HYPRE_PCGSetTol(pcg, rtol) );
    HYPRE_CHK( HYPRE_PCGSetMaxIter(pcg, 1000) );
    HYPRE_CHK( HYPRE_PCGSetTwoNorm(pcg, 1) );
    HYPRE_CHK( HYPRE_PCGSetPrintLevel(pcg, 0) );
    HYPRE_CHK( HYPRE_PCGSetRecomputeResidualPeriod(pcg, 50) ); // shim maps to old name if needed

    HYPRE_CHK( HYPRE_BoomerAMGCreate(&amg) );
    HYPRE_CHK( HYPRE_BoomerAMGSetPrintLevel(amg, 0) );
    HYPRE_CHK( HYPRE_BoomerAMGSetCoarsenType(amg, 8) );         // HMIS
    HYPRE_CHK( HYPRE_BoomerAMGSetInterpType(amg, 6) );          // extended classical
    HYPRE_CHK( HYPRE_BoomerAMGSetStrongThreshold(amg, 0.25) );  // good for 2D Poisson
    HYPRE_CHK( HYPRE_BoomerAMGSetAggNumLevels(amg, 1) );        // 1 aggressive level
    HYPRE_CHK( HYPRE_BoomerAMGSetAggNumPaths(amg, 2) );         // shim maps to old name if needed
    HYPRE_CHK( HYPRE_BoomerAMGSetRelaxType(amg, 6) );           // symmetric GS
    HYPRE_CHK( HYPRE_BoomerAMGSetNumSweeps(amg, 1) );
    HYPRE_CHK( HYPRE_BoomerAMGSetMaxLevels(amg, 25) );
    // Strict preconditioner: a single V-cycle
    HYPRE_CHK( HYPRE_BoomerAMGSetTol(amg, 0.0) );
    HYPRE_CHK( HYPRE_BoomerAMGSetMaxIter(amg, 1) );

    HYPRE_CHK( HYPRE_PCGSetPrecond(pcg,
      (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
      (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, amg) );

    const double t1 = MPI_Wtime();
    HYPRE_CHK( HYPRE_ParCSRPCGSetup(pcg, parA, parb, parx) );
    const double t2 = MPI_Wtime();

    // Non-convergence (256) is non-fatal: still record stats
    {
      const int ierr = HYPRE_ParCSRPCGSolve(pcg, parA, parb, parx);
      if (ierr && ierr != 256) hypre_chk("HYPRE_ParCSRPCGSolve", ierr);
    }
    const double t3 = MPI_Wtime();

    int its = 0;      HYPRE_CHK( HYPRE_PCGGetNumIterations(pcg, &its) );
    double rel = 0.0; HYPRE_CHK( HYPRE_PCGGetFinalRelativeResidualNorm(pcg, &rel) );

    // ---------------- Pull solution back ----------------
    u_local.resize((size_t)csr.nrows);
    std::vector<HYPRE_BigInt> rows((size_t)csr.nrows);
    for (int r = 0; r < csr.nrows; ++r) rows[(size_t)r] = (HYPRE_BigInt)(r0 + r);
    std::vector<double> vals((size_t)csr.nrows, 0.0);
    HYPRE_Int nvals = (HYPRE_Int)csr.nrows;
    HYPRE_CHK( HYPRE_IJVectorGetValues(x, nvals, rows.data(), vals.data()) );
    for (int r = 0; r < csr.nrows; ++r) u_local[(size_t)r] = vals[(size_t)r];

    // Stats matching the PETSc flow
    S.iters = its;
    S.rtol_achieved = rel;
    S.t_setup = t2 - t0;
    S.t_solve = t3 - t2;

    // Cleanup
    HYPRE_CHK( HYPRE_ParCSRPCGDestroy(pcg) );
    HYPRE_CHK( HYPRE_BoomerAMGDestroy(amg) );
    HYPRE_CHK( HYPRE_IJMatrixDestroy(A) );
    HYPRE_CHK( HYPRE_IJVectorDestroy(b) );
    HYPRE_CHK( HYPRE_IJVectorDestroy(x) );

    return S;
  }
};

ISolver* create_solver() { return new HypreSolver(); }
