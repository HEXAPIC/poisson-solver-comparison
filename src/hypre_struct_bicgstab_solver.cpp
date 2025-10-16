#include "solver_iface.hpp"

#include <vector>
#include <array>
#include <mpi.h>
#include <cstdio>
#include <algorithm>

// HYPRE (Struct interface)
#include "HYPRE.h"
#include "HYPRE_utilities.h"
#include "HYPRE_struct_mv.h"
#include "HYPRE_struct_ls.h"

// ---------- helpers ----------
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

static inline void push_box(std::vector<std::array<int,2>>& lo,
                            std::vector<std::array<int,2>>& hi,
                            int i0, int j0, int i1, int j1)
{
  lo.push_back(std::array<int,2>{i0, j0});
  hi.push_back(std::array<int,2>{i1, j1});
}

// Map a contiguous global index range [r0, r1) in row-major (g=i*N+j)
// to up to three rectangular boxes on the (i,j) grid.
static void build_boxes_from_range(int N, int r0, int r1,
                                   std::vector<std::array<int,2>>& lowers,
                                   std::vector<std::array<int,2>>& uppers)
{
  if (r1 <= r0) return;
  const int g0 = r0, g1 = r1 - 1;
  const int i0 = g0 / N, j0 = g0 % N;
  const int i1 = g1 / N, j1 = g1 % N;

  if (i0 == i1) {
    push_box(lowers, uppers, i0, j0, i1, j1);
    return;
  }

  if (j0 != 0) push_box(lowers, uppers, i0, j0, i0, N - 1);

  const int lo_i = (j0 == 0) ? i0 : i0 + 1;
  const int hi_i = (j1 == N - 1) ? i1 : i1 - 1;
  if (hi_i >= lo_i) push_box(lowers, uppers, lo_i, 0, hi_i, N - 1);

  if (j1 != N - 1) push_box(lowers, uppers, i1, 0, i1, j1);
}

// ---------- solver ----------
struct HypreStructBiCGSTABSolver final : ISolver {
  Stats solve(const CSR& /*unused*/, const std::vector<double>& b_local,
              int N, int r0, int r1, double rtol,
              std::vector<double>& u_local, MPI_Comm comm) override
  {
    hypre_maybe_init();

    Stats S{};
    if (r1 <= r0) { u_local.clear(); return S; }

    // ---- Grid definition
    HYPRE_StructGrid grid;
    HYPRE_CHK( HYPRE_StructGridCreate(comm, 2, &grid) );

    std::vector<std::array<int,2>> lowers, uppers;
    build_boxes_from_range(N, r0, r1, lowers, uppers);
    for (size_t b = 0; b < lowers.size(); ++b)
      HYPRE_CHK( HYPRE_StructGridSetExtents(grid, lowers[b].data(), uppers[b].data()) );
    HYPRE_CHK( HYPRE_StructGridAssemble(grid) );

    // ---- 5-point stencil: center, ±i, ±j
    HYPRE_StructStencil stencil;
    HYPRE_CHK( HYPRE_StructStencilCreate(2, 5, &stencil) );
    int off0[2] = { 0,  0};
    int off1[2] = {-1,  0};
    int off2[2] = {+1,  0};
    int off3[2] = { 0, -1};
    int off4[2] = { 0, +1};
    HYPRE_CHK( HYPRE_StructStencilSetElement(stencil, 0, off0) );
    HYPRE_CHK( HYPRE_StructStencilSetElement(stencil, 1, off1) );
    HYPRE_CHK( HYPRE_StructStencilSetElement(stencil, 2, off2) );
    HYPRE_CHK( HYPRE_StructStencilSetElement(stencil, 3, off3) );
    HYPRE_CHK( HYPRE_StructStencilSetElement(stencil, 4, off4) );

    // ---- Matrix & vectors
    HYPRE_StructMatrix A;
    HYPRE_CHK( HYPRE_StructMatrixCreate(comm, grid, stencil, &A) );
    HYPRE_CHK( HYPRE_StructMatrixInitialize(A) );

    HYPRE_StructVector b, x;
    HYPRE_CHK( HYPRE_StructVectorCreate(comm, grid, &b) );
    HYPRE_CHK( HYPRE_StructVectorInitialize(b) );
    HYPRE_CHK( HYPRE_StructVectorCreate(comm, grid, &x) );
    HYPRE_CHK( HYPRE_StructVectorInitialize(x) );

    const double h = 1.0 / (N + 1);
    const double invh2 = 1.0 / (h * h);

    // ---- Fill A (5-pt Laplacian with Dirichlet 0 BC)
    double t0 = MPI_Wtime();
    for (size_t bidx = 0; bidx < lowers.size(); ++bidx) {
      const int iL = lowers[bidx][0], jL = lowers[bidx][1];
      const int iU = uppers[bidx][0], jU = uppers[bidx][1];
      for (int i = iL; i <= iU; ++i) {
        for (int j = jL; j <= jU; ++j) {
          double vals[5]; int idxs[5]; int n = 0;
          vals[n] =  4.0 * invh2; idxs[n] = 0; ++n;          // center
          if (i-1 >= 0) { vals[n] = -invh2; idxs[n] = 1; ++n; }  // i-1
          if (i+1 <  N) { vals[n] = -invh2; idxs[n] = 2; ++n; }  // i+1
          if (j-1 >= 0) { vals[n] = -invh2; idxs[n] = 3; ++n; }  // j-1
          if (j+1 <  N) { vals[n] = -invh2; idxs[n] = 4; ++n; }  // j+1

          int ij[2] = {i, j};
          HYPRE_CHK( HYPRE_StructMatrixSetValues(A, ij, n, idxs, vals) );
        }
      }
    }
    HYPRE_CHK( HYPRE_StructMatrixAssemble(A) );

    // ---- Fill RHS b from your b_local (global row-major ids), set x=0
    HYPRE_CHK( HYPRE_StructVectorSetConstantValues(x, 0.0) );
    for (int g = r0; g < r1; ++g) {
      const int i = g / N, j = g % N;
      int ij[2] = {i, j};
      double v = b_local[(size_t)(g - r0)];
      HYPRE_CHK( HYPRE_StructVectorSetValues(b, ij, &v) );
    }
    HYPRE_CHK( HYPRE_StructVectorAssemble(b) );
    HYPRE_CHK( HYPRE_StructVectorAssemble(x) );

    // ---- BiCGSTAB + PFMG(preconditioner)
    HYPRE_StructSolver bicg = nullptr;
    HYPRE_StructSolver pfmg = nullptr;

    HYPRE_CHK( HYPRE_StructBiCGSTABCreate(comm, &bicg) );
    HYPRE_CHK( HYPRE_StructBiCGSTABSetTol(bicg, rtol) );
    HYPRE_CHK( HYPRE_StructBiCGSTABSetMaxIter(bicg, 1000) );
    HYPRE_CHK( HYPRE_StructBiCGSTABSetLogging(bicg, 1) );     // <-- for residual reporting
    HYPRE_CHK( HYPRE_StructBiCGSTABSetPrintLevel(bicg, 0) );

    HYPRE_CHK( HYPRE_StructPFMGCreate(comm, &pfmg) );
    // preconditioner settings (strict preconditioner: single V-cycle)
    HYPRE_CHK( HYPRE_StructPFMGSetTol(pfmg, 0.0) );
    HYPRE_CHK( HYPRE_StructPFMGSetMaxIter(pfmg, 1) );
    HYPRE_CHK( HYPRE_StructPFMGSetNumPreRelax(pfmg, 1) );
    HYPRE_CHK( HYPRE_StructPFMGSetNumPostRelax(pfmg, 1) );
    HYPRE_CHK( HYPRE_StructPFMGSetRelaxType(pfmg, 2) ); // RB-GS tends to be robust

    // attach PFMG as preconditioner
    HYPRE_CHK( HYPRE_StructBiCGSTABSetPrecond(
      bicg,
      HYPRE_StructPFMGSolve,
      HYPRE_StructPFMGSetup,
      pfmg) );

    double t1 = MPI_Wtime();
    HYPRE_CHK( HYPRE_StructBiCGSTABSetup(bicg, A, b, x) );
    double t2 = MPI_Wtime();
    {
      const int ierr = HYPRE_StructBiCGSTABSolve(bicg, A, b, x);
      // treat non-convergence as non-fatal to record stats (BiCGSTAB returns 256 on no convergence)
      if (ierr && ierr != 256) hypre_chk("HYPRE_StructBiCGSTABSolve", ierr);
    }
    double t3 = MPI_Wtime();

    int its = 0;      HYPRE_CHK( HYPRE_StructBiCGSTABGetNumIterations(bicg, &its) );
    double rel = 0.0; HYPRE_CHK( HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(bicg, &rel) );

    // ---- gather x back in [r0, r1)
    u_local.resize((size_t)(r1 - r0));
    for (int g = r0; g < r1; ++g) {
      int i = g / N, j = g % N;
      int ij[2] = {i, j};
      double val = 0.0;
      HYPRE_CHK( HYPRE_StructVectorGetValues(x, ij, &val) );
      u_local[(size_t)(g - r0)] = val;
    }

    S.iters = its;
    S.rtol_achieved = rel;
    S.t_setup = t2 - t1;
    S.t_solve = t3 - t2;

    // ---- cleanup
    HYPRE_CHK( HYPRE_StructPFMGDestroy(pfmg) );
    HYPRE_CHK( HYPRE_StructBiCGSTABDestroy(bicg) );
    HYPRE_CHK( HYPRE_StructVectorDestroy(x) );
    HYPRE_CHK( HYPRE_StructVectorDestroy(b) );
    HYPRE_CHK( HYPRE_StructMatrixDestroy(A) );
    HYPRE_CHK( HYPRE_StructStencilDestroy(stencil) );
    HYPRE_CHK( HYPRE_StructGridDestroy(grid) );

    return S;
  }
};

ISolver* create_solver() { return new HypreStructBiCGSTABSolver(); }
