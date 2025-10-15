#include "solver_iface.hpp"
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

static void parse_args(int argc, char** argv, int &N, double &rtol) {
  N = 128; rtol = 1e-8;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    if (a == "--N" && i + 1 < argc)      N = std::atoi(argv[++i]);
    else if (a == "--rtol" && i + 1 < argc) rtol = std::atof(argv[++i]);
  }
}

extern ISolver* create_solver(); // factory symbol from backend

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, size; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);

  int N; double rtol; parse_args(argc, argv, N, rtol);
  int r0, r1; row_partition(N, rank, size, r0, r1);

  CSR A; build_poisson_2d_csr(N, r0, r1, A);
  std::vector<double> b_local, u_exact_local;
  build_rhs_and_exact(N, r0, r1, b_local, u_exact_local);

  std::vector<double> u_local;
  Stats S;

  double t0 = MPI_Wtime();
  ISolver* solver = create_solver();
  S = solver->solve(A, b_local, N, r0, r1, rtol, u_local, comm);
  delete solver;
  double t1 = MPI_Wtime();

  compute_errors(N, u_local, u_exact_local, r0, r1, comm, S.Linf, S.L2);

  if (rank == 0) {
#if defined(BACKEND_PETSC)
    printf("Backend: PETSc (CG + GAMG)\n");
#elif defined(BACKEND_HYPRE)
    printf("Backend: HYPRE (PCG + BoomerAMG)\n");
#else
    printf("Backend: (unknown)\n");
#endif
    printf("=== Poisson 2D N=%d (n=%d), ranks=%d ===\n", N, N*N, size);
    printf("rtol target: %.3e\n", rtol);
    printf("Setup time:  %.6f s\n", S.t_setup);
    printf("Solve time:  %.6f s\n", S.t_solve);
    printf("Total time:  %.6f s\n", t1 - t0);
    printf("Iters:       %d\n", S.iters);
    printf("Rel. res:    %.3e\n", S.rtol_achieved);
    printf("Errors:      L2 = %.6e, Linf = %.6e\n", S.L2, S.Linf);
  }

  MPI_Finalize();
  return 0;
}
