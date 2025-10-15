#pragma once
#include <vector>
#include <mpi.h>

struct CSR {
  std::vector<int> ia;  // size nrows+1
  std::vector<int> ja;  // col indices
  std::vector<double> a;// values
  int nrows = 0, ncols = 0;
};

struct Stats {
  int iters = -1;
  double rtol_achieved = -1.0;
  double t_setup = 0.0, t_solve = 0.0;
  double Linf = 0.0, L2 = 0.0;
};

void build_poisson_2d_csr(int N, int r0, int r1, CSR &csr);
void build_rhs_and_exact(int N, int r0, int r1,
                         std::vector<double> &b_local,
                         std::vector<double> &u_exact_local);
void compute_errors(int N, const std::vector<double> &u_local,
                    const std::vector<double> &u_exact_local,
                    int r0, int r1, MPI_Comm comm, double &Linf, double &L2);
void row_partition(int N, int rank, int size, int &r0, int &r1);

struct ISolver {
  virtual ~ISolver() = default;
  virtual Stats solve(const CSR &csr,
                      const std::vector<double> &b_local,
                      int N, int r0, int r1, double rtol,
                      std::vector<double> &u_local,
                      MPI_Comm comm) = 0;
};

ISolver* create_solver();
