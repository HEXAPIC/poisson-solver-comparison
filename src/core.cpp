#include "solver_iface.hpp"
#include <algorithm>
#include <cmath>

static inline int g_i(int g, int N) { return g / N; }
static inline int g_j(int g, int N) { return g % N; }

void row_partition(int N, int rank, int size, int &r0, int &r1) {
  const int nn = N * N;
  const int base = nn / size, rem = nn % size;
  r0 = rank * base + std::min(rank, rem);
  r1 = r0 + base + (rank < rem ? 1 : 0);
}

void build_poisson_2d_csr(int N, int r0, int r1, CSR &csr) {
  const int local_n = r1 - r0;
  csr.nrows = local_n;
  csr.ncols = N * N;
  csr.ia.resize(local_n + 1);
  csr.ja.reserve(local_n * 5);
  csr.a.reserve(local_n * 5);

  const double h = 1.0 / (N + 1);
  const double invh2 = 1.0 / (h * h);

  csr.ia[0] = 0;
  for (int r = 0; r < local_n; ++r) {
    int g = r0 + r;
    int i = g_i(g, N);
    int j = g_j(g, N);

    // center
    csr.ja.push_back(g);
    csr.a.push_back(4.0 * invh2);
    // left
    if (j - 1 >= 0) { csr.ja.push_back(g - 1); csr.a.push_back(-invh2); }
    // right
    if (j + 1 <  N) { csr.ja.push_back(g + 1); csr.a.push_back(-invh2); }
    // down
    if (i - 1 >= 0) { csr.ja.push_back(g - N); csr.a.push_back(-invh2); }
    // up
    if (i + 1 <  N) { csr.ja.push_back(g + N); csr.a.push_back(-invh2); }

    // sort this small row by column index (stable insertion sort)
    int s = csr.ia[r];
    int e = static_cast<int>(csr.ja.size());
    for (int a = s + 1; a < e; ++a) {
      int cj = csr.ja[a]; double cv = csr.a[a];
      int b = a - 1;
      while (b >= s && csr.ja[b] > cj) {
        csr.ja[b + 1] = csr.ja[b];
        csr.a[b + 1] = csr.a[b];
        --b;
      }
      csr.ja[b + 1] = cj; csr.a[b + 1] = cv;
    }
    csr.ia[r + 1] = static_cast<int>(csr.ja.size());
  }
}

void build_rhs_and_exact(int N, int r0, int r1,
                         std::vector<double> &b_local,
                         std::vector<double> &u_exact_local) {
  const double h = 1.0 / (N + 1);
  const int local_n = r1 - r0;
  b_local.resize(local_n);
  u_exact_local.resize(local_n);

  for (int g = r0; g < r1; ++g) {
    int r = g - r0;
    int i = g_i(g, N), j = g_j(g, N);
    double x = (i + 1) * h;
    double y = (j + 1) * h;
    double u = std::sin(M_PI * x) * std::sin(M_PI * y);
    double f = 2.0 * M_PI * M_PI * u; // -Δu = 2π^2 u
    b_local[r] = f;
    u_exact_local[r] = u;
  }
}

void compute_errors(int N, const std::vector<double> &u_local,
                    const std::vector<double> &u_exact_local,
                    int /*r0*/, int /*r1*/, MPI_Comm comm, double &Linf, double &L2) {
  const double h = 1.0 / (N + 1);
  double loc_inf = 0.0, loc_l2 = 0.0;
  for (size_t k = 0; k < u_local.size(); ++k) {
    double e = std::abs(u_local[k] - u_exact_local[k]);
    loc_inf = std::max(loc_inf, e);
    loc_l2  += e * e;
  }
  MPI_Allreduce(&loc_inf, &Linf, 1, MPI_DOUBLE, MPI_MAX, comm);
  double gsum = 0.0;
  MPI_Allreduce(&loc_l2, &gsum, 1, MPI_DOUBLE, MPI_SUM, comm); // use separate recv buffer
  L2 = std::sqrt(h * h * gsum);
}
