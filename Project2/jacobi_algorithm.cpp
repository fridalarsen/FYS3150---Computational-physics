#include <cmath>
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
  // define matrix
  int N = 5;
  vec d (N, fill::ones);
  vec a (N-1, fill::ones);
  mat T = -2.0*diagmat(d) + diagmat(a, -1) + diagmat(a, 1);

  // use armadillo to find eigenvalues and eigenvectors
  vec eigval;
  mat eigvec;

  eig_sym(eigval, eigvec, T);
  eigval.print();
  eigvec.print();

  // set up algorithm
  double maxit = 10000;
  double tol = 1.0e-9;
  int it = 0;

  // define matrix for holding eigenvectors
  mat A = -2.0*diagmat(d) + diagmat(a, -1) + diagmat(a, 1);
  vec diag (N, fill::ones);
  mat eigvec_ = diagmat(diag);

  double max_offdiag = 0.0;
  // find maximum off-diagonal element of initial matrix
  for (int i = 0; i < N; ++i)
  {
    for (int j = i+1; j < N; ++j)
    {
      double elem_ij = fabs(A(i,j));
      if (elem_ij > max_offdiag)
      {
        max_offdiag = elem_ij;
      }
    }
  }

  // perform algorithm
  double elem_ij;
  double c;
  double s;
  double t;
  double tau;
  double a_kk;
  double a_ll;
  double a_ik;
  double a_il;
  double ev_ik;
  double ev_il;
  int k;
  int l;

  while (max_offdiag > tol && it <= maxit)
  {
    max_offdiag = 0.0;
    k = 0;
    l = 0;

    // find maximum off-diagonal element
    for (int i = 0; i < N; ++i)
    {
      for (int j = i+1; j < N; ++j)
      {
        elem_ij = fabs(A(i,j));
        if (elem_ij > max_offdiag)
        {
          max_offdiag = elem_ij;
          k = i;
          l = j;
        }
      }
    }

    // Do the Jacobi, as they say on the fagspraak
    if (! (k == 0 && l == 0))
    {
      if (A(k,l) == 0.0)
      {
        c = 1.0;
        s = 0.0;
      } else {
        tau = (A(l,l)-A(k,k)) / (2*A(k,l));
        if (tau >= 0.0)
        {
          t = 1 / (tau + sqrt(1 + tau*tau));
        } else {
          t = -1 / (-tau + sqrt(1 + tau*tau));
        }
        c = 1 / sqrt(1 + t*t);
        s = c * t;
      }
      a_kk = A(k,k);
      a_ll = A(l,l);

      A(k,k) = a_kk*(c*c) - 2*A(k,l)*c*s + a_ll*(s*s);
      A(l,l) = a_ll*(c*c) + 2*A(k,l)*c*s + a_kk*(s*s);
      A(k,l) = 0.0;
      A(l,k) = 0.0;

      for (int ii = 0; ii < N; ++ii)
      {
        if (ii != k && ii != l)
        {
          a_ik = A(ii,k);
          a_il = A(ii,l);
          A(ii,k) = a_ik*c - a_il*s;
          A(k,ii) = A(ii,k);
          A(ii,l) = a_il*c + a_ik*s;
          A(l,ii) = A(ii,l);
        }
        // update eigenvectors
        ev_ik = eigvec_(ii,k);
        ev_il = eigvec_(ii,l);

        eigvec_(ii,k) = c*ev_ik - s*ev_il;
        eigvec_(ii,l) = c*ev_il + s*ev_ik;
      }
    }
    it++;
  }
  vec eigenvalues = A.diag();
  eigenvalues.print();
  eigvec_.print();
}
