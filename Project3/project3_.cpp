#include <iostream>
#include <cmath>
#include "gauss-laguerre.cpp"

using namespace std;

void gauss_laguerre(double *x, double *w, int n, double alf);

double Legendre(int n, double x)
/* Function for computing Legendre polynomial of degree n.

   Arguments:
      n (int): order of polynomial
      x (double): free variable.
*/
{
  int j;
  double L;
  if (n == 0){
    L = 1;
  }else if (x==1){
    L = 1;
  }else{
    for (j=0; j < n; j++){
      L = ((2*j+1)*x*Legendre(j,x) - j*Legendre(j-1, x)) / (j+1);
    }
  }
  return L;
}

void GaussLegendre(double a, double b, double *x, double *w, int N,
                              double tol)
/* Function for finding zeros and weights using Legendre polynomials in the
   Gaussian quadrature.

   Arguments:
      a (double): lower integration limit.
      b (double): upper integration limit.
      x (double *): empty 1d column array, length N.
      w (double *): empty 1d column array, length N.
      N (int): number of integration points.

   "Returns":
      x (double *): filled with zeros of Legendre polynomial of Nth degree.
      w (double *): filled with integration weights.
*/
{
  double pi = 3.14159265359;
  int m = (N + 1)/2;
  double guess;
  double next_guess;
  double leg, dleg;

  // transform integration limits
  double center = (b+a) / 2;
  double radius = (b-a) / 2;

  // for loop for finding zeros of L_N
  for(int i=0; i <= m; i++){
    next_guess = cos(pi * ((i+1)-0.25)/(N+0.5));  // initial guess of zero x-value
    guess = next_guess + 2*tol;               // dummy value to enter while-loop

    // Newton's method
    while(fabs(guess - next_guess) > tol){
      guess = next_guess;

      leg = Legendre(N, guess);          // current Legendre polynomial value
      dleg = N*(guess * leg - Legendre(N-1, guess)) / (guess*guess - 1); // derivative

      next_guess = guess - (leg/dleg);
    }

    // set x-values
    x[i]   = center - radius*next_guess;
    x[N-1-i] = center + radius*next_guess;   // using the fact that the roots are symmetric

    // set w-values
    w[i]   = 2*radius / ((1 - next_guess*next_guess) * dleg*dleg);
    w[N-1-i] = w[i];
  }
}

double f(double x1, double y1, double z1, double x2, double y2, double z2,
         double alpha=2, double tol = 1e-10)
{
    double r1 = sqrt(x1*x1 + y1*y1 + z1*z1);
    double r2 = sqrt(x2*x2 + y2*y2 + z2*z2);

    double num = exp(-2*alpha*(r1+r2));
    double denom = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));

    if(denom < tol){
      return 0;
    }else{
      return num/denom;
    }
}

double g(double r1, double r2, double theta1, double theta2, double phi1,
         double phi2, double tol)
{
  double sin1 = sin(theta1);
  double sin2 = sin(theta2);
  double cos1 = cos(theta1);
  double cos2 = cos(theta2);

  double cos_diff = cos(phi1 - phi2);
  double cos_beta = cos1*cos2 + sin1*sin2*cos_diff;

  double r12r12 = r1*r1 + r2*r2 - 2*r1*r2*cos_beta;

  if (r12r12 < tol){
    return 0;
  }else{
    return sin1*sin2*exp(-3*(r1+r2))/sqrt(r12r12);
  }
}

double EstimateCartesianInt(double *x, double *w, int N)
/* Function for computing a six-dimensional integral numerically using Gauss-
   Legendre quadrature.

   Arguments:
      x (double *): empty 1d column array, length N.
      w (double *): empty 1d column array, length N.
      N (int): number of integration points.

   Returns:
      I (float): estimated value of integral.
*/
{
    double I = 0.;
    double x1, y1, z1, x2, y2, z2;
    double w1, w2, w3, w4, w5, w6;

    for (int i=0; i<N; i++){
      x1 = x[i];
      w1 = w[i];
    for (int j=0; j<N; j++){
      y1 = x[j];
      w2 = w1*w[j];
    for (int k=0; k<N; k++){
      z1 = x[k];
      w3 = w2*w[k];
    for (int l=0; l<N; l++){
      x2 = x[l];
      w4 = w3*w[l];
    for (int m=0; m<N; m++){
      y2 = x[m];
      w5 = w4*w[m];
    for (int n=0; n<N; n++){
      z2 = x[n];
      w6 = w5*w[n];
      I += w6*f(x1,y1,z1,x2,y2,z2);
    }}}}}}
    return I;
}

double EstimateSphericalInt(double *r, double *w_r, double *theta,
                            double *w_theta, double *phi, double *w_phi, int N,
                            double tol)
{
    double I = 0.;
    double r1, r2, theta1, theta2, phi1, phi2;
    double w1, w2, w3, w4, w5, w6;

    for (int i=1; i<N+1; i++){
      r1 = r[i];
      w1 = w_r[i];
    for (int j=1; j<N+1; j++){
      r2 = r[j];
      w2 = w1*w_r[j];
    for (int k=0; k<N; k++){
      theta1 = theta[k];
      w3 = w2*w_theta[k];
    for (int l=0; l<N; l++){
      theta2 = theta[l];
      w4 = w3*w_theta[l];
    for (int m=0; m<N; m++){
      phi1 = phi[m];
      w5 = w4*w_phi[m];
    for (int n=0; n<N; n++){
      phi2 = phi[n];
      w6 = w5*w_phi[n];
      I += w6*g(r1, r2, theta1, theta2, phi1, phi2, tol);
    }}}}}}
    return I;
}

extern "C" double Solve3a(double lambda, int N, double tol)
{
  double *x = new double [N];
  double *w = new double [N];

  GaussLegendre(-lambda, lambda, x, w, N, tol);

  double I = EstimateCartesianInt(x, w, N);

  delete[] x;
  delete[] w;

  return I;
}

extern "C" double Solve3b(int N, double tol)
{
  double pi = 3.14159265359;
  double alf = 2.0;

  double *theta = new double [N];
  double *phi = new double [N];
  double *w_theta = new double [N];
  double *w_phi = new double [N];

  GaussLegendre(0, pi, theta, w_theta, N, tol);
  GaussLegendre(0, 2*pi, phi, w_phi, N, tol);

  double *r = new double [N+1];
  double *w_r = new double [N+1];

  gauss_laguerre(r, w_r, N, alf);

  double I = EstimateSphericalInt(r, w_r, theta, w_theta, phi, w_phi, N, tol);

  delete[] theta;
  delete[] phi;
  delete[] w_theta;
  delete[] w_phi;
  delete[] r;
  delete[] w_r;

  return I;
}
