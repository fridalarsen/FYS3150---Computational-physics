#include <iostream>
#include <cmath>

using namespace std;


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


extern "C" void GaussLegendre(double a, double b, double **x, double **w, int N,
                              double tol)
/* Function for finding zeros and weights using Legendre polynomials in the
   Gaussian quadrature.

   Arguments:
      a (double): lower integration limit
      b (double): upper integration limit
      x (double **): empty 1d column array, length N
      w (double **): empty 1d column array, length N
      N (int): number of integration points

   "Returns":
      x (double **): filled with zeros of Legendre polynomial of Nth degree.
      w (double **): filled with integration weights.
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
    x[i][0]   = center - radius*next_guess;
    x[N-1-i][0] = center + radius*next_guess;   // using the fact that the roots are symmetric

    // set w-values
    w[i][0]   = 2*radius / ((1 - next_guess*next_guess) * dleg*dleg);
    w[N-1-i][0] = w[i][0];
  }
}
