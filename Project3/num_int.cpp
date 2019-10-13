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


void GaussLegendre()
{
  int N;


}
