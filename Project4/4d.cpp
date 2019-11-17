#include <iostream>
#include "ising_model.cpp"
#include <string>

using namespace std;

int main(){
  double k_B = 1;
  double T1 = 1.0;
  double T2 = 2.4;
  double beta1 = 1/(T1*k_B);
  double beta2 = 1/(T2*k_B);
  int L = 20;

  int burn_in = 1000;
  int max_cycles = 5500;

  IsingModel IM1(L);
  IM1.set_parameters(beta1);
  IM1.initialize_grid(burn_in);

  IsingModel IM2(L);
  IM2.set_parameters(beta2);
  IM2.initialize_grid(burn_in);

  int* E_ = new int[max_cycles];
  int* M_ = new int[max_cycles];
  double* a_c = new double[max_cycles];

  IM1.analyze_initialize_grid(max_cycles, E_, M_, a_c);
  string f1 = "Results/4d_T1_E.dat";
  write_to_file(f1, E_, max_cycles);

  IM2.analyze_initialize_grid(max_cycles, E_, M_, a_c);
  string f2 = "Results/4d_T2_E.dat";
  write_to_file(f2, E_, max_cycles);

  delete[] E_;
  delete[] M_;
  delete[] a_c;
}
