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

  int max_cycles = 10000;
  int* E_ = new int[max_cycles];
  int* M_ = new int[max_cycles];
  double* a_c = new double[max_cycles];
  IsingModel IM(L);
  IM.set_parameters(beta1);

  IsingModel IM2(L);
  IM2.set_parameters(beta2);

  // begin with ordered grid (all spins pointing up)
  IM.analyze_initialize_grid(max_cycles, E_, M_, a_c);
  string f1 = "Results/burn_in_E_4c_1.dat";
  string f2 = "Results/burn_in_M_4c_1.dat";
  string accept1 = "Results/4c_acceptance_1.dat";
  write_to_file(f1, E_, max_cycles);
  write_to_file(f2, M_, max_cycles);
  write_to_file(accept1, a_c, max_cycles);

  IM2.analyze_initialize_grid(max_cycles, E_, M_, a_c);
  string f1_ = "Results/burn_in_E_4c_3.dat";
  string f2_ = "Results/burn_in_M_4c_3.dat";
  string accept2 = "Results/4c_acceptance_2.dat";
  write_to_file(f1_, E_, max_cycles);
  write_to_file(f2_, M_, max_cycles);
  write_to_file(accept2, a_c, max_cycles);

  // begin with random grid (spins randomly pointing up or down)
  IM.initialize_random_grid();
  IM.analyze_initialize_grid(max_cycles, E_, M_, a_c);
  string f3 = "Results/burn_in_E_4c_2.dat";
  string f4 = "Results/burn_in_M_4c_2.dat";
  string accept3 = "Results/4c_acceptance_3.dat";
  write_to_file(f3, E_, max_cycles);
  write_to_file(f4, M_, max_cycles);
  write_to_file(accept3, a_c, max_cycles);

  IM2.initialize_random_grid();
  IM2.analyze_initialize_grid(max_cycles, E_, M_, a_c);
  string f3_ = "Results/burn_in_E_4c_4.dat";
  string f4_ = "Results/burn_in_M_4c_4.dat";
  string accept4 = "Results/4c_acceptance_4.dat";
  write_to_file(f3_, E_, max_cycles);
  write_to_file(f4_, M_, max_cycles);
  write_to_file(accept4, a_c, max_cycles);

  delete[] E_;
  delete[] M_;
  delete[] a_c;
}
