#include <iostream>
#include "ising_model.cpp"
#include <string>

int main(int argc, char *argv[]){
  double k_B = 1;
  double T = 1;
  double beta = 1/(T*k_B);
  int L = 2;

  int n = atoi(argv[1]);

  int* MC_cycles = new int[n];
  MC_cycles[0] = 10;
  int sum_ = 10;
  for(int i=1; i<n; i++){
    MC_cycles[i] = (int)pow(10, (i+1));
    sum_ += MC_cycles[i];
  }
  double* E_mean = new double[n];
  double* M_mean = new double[n];
  double* C_v = new double[n];
  double* chi = new double[n];

  IsingModel IM(L);
  IM.set_parameters(beta);
  IM.MonteCarlo(MC_cycles[0], E_mean[0], C_v[0], M_mean[0], chi[0]);
  sum_ = 10;
  for(int i=1; i<n; i++){
    E_mean[i] = E_mean[i-1];
    C_v[i] = C_v[i-1];
    M_mean[i] = M_mean[i-1];
    chi[i] = chi[i-1];
    IM.MonteCarlo(MC_cycles[i], E_mean[i], C_v[i],
                           M_mean[i], chi[i]);
    sum_ += MC_cycles[i];
  }
  string f1 = "Results/E_mean_4b.dat";
  string f2 = "Results/M_mean_4b.dat";
  string f3 = "Results/C_V_4b.dat";
  string f4 = "Results/chi_4b.dat";

  write_to_file(f1, E_mean, n);
  write_to_file(f2, M_mean, n);
  write_to_file(f3, C_v, n);
  write_to_file(f4, chi, n);

  delete[] MC_cycles;
  delete[] E_mean;
  delete[] M_mean;
  delete[] C_v;
  delete[] chi;

}
