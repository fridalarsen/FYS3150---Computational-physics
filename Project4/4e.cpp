#include <iostream>
#include "ising_model.cpp"
#include <string>
#include <sstream>

using namespace std;

int main(){
  double k_B = 1;
  double beta;

  int burn_in = 1000;
  int MC_cycles = 100000;
  int n = 25;

  int L [4] = {40, 60, 80, 100};
  double T0 = 2.0;
  double T1 = 2.3;
  double T;
  double dT = (T1-T0)/(double)(n-1);

  double* E_mean = new double[n];
  double* M_mean = new double[n];
  double* C_v = new double[n];
  double* chi = new double[n];

  for(int l=0; l<4; l++){
    IsingModel IM(L[l]);
    IM.initialize_grid(burn_in);
    T = T0;
    for(int i=0; i<n; i++){
      beta = 1/(T*k_B);
      IM.set_parameters(beta);
      IM.MonteCarlo(MC_cycles, E_mean[i], C_v[i], M_mean[i], chi[i]);
      T += dT;
    }

    ostringstream f1, f2, f3, f4;
    f1 << "Results/4e_E_L" << l << ".dat";
    f2 << "Results/4e_M_L" << l << ".dat";
    f3 << "Results/4e_Cv_L" << l << ".dat";
    f4 << "Results/4e_chi_L" << l << ".dat";

    write_to_file(f1.str(), E_mean, n);
    write_to_file(f2.str(), M_mean, n);
    write_to_file(f3.str(), C_v, n);
    write_to_file(f4.str(), chi, n);

    cout << "Finished L=" << L[l] << endl;
  }

  delete[] E_mean;
  delete[] M_mean;
  delete[] C_v;
  delete[] chi;
}
