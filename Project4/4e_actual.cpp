#include <iostream>
#include "ising_model.cpp"
#include <string>
#include <sstream>
#include "mpi.h"

using namespace std;

int main(){
  double k_B = 1;
  double beta;

  int burn_in = 10000;
  int MC_cycles = (int)(pow(10,6));
  int L = 80;

  int n_local = 15;
  double T0 = 2.24;
  double T1 = 2.3;
  double dT;

  int numprocs, myid;
  int nameLen;
  char processorName[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Get_processor_name(processorName, &nameLen);

  int n_tot = n_local * numprocs;

  double* global_T = nullptr;
  double* global_E_mean = nullptr;
  double* global_M_mean = nullptr;
  double* global_C_v = nullptr;
  double* global_chi = nullptr;

  double* T = new double [n_local];
  double* E_mean = new double [n_local];
  double* M_mean = new double [n_local];
  double* C_v = new double [n_local];
  double* chi = new double [n_local];

  if (myid == 0) {
    global_T = new double [n_tot];
    global_E_mean = new double [n_tot];
    global_M_mean = new double [n_tot];
    global_C_v = new double [n_tot];
    global_chi = new double [n_tot];

    dT = (T1-T0)/(double)(n_tot-1);
    global_T[0] = T0;
    for (int i=1; i<n_tot-1; i++){
      global_T[i] = global_T[i-1] + dT;
    }
    global_T[n_tot-1] = T1;
  }

  MPI_Scatter(global_T, n_local, MPI_DOUBLE, T, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(global_E_mean, n_local, MPI_DOUBLE, E_mean, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(global_M_mean, n_local, MPI_DOUBLE, M_mean, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(global_C_v, n_local, MPI_DOUBLE, C_v, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(global_chi, n_local, MPI_DOUBLE, chi, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int seed = ((unsigned)time(NULL)+myid*numprocs*nameLen);

  IsingModel IM(L,seed);
  IM.initialize_grid(burn_in);

  for(int i=0; i<n_local; i++){
    beta = 1/(T[i]*k_B);
    IM.set_parameters(beta);
    IM.MonteCarlo(MC_cycles, E_mean[i], C_v[i], M_mean[i], chi[i]);

    cout << "rank " << myid << " done with index " << i << " and T = " << T[i] << endl;
  }

  MPI_Gather(T, n_local, MPI_DOUBLE, global_T, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(E_mean, n_local, MPI_DOUBLE, global_E_mean, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(M_mean, n_local, MPI_DOUBLE, global_M_mean, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(C_v, n_local, MPI_DOUBLE, global_C_v, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(chi, n_local, MPI_DOUBLE, global_chi, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (myid == 0){
    ostringstream f1, f2, f3, f4;
    f1 << "Results/4e_E_L" << L << ".dat";
    f2 << "Results/4e_M_L" << L << ".dat";
    f3 << "Results/4e_Cv_L" << L << ".dat";
    f4 << "Results/4e_chi_L" << L << ".dat";

    write_to_file(f1.str(), global_E_mean, n_tot);
    write_to_file(f2.str(), global_M_mean, n_tot);
    write_to_file(f3.str(), global_C_v, n_tot);
    write_to_file(f4.str(), global_chi, n_tot);

    delete[] global_T;
    delete[] global_E_mean;
    delete[] global_M_mean;
    delete[] global_C_v;
    delete[] global_chi;
  }

  delete[] T;
  delete[] E_mean;
  delete[] M_mean;
  delete[] C_v;
  delete[] chi;

  MPI_Finalize();

  return 0;
}
