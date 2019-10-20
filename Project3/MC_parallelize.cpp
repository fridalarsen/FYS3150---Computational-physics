#include <iostream>
#include <cmath>
#include "project3_.cpp"
#include "mpi.h"
#include <iomanip>
#include <time.h>

using namespace std;

extern "C" void Solve3d(int N, double & I, double & var, int seed);

int main(int argc, char *argv[]){
  int N = atoi(argv[1]);
  double I;
  double var;
  int numprocs, myid;
  double final_I, final_var;
  int nameLen;
  char processorName[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Get_processor_name(processorName, &nameLen);

  int seed = ((unsigned)time(NULL)+myid*numprocs*nameLen);
  Solve3d(N, I, var, seed);

  MPI_Reduce(&I, &final_I, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&var, &final_var, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Finalize();

  final_I /= (double) numprocs;
  final_var /= (double) numprocs;

  if (myid == 0){
    ofstream myfile;
    myfile.open ("results_3e.txt");
    myfile << setprecision(16);
    myfile << final_I << endl;
    myfile << final_var << endl;

    myfile.close();
  }

  return 0;

}
