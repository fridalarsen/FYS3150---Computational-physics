#include <iostream>
#include <random>
#include <cmath>
#include <string>
#include <fstream>
#include <iomanip>

using namespace std;

class IsingModel {
  private:
    int N;
    int** grid;
    int M;
    int E;
    double J;
    random_device rd;
    mt19937_64 generator;
    uniform_int_distribution<int> random_index;
    uniform_real_distribution<double> probability;
    double beta;
    int accepted;

    int periodic_boundaries(int index){
      /*
      Checks whether an index is within the grid or not. Ensures periodic
      boundaries.
      */
      if(index == -1){
        return (N-1);
      }else if(index == N){
        return 0;
      }else{
        return index;
      }
    }

    void spin_flip(){
      /*
      Function for flipping a random spin in the grid. Accepts the flip if the
      energy change is negative.
      */
      int i = random_index(generator);
      int j = random_index(generator);

      int neighbours = 0;
      neighbours += grid[periodic_boundaries(i)][periodic_boundaries(j-1)];
      neighbours += grid[periodic_boundaries(i)][periodic_boundaries(j+1)];
      neighbours += grid[periodic_boundaries(i-1)][periodic_boundaries(j)];
      neighbours += grid[periodic_boundaries(i+1)][periodic_boundaries(j)];

      int delta_E = 2*J*grid[i][j]*neighbours;

      if(delta_E <= 0){
        grid[i][j] = -grid[i][j];
        M += 2*grid[i][j];
        E += delta_E;
        accepted += 1;
      }else{
        double w = exp(-beta*delta_E);
        double r = probability(generator);
        if(r <= w){
          grid[i][j] = -grid[i][j];
          M += 2*grid[i][j];
          E += delta_E;
          accepted += 1;
        }
      }
    }

    void randomize_grid(){
      /*
      Function for performing spin_flip N times.
      */
      for(int i=0; i<N*N; i++){
        spin_flip();
      }
    }

  public:
    IsingModel(int n) : generator(rd()),
    random_index(uniform_int_distribution<int>(0,n-1)),
    probability(uniform_real_distribution<double>(0,1)){
      /*
      Generator.
      Arguments:
        n (int): Specify number of grid points (nxn).
      */
      N = n;
      grid = new int* [N];

      for(int i=0; i<N; i++){
        grid[i] = new int [N];
        for(int j=0; j<N; j++){
          grid[i][j] = 1;
        }
      }
      M = N*N;
      E = -2*N*N;
      beta = 0.25;
      J = 1;
      accepted = 0;
    }
    ~IsingModel(){
      /*
      Destructor.
      */
      for(int i=0; i<N; i++){
        delete[] grid[i];
      }
      delete[] grid;
    }

    void print_grid(){
      /*
      Function for printing the values of the spins in the grid.
      */
      for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
          if(grid[i][j] == 1){
            cout << " " << grid[i][j] << " ";
          }else{
            cout << grid[i][j] << " ";
          }
        }
        cout << endl;
      }
    }

    void set_parameters(double beta_=0.25, double J_=1){
      /*
      Function for setting parameters of the model.
      Arguments:
        beta_ (double): Optional. Beta-parameter, describes temperature.
        J_ (double): Optional. Coupling constant of strength between neighbouring
                  spins.
      */
      beta = beta_;
      J = J_;
    }

    void initialize_grid(int cycles){
      /*
      Function for finding an initial equilibrium, Monte Carlo burn in.
      Arguments:
        cycles (int): Number of burn in cycles to perform.
      */
      for(int i=0; i<cycles; i++){
        randomize_grid();
      }
    }

    void initialize_random_grid(){
      /*
      Function for creating a random grid using an even distribution.
      */
      for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
          int p = grid[i][j];

          double val = probability(generator);
          if(val < 0.5){
            grid[i][j] = 1;
          }else{
            grid[i][j] = -1;
          }
          if(grid[i][j] != p){
            int neighbours = 0;
            neighbours += grid[periodic_boundaries(i)][periodic_boundaries(j-1)];
            neighbours += grid[periodic_boundaries(i)][periodic_boundaries(j+1)];
            neighbours += grid[periodic_boundaries(i-1)][periodic_boundaries(j)];
            neighbours += grid[periodic_boundaries(i+1)][periodic_boundaries(j)];

            E += 2*J*grid[i][j]*neighbours;
            M += 2*grid[i][j];
          }
        }
      }
    }

    void analyze_initialize_grid(int max_cycles, int* E_, int* M_,
                                 double* accepted_config){
      /*
      Function for estimating mean energy and magnetization to be used for
      finding the optimal amount of burn in required. Note that the values
      returned from this function are not properly scaled -- only their overall
      behaviour is of interest.
      Arguments:
        max_cycles (int): Maximum number of Monte Carlo burn in cycles.
        E_ (int array): Array to store mean energy values.
        M_ (int array): Array to store mean magnetization values.
        accepted_config (double array): Array to store number of accepted
                                        configurations.
      */
      for(int i=0; i<max_cycles; i++){
        accepted = 0;
        randomize_grid();
        E_[i] = E;
        M_[i] = M;
        accepted_config[i] = accepted;
      }
    }

    void MonteCarlo(int MC_cycles, double& E_mean, double& C_V, double& M_mean,
                    double& chi){
      /*
      Function for performing the Monte Carlo simulation for determining the
      mean energy, magnetization, specific heat and susceptibility.
      Arguments:
        MC_cycles (int): Number of Monte Carlo cycles to perform.
        E_mean (double pointer): Mean energy.
        C_V (double pointer): Specific heat.
        M_mean (double pointer): Mean magnetization.
        chi (double pointer): Susceptibility.
      */
      E_mean = 0;
      C_V = 0;
      M_mean = 0;
      chi = 0;
      double M_mean_signed = 0;
      for(int i=0; i<MC_cycles; i++){
        randomize_grid();
        E_mean += E;
        M_mean += fabs(M);
        M_mean_signed += M;

        C_V += E*E;
        chi += M*M;
      }
      E_mean = E_mean / (double)MC_cycles;
      M_mean = M_mean / (double)MC_cycles;
      C_V = C_V / (double)MC_cycles;
      chi = chi / (double)MC_cycles;
      M_mean_signed = M_mean_signed / (double)MC_cycles;

      C_V = beta*beta*(C_V - E_mean*E_mean);
      chi = beta*(chi - M_mean_signed*M_mean_signed);
    }
  };

template<class T>
void write_to_file(string filename, T* data, int len, int precision = 16){
  /*
  Function for writing the data of an array to a file.
  Arguments:
    filename (str): Name of file to hold data.
    data (double array): Array of data to be written to file.
    len (int): Length of data-array.
    precision (int): Optional. Precision of data to be written to file.
  */
  ofstream myfile;
  myfile.open(filename);
  myfile << setprecision(precision);
  for(int i=0; i<len; i++){
    myfile << data[i] << endl;
  }
  myfile.close();
}
