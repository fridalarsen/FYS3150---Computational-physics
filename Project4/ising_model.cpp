#include <iostream>
#include <random>
#include <cmath>

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

    int periodic_boundaries(int index){
      /*
      Checks whether an index is within the grid or not.
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
      }else{
        double w = exp(-beta*delta_E);
        double r = probability(generator);
        if(r <= w){
          grid[i][j] = -grid[i][j];
          M += 2*grid[i][j];
          E += delta_E;
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
      E = 2*N*N;
      beta = 0.25;
      J = 1;
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

    void analyze_initialize_grid(int max_cycles, int* E_, int* M_){
      /*
      Function for estimating mean energy and magnetization to be used for
      finding the optimal amount of burn in required. Note that the values
      returned from this function are not properly scaled -- only their overall
      behaviour is of interest.
      Arguments:
        max_cycles (int): Maximum number of Monte Carlo burn in cycles.
        E_ (int array): Array to store mean energy values.
        M_ (int array): Array to store mean magnetization values.
      */
      for(int i=0; i<max_cycles; i++){
        randomize_grid();
        E_[i] = E;
        M_[i] = M;
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
      for(int i=0; i<MC_cycles; i++){
        randomize_grid();
        E_mean += E;
        M_mean += fabs(M);

        C_V += E*E;
        chi += M*M;
      }
      E_mean = E_mean / (double)MC_cycles;
      M_mean = M_mean / (double)MC_cycles;
      C_V = C_V / (double)MC_cycles;
      chi = chi / (double)MC_cycles;

      C_V = beta*beta*(C_V - E_mean*E_mean); // must be *k_B in main
      chi = beta*(chi - M_mean*M_mean);
    }

    void continue_MonteCarlo(int MC_cycles_p, int MC_cycles_n, double& E_mean,
                             double& C_V, double& M_mean, double& chi){
      /*
      Function for continuing Monte Carlo simulation.
      Arguments:
        MC_cycles_p: Number of Monte Carlo cycles already performed.
        MC_cycles_n: Number of additional Monte Carlo cycles to perform.
        E_mean, C_V, M_mean, chi: See MonteCarlo function.
      */
      double E_temp;
      dobule CV_temp;
      double M_temp;
      double chi_temp;
      MonteCarlo(MC_cycles_n, E_temp, CV_temp, M_temp, chi_temp);

      double a = (double)MC_cycles_p / (double)(MC_cycles_p+MC_cycles_n);
      double b = (double)MC_cycles_n / (double)(MC_cycles_p+MC_cycles_n);
      E_mean = a*E_mean + b*E_temp; // sum of averages, look it up loser
      C_V = a*C_V + b*CV_temp;
      M_mean = a*M_mean + b*M_temp;
      chi = a*chi + b*chi_temp;
    }
};

void write_to_file(str filename, double* data, int len, int precision = 16){
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
