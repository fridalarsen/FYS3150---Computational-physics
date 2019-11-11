#include <iostream>
#include <random>
#include <cmath>

using namespace std;

class IsingModel {
  private:
    int N;
    int** grid;
    int M;
    double J;
    int E;
    random_device rd;
    mt19937_64 generator;
    uniform_int_distribution<int> random_index;
    uniform_real_distribution<double> probability;
    double beta;

    int periodic_boundaries(int index){
      if(index == -1){
        return (N-1);
      }else if(index == N){
        return 0;
      }else{
        return index;
      }
    }

    void spin_flip(){
      int i = random_index(generator);
      int j = random_index(generator);

      int neighbours = 0;
      neighbours += grid[periodic_boundaries(i)][periodic_boundaries(j-1)];
      neighbours += grid[periodic_boundaries(i)][periodic_boundaries(j+1)];
      neighbours += grid[periodic_boundaries(i-1)][periodic_boundaries(j)];
      neighbours += grid[periodic_boundaries(i+1)][periodic_boundaries(j)];

      int delta_E = 2*grid[i][j]*neighbours;

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
      for(int i=0; i<N*N; i++){
        spin_flip();
      }
    }

  public:
    IsingModel(int n) : generator(rd()),
    random_index(uniform_int_distribution<int>(0,n-1)),
    probability(uniform_real_distribution<double>(0,1)){
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

    }
    ~IsingModel(){
      for(int i=0; i<N; i++){
        delete[] grid[i];
      }
      delete[] grid;
    }

    void print_grid(){
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

    void initialize_grid(int cycles){
      // Monte Carlo burn in
      for(int i=0; i<cycles; i++){
        randomize_grid();
      }
    }

    void analyze_initialize_grid(int max_cycles, int* E_, int* M_){
      for(int i=0; i<max_cycles; i++){
        randomize_grid();
        E_[i] = E;
        M_[i] = M;
      }
    }

    void MonteCarlo(int MC_cycles, double& E_mean, double& C_V, double& M_mean,
                    double& chi){
      E_mean = 0;
      C_V = 0;
      M_mean = 0;
      chi = 0;
      for(int i=0; i<MC_cycles; i++){
        randomize_grid();
        E_mean += E;
        M_mean += M;

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
};

int main(){
  IsingModel IM(10);
  IM.print_grid();
  IM.initialize_grid(50);
  cout << "-----" << endl;
  IM.print_grid();
}
