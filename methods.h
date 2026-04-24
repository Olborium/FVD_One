#ifndef METHODS_H
#define METHODS_H
#define _USE_MATH_DEFINES
#include <functional>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <map>
#include <stdexcept>
#include <fftw3.h>

extern double TEMP;             // Temperature in units of lambda*T/m^3	
extern double SIZE;             // Size of the box					
extern double DX;               // Lattice spacing				
extern int N_x;                 // Number of lattice points; x_{N_x} \equiv x_0; N_x must be power of 2.				
extern double DT;               // Time step				

extern double kappa;

void load_data(std::vector<double>& data, const std::string& name);
void load_data(std::vector<std::vector<double>>& data, const std::string& name);

class BasicObjects {

    public:

        double* var_k;
        std::vector<double*> vars;                  // pointers to the variables (phi, pi)
        std::vector<fftw_plan> fft_f;               
        std::vector<fftw_plan> fft_i;
        std::vector<fftw_plan> fft_xk;

        BasicObjects();
        ~BasicObjects();
};

class InitialState : public BasicObjects {

    public:

        int sign;
        unsigned int rng_seed;
        void prepare_state();
        void prepare_barrier_state(std::vector<std::vector<double>>&, std::vector<double>&);
        void set_seed(unsigned int);

        InitialState(int sign);
};

class Evolve : public InitialState {

    public:
        
        double timespan;
        double eta;
        int split_order;
        int stoch_order;

        Evolve(double eta, int split_order, int stoch_order, int sign); 

        void evolve(std::vector<double>&, std::vector<double>&, double, int snapshot_stride = 0, std::vector<double>* snapshots = nullptr);

        double e_kin();                          // kinetic energy
        double e_pot();                          // potential energy
        void ps(std::vector<double>&, int);           // power spectrum
};

template <typename... Args>
void save_data(int prec, std::vector<double>& data, const std::string& name, Args&&... args)
{
	std::string long_name = name;
	((long_name += "_" + std::to_string(args)), ...);
	long_name += ".txt";	
	std::ofstream out(long_name, std::ios::app);
	if (!out.is_open()) {
		std::cerr << "Cannot open file!" << std::endl;
	} else {
		out << std::fixed << std::setprecision(prec);
        for(double& element : data){
			out << element << " ";
		}
		out << std::endl;
		out.close();
	}
};

#endif
