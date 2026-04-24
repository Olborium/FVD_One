#include "methods.h"

////////////////////////////////////////////////////////////////////
/*
							Constants
*/
////////////////////////////////////////////////////////////////////

double TEMP;
double SIZE;
double DX;
double DT;
int N_x;

double kappa;

std::vector<std::vector<double>> A = {{0.5L, 0.5L},
	{0.291666666666666666666666666667, 0.75L, -0.041666666666666666666666666667},
	{0.1344961992774310892, -0.2248198030794208058, 0.7563200005156682911, 0.3340036032863214255}
};
std::vector<std::vector<double>> B = {{1.0L, 0.0L},
	{0.66666666666666666666666666666667, -0.66666666666666666666666666666667, 1.0L},
	{0.5153528374311229364, -0.085782019412973646, 0.4415830236164665242, 0.1288461583653841854}
};

////////////////////////////////////////////////////////////////////
/*
					Initial state preparation
*/
////////////////////////////////////////////////////////////////////

void prepare_in_state_RJ( std::vector<double*> vars, std::vector<fftw_plan> fft_i, std::vector<fftw_plan> fft_f, int sign, unsigned int seed)
{
	std::seed_seq seq{seed, 0u};
	std::mt19937 gen(seq);
	double ThMassSquared = 1.0 + 3.0*sign*TEMP/2.0;
// Zero mode
	double O_0 = std::sqrt(ThMassSquared);
	std::normal_distribution<double> distr1(0.0, std::sqrt(TEMP/DX)/O_0);
	std::normal_distribution<double> distr2(0.0, std::sqrt(TEMP/DX));
	vars[0][0] = distr1(gen);
	vars[1][0] = distr2(gen);
// Positive&negative frequencies
	for (int i = 1; i < N_x/2; i++) {
		double k_i = 2.0*M_PI*i/SIZE;
		double O_i = std::sqrt(ThMassSquared+2.0/(DX*DX)*(1.0-std::cos(DX*k_i)));
		std::normal_distribution<double> distr1(0.0, std::sqrt(TEMP/2.0/DX)/O_i);
		std::normal_distribution<double> distr2(0.0, std::sqrt(TEMP/2.0/DX));
		vars[0][i] = distr1(gen);
		vars[0][N_x-i] = distr1(gen);
		vars[1][i] = distr2(gen);
		vars[1][N_x-i] = distr2(gen);
		}	
// Maximum momentum mode
	double k_max = M_PI*N_x/SIZE;
	double O_max = std::sqrt(ThMassSquared+2.0/(DX*DX)*(1.0-std::cos(DX*k_max)));
	std::normal_distribution<double> distr3(0.0, std::sqrt(TEMP/DX)/O_max);
	vars[0][N_x/2] = distr3(gen);
	vars[1][N_x/2] = distr2(gen);

	fftw_execute(fft_i[0]);
	fftw_execute(fft_i[1]);

	for(int i = 0; i < N_x; i++){
		vars[0][i] = vars[0][i]/std::sqrt(N_x);
		vars[1][i] = vars[1][i]/std::sqrt(N_x);
	}
}

// double norm_distr(double y, double s) 
// {
// 	return 2.0/std::sqrt(2.0*M_PI)*std::exp(-y*y/2.0/s/s);
// }

// double der_norm_distr(double y, double s)
// {
// 	return y/s/s*std::exp(-y*y/2.0/s/s);
// }


void prepare_excited_sphaleron( std::vector<double*> vars, std::vector<fftw_plan> fft_i, std::vector<fftw_plan> fft_f, int sign, std::vector<std::vector<double>>& modes, std::vector<double>& O2, unsigned int seed)
{
	if(sign > 0) {std::cerr << "Invalid sign! The in-state is null." << std::endl; } else {
		std::seed_seq seq{seed, 1u};
		std::mt19937 gen(seq);
		std::vector<double> Q(N_x);
		std::vector<double> P(N_x);
// Negative mode
		Q[0] = 0.0;
		
		std::uniform_real_distribution<double> uni(std::nextafter(0.0, 1.0), 1.0);
		double u = uni(gen);
		double y = -std::sqrt(-2.0 * TEMP / DX * std::log(u));
		P[0] = -y;

		// std::normal_distribution<double> distr_norm(0.0, std::sqrt(TEMP/DX));
		// std::uniform_real_distribution<double> distr_unif(0.0, 0.1);
		// bool success = false;
		// double y;
		// while(success == false) {
		// 	y = std::abs(distr_norm(gen));
		// 	double u = distr_unif(gen);
		// 	double M = 5.0;
		// 	if (u < der_norm_distr(y,std::sqrt(TEMP/DX))/M/norm_distr(y,std::sqrt(TEMP/DX)) ) {
		// 		success = true;
		// 	}
		// }
//		std::normal_distribution<double> distr(0.0, std::sqrt(TEMP/DX));

//		P[0] = -std::abs(distr(gen));	
//		P[0] = y;

// Positive modes
		for(int i = 1; i < N_x-1; i++) {
			std::normal_distribution<double> distr1(0.0, std::sqrt(TEMP/DX/O2[i]));
			std::normal_distribution<double> distr2(0.0, std::sqrt(TEMP/DX));
			Q[i] = distr1(gen);
			P[i] = distr2(gen);
		}
// Zero mode
//		double Es = 24.7738; // sphaleron energy (Liouville)
		double Es = 4.0/3.0; // sphaleron energy (phi4)
		std::normal_distribution<double> distr0(0.0, std::sqrt(TEMP/Es));
		Q[N_x-1] = 0.0;
		P[N_x-1] = distr0(gen);		
// Initialise state
		for(int i = 0; i < N_x; i++) {
			double sumQ = 0.0;
			double sumP = 0.0;
			for(int j = 0; j < N_x; j++) {
				sumQ += modes[i][j]*Q[j];
				sumP += modes[i][j]*P[j];
			}
			vars[0][i] = modes[i][N_x] + sumQ;
			vars[1][i] = sumP;
		}		
	}
}

////////////////////////////////////////////////////////////////////
/*
					Evolution
*/
////////////////////////////////////////////////////////////////////

struct S {
	double d1;
	double d2;
};

struct Q {
	double d1;
	double d2;
	double d3;
};

void evol( 	std::vector<double*> vars, std::vector<fftw_plan> fft_f, std::vector<fftw_plan> fft_i,
				double time, std::vector<double>& decay_times, std::vector<double>& decay_to_fvd_counter,
				double eta, int split_order, int stoch_order, int sign, unsigned int seed,
				int snapshot_stride, std::vector<double>* snapshots )
{
	std::seed_seq seq{seed, 2u};
	std::mt19937 gen(seq);
	std::normal_distribution<double> distr(0.0, 1.0);
	std::vector<double> vars0(N_x);
	std::vector<Q> triples(N_x);
	double sigma = std::sqrt(2.0*eta*TEMP*N_x);
	int w = split_order - 2;
	bool overdamped;
	int num_of_steps = std::floor(time/DT);
	double decay_time = 0.0;
	bool decay = false;	
	bool in_fvd = false;
	bool in_fvd_prev = false;
	int crossing_counter = 0;	
	Q q;
	S s;
	int ord_st[3] = {1, stoch_order > 1 ? 1 : 0, stoch_order > 2 ? 1 : 0};

	for (int i = 0; i < num_of_steps; i++) {
		if(decay==true) {break;}
	// Generate noise
		for (int j = 0; j < N_x; j++) {
			double n1 = distr(gen);
			double n2 = distr(gen);
			double n3 = distr(gen);
			double y = 1.0;
			if(j==0||j==N_x/2) {y=2.0;}
			q.d1 = ord_st[0]*std::sqrt(y/(2.0*DX*DT))*n1;
			q.d2 = ord_st[1]*std::sqrt(y*DT/6.0/DX)/2.0*n2;
			q.d3 = ord_st[2]*DT*std::sqrt(y*DT/10.0/DX)/12.0*n3;
			triples[j] = q;
			vars0[j] = vars[0][j];
		}
		for (int n = 0; n < split_order; n++) {
	// Kick
			for (int j = 0; j < N_x; j++) {
				vars[1][j] = vars[1][j] - A[w][n]*sign*DT*vars[0][j]*vars[0][j]*vars[0][j];
//				vars[1][j] = vars[1][j] - A[w][n]*sign*DT*kappa*std::exp(vars[0][j]);
			}	
	// Drift
			fftw_execute(fft_f[0]);
			fftw_execute(fft_f[1]);
			for (int j = 0; j < N_x; j++) {
				double kj;
				if (j<N_x/2+1) {
					kj = 2.0*M_PI*j/SIZE;
				} else {
					kj = 2.0*M_PI*(N_x-j)/SIZE;
				}
				double Oj = std::sqrt(1.0+2.0/(DX*DX)*(1.0-std::cos(DX*kj)));
				if (Oj > eta/2.0) {
					overdamped = false;
				} else {
					overdamped = true;
				}
				double Oej = std::sqrt(std::abs(Oj*Oj-eta*eta/4.0));
				
				s.d1 = triples[j].d2 - eta*triples[j].d3;
				s.d2 = triples[j].d1 - eta*triples[j].d2 - (Oj*Oj-eta*eta+3.0*sign*vars0[j]*vars0[j])*triples[j].d3;

				double a0 = vars[0][j] - sigma*(eta*s.d1+s.d2)/Oj/Oj;
				double b0 = (vars[1][j]+eta*vars[0][j]/2)/Oej + sigma*(s.d1-eta*(eta*s.d1+s.d2)/2.0/Oj/Oj)/Oej;
				double c0 = sigma*(eta*s.d1+s.d2)/Oj/Oj;
				double a1 = vars[1][j] + sigma*s.d1;
				double b1 = -(Oj*Oj*vars[0][j]+eta*vars[1][j]/2.0)/Oej + sigma*(eta*s.d1+2.0*s.d2)/2.0/Oej;
				double c1 = -sigma*s.d1;
				if (overdamped == false) {
					vars[0][j] = (a0*std::cos(B[w][n]*DT*Oej) + b0*std::sin(B[w][n]*DT*Oej))*std::exp(-eta*B[w][n]*DT/2.0) + c0;
					vars[1][j] = (a1*std::cos(B[w][n]*DT*Oej) + b1*std::sin(B[w][n]*DT*Oej))*std::exp(-eta*B[w][n]*DT/2.0) + c1;
				} else {
					vars[0][j] = (a0*std::cosh(B[w][n]*DT*Oej) + b0*std::sinh(B[w][n]*DT*Oej))*std::exp(-eta*B[w][n]*DT/2.0) + c0;
					vars[1][j] = (a1*std::cosh(B[w][n]*DT*Oej) + b1*std::sinh(B[w][n]*DT*Oej))*std::exp(-eta*B[w][n]*DT/2.0) + c1;				
				}
			}
			fftw_execute(fft_i[0]);
			fftw_execute(fft_i[1]);
			for (int j = 0; j < N_x; j++) {
				vars[0][j] /= N_x;
				vars[1][j] /= N_x;
			}							
		}
	// Check for decay
		if(sign<0) {
			if(i%1==0) {
				for(int j = 0; j < N_x; j++) {
					if(std::abs(vars[0][j])>5.0) { // 10.0 for Liouville
//						std::cout << "Decay!!" << std::endl;
						decay = true;
						decay_time = i*DT+0.001;
						break;
					}
				}	
			}
		}
	// Check for return to the false vacuum
		if(i%40==0) {
			double max_val = 0.0;
			for(int j = 0; j < N_x; j++) {
				if (std::abs(vars[0][j]) > max_val) {max_val = std::abs(vars[0][j]);}
			}
			if(max_val < 1.0) {in_fvd = true;} else {in_fvd = false;} // 4.0 for Liouville
			if(in_fvd != in_fvd_prev) {crossing_counter++;}	
			in_fvd_prev = in_fvd;
		}
		if (snapshots != nullptr && snapshot_stride > 0 && (i + 1) % snapshot_stride == 0) {
			snapshots->insert(snapshots->end(), vars[0], vars[0] + N_x);
		}
	}
	decay_times.push_back(decay_time);
	decay_to_fvd_counter.push_back(crossing_counter);
}

////////////////////////////////////////////////////////////////////
/*
					Properties of the state
*/
////////////////////////////////////////////////////////////////////

void take_spectrum(std::vector<double>& phi_k2, std::vector<double*> vars, double* var_k, std::vector<fftw_plan> fft_xk, int n)
{
	fftw_execute(fft_xk[n]);

	phi_k2[0] = var_k[0]*var_k[0]/N_x;
	for(int i = 1; i < N_x/2; ++i) {
 		phi_k2[i] = (var_k[i]*var_k[i] + var_k[N_x-i]*var_k[N_x-i])/N_x;
 	}
	phi_k2[N_x/2] = var_k[N_x/2]*var_k[N_x/2]/N_x;
}

double energy_kinetic(std::vector<double*> vars)
{
	double energy = 0;	

	for (int j = 0; j < N_x; j++) {
		energy += DX*0.5*vars[1][j]*vars[1][j];
	}
	return energy;
}

double energy_potential(std::vector<double*> vars, int sign)
{
	double phi_x;
	double energy = 0;

	for (int j = 0; j < N_x; j++) {
		phi_x = (vars[0][(j+1)&(N_x-1)]-vars[0][j])/DX;
		energy += DX*(0.5*phi_x*phi_x + 0.5*vars[0][j]*vars[0][j] 
				+ 0.25*sign*vars[0][j]*vars[0][j]*vars[0][j]*vars[0][j] );
	}
	return energy;		
}

////////////////////////////////////////////////////////////////////
/*
						Methods of classes
*/
////////////////////////////////////////////////////////////////////

BasicObjects::BasicObjects() {

	var_k = (double*) fftw_malloc(sizeof(double) * N_x);
	for(int i = 0; i < 2; i++) {

		vars.push_back( (double*) fftw_malloc(sizeof(double) * N_x) );
		fft_f.push_back( fftw_plan_r2r_1d(N_x, vars[i], vars[i], FFTW_R2HC, FFTW_ESTIMATE) );
		fft_i.push_back( fftw_plan_r2r_1d(N_x, vars[i], vars[i], FFTW_HC2R, FFTW_ESTIMATE) );
		fft_xk.push_back( fftw_plan_r2r_1d(N_x, vars[i], var_k, FFTW_R2HC, FFTW_ESTIMATE) );
	}
};

BasicObjects::~BasicObjects() {

	for(int i = 0; i < 2; i++) {

		fftw_destroy_plan(fft_f[i]);
		fftw_destroy_plan(fft_i[i]);
		fftw_destroy_plan(fft_xk[i]);
		fftw_free(vars[i]);
	}
	fftw_free(var_k);	
};

InitialState::InitialState(int sign) : sign(sign), rng_seed(0), BasicObjects() {};

void InitialState::set_seed(unsigned int seed) {
	rng_seed = seed;
}

void InitialState::prepare_state() {
	return prepare_in_state_RJ( BasicObjects::vars, BasicObjects::fft_i, BasicObjects::fft_f, InitialState::sign, InitialState::rng_seed );
};	 

void InitialState::prepare_barrier_state(std::vector<std::vector<double>>& modes, std::vector<double>& O2) {
	return prepare_excited_sphaleron( BasicObjects::vars, BasicObjects::fft_i, BasicObjects::fft_f, InitialState::sign, modes, O2, InitialState::rng_seed);
};

Evolve::Evolve( double eta, int split_order, int stoch_order, int sign)
	try : eta(eta), split_order(split_order), stoch_order(stoch_order), InitialState(sign) {
		if (split_order != 2 && split_order != 3 && split_order != 4) {
			throw std::invalid_argument("Invalid splitting order!");
		}
		if (stoch_order != 1 && stoch_order != 2 && stoch_order != 3) {
			throw std::invalid_argument("Invalid stochastic order!");
		}
	}
	catch (const std::invalid_argument& e) {
		std::cerr << e.what() << std::endl;
	}

void Evolve::evolve(std::vector<double>& decay_times, std::vector<double>& osc_counter, double timespan, int snapshot_stride, std::vector<double>* snapshots) {
	return evol( BasicObjects::vars, BasicObjects::fft_f, BasicObjects::fft_i, 
				timespan, decay_times, osc_counter, Evolve::eta, Evolve::split_order, Evolve::stoch_order,
				InitialState::sign, InitialState::rng_seed, snapshot_stride, snapshots );
}

double Evolve::e_kin() {
	return energy_kinetic(BasicObjects::vars);
}
double Evolve::e_pot() {
	return energy_potential(BasicObjects::vars, InitialState::sign);
};
void Evolve::ps(std::vector<double>& phi_k2, int n) {
	return take_spectrum(phi_k2, BasicObjects::vars, BasicObjects::var_k, BasicObjects::fft_xk, n);
}

void load_data(std::vector<std::vector<double>>& data, const std::string& name) {
    std::ifstream inputFile(name);
    if (!inputFile) {
        std::cerr << "Unable to open the mode file " << std::endl;
    } else {
		std::string line;
		
		while (std::getline(inputFile, line)) {
			std::istringstream iss(line);
			std::vector<double> row;

			double value;
			while (iss >> value) {
				row.push_back(value);
			}
			data.push_back(row);	
    	}
    inputFile.close();
	}
}

void load_data(std::vector<double>& data, const std::string& name) {
    std::ifstream inputFile(name);
    if (!inputFile) {
        std::cerr << "Unable to open the eigenvalue file " << std::endl;
    } else {
	
		double value;
		while (inputFile >> value) {
			data.push_back(value);
		}
    inputFile.close();
	}
}
