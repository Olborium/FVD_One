#include "methods_multi.h"

// Stochastic integrator test:
// Computes evolution of the power spectrum of phi and pi

int main(int argc, char* argv[])
{	
	SIZE = std::stod(argv[1]);		
	N_x = std::stoi(argv[2]);
	DT = std::stod(argv[3]);	
	int N_ENS = std::stoi(argv[4]);	
	int N_SAMPLE = std::stoi(argv[5]);
	double TIME_SPAN = std::stod(argv[6]);
	std::string DISTR(argv[7]);	
	std::string INTEG(argv[8]);
	TEMP = std::stod(argv[9]);	
	double eta = std::stod(argv[10]);
	double lambda = std::stod(argv[11]);	
	double m[1] = {1.0};
	double l[1] = {lambda};
	DX = SIZE/N_x;
///////////////////////////////////////////////////// 	
	std::vector<double> decay_times;
	std::vector<bool> if_decayed(N_ENS, false);	
	std::vector<std::vector<std::vector<double> > > power_spectrum_phi(N_ENS, std::vector<std::vector<double> >(N_SAMPLE, std::vector<double>(N_x)));
	std::vector<std::vector<std::vector<double> > > power_spectrum_chi(N_ENS, std::vector<std::vector<double> >(N_SAMPLE, std::vector<double>(N_x)));
	std::vector<std::vector<double> > power_spectrum_av_phi(N_SAMPLE, std::vector<double>(N_x));
	std::vector<std::vector<double> > power_spectrum_av_chi(N_SAMPLE, std::vector<double>(N_x));	
	std::vector<double> phi_k2 (N_x);
	std::vector<double> chi_k2 (N_x);	
/////////////////////////////////////////////////////

	Evolve sol(eta, m, l, DISTR, INTEG);
	
	int num_of_decays = 0;
	for(int i = 0; i < N_ENS; i++) {
//		if(i%1 == 0) {
//			std::cout << "Calculating " << i << "\'th element of the ensemble..." << std::endl; }		
		sol.prepare_state();
		for (int n = 0; n < N_SAMPLE; n++) {	
//			if (n%10 == 0) { std::cout << "Taking " << n << "\'th sample..." << std::endl; }
			sol.evolve(decay_times, TIME_SPAN);
			sol.ps(phi_k2, 0);
			sol.ps(chi_k2, 1);
			for (int j = 0; j < N_x; j++) {
				power_spectrum_phi[i][n][j] += phi_k2[j];
				power_spectrum_chi[i][n][j] += chi_k2[j];
				}
			if(decay_times[decay_times.size()-1] > 0) {
				num_of_decays++;
				if_decayed[i] = true;
				break;
			}
		}
	}
	for (int i = 0; i < N_ENS; i++) {
		if(!if_decayed[i]) {
			for (int n = 0; n < N_SAMPLE; n++) {
				for (int j = 0; j < N_x; j++) {
					power_spectrum_av_phi[n][j] += power_spectrum_phi[i][n][j];
					power_spectrum_av_chi[n][j] += power_spectrum_chi[i][n][j];
				}
			}
		}
	}
	for(int n = 0; n < N_SAMPLE; n++) {
		for (int j = 0; j < N_x; j++) {
			power_spectrum_av_phi[n][j] /= (N_ENS-num_of_decays);
			power_spectrum_av_chi[n][j] /= (N_ENS-num_of_decays);
		}
	}
	for (auto& time_slice : power_spectrum_av_phi) {
		save_data(time_slice, "Out/Out_power_spectrum_Lang/ps_phi", eta, TEMP, lambda);
	}	
	for (auto& time_slice : power_spectrum_av_chi) {
		save_data(time_slice, "Out/Out_power_spectrum_Lang/ps_chi", eta, TEMP, lambda);
	}	

	return 0;
}