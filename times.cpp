#include "methods.h"

int main(int argc, char* argv[])
{	
	SIZE = std::stod(argv[1]);
	N_x = std::stoi(argv[2]);
	DT = std::stod(argv[3]);
	int N_ENS = std::stoi(argv[4]);
	double eta = std::stod(argv[5]);
	double TIME_SPAN = std::stod(argv[6]);
	TEMP = std::stod(argv[7]);
	DX = SIZE/N_x;
	kappa = 0.05;                              // only for Liouville

///////////////////////////////////////////////////// 
	std::vector<double> O2; 
	std::vector<std::vector<double>> modes;
	load_data(modes, "modes_phi4.txt");        // modes_Liouville.txt for Liouville
	load_data(O2, "eigenvalues_phi4.txt");     // eigenvalues_Liouville.txt for Liouville

///////////////////////////////////////////////////// 

	std::vector<double> decay_times, osc_counter;
	
/////////////////////////////////////////////////////

	Evolve sol(eta, 4, 3, -1);		// for the conservative case use sol(0, 4, 1, -1);

	for(int i = 0; i < N_ENS; i++) {

		sol.prepare_barrier_state(modes, O2);

		sol.evolve(decay_times, osc_counter, TIME_SPAN);
    
	}
	
	std::random_device rd;
    int seed = rd();
	save_data(2, decay_times, "Phi4/times", TEMP, eta, std::abs(seed));   // Phi4/ftimes for P_{+-} (returns to the false vacuum)
	save_data(2, osc_counter, "Phi4/osc_counter", TEMP, eta, std::abs(seed));

	return 0;
}
