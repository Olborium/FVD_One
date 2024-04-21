#include "methods_multi.h"

// Records the evolution of the field across the barrier

int main(int argc, char* argv[])
{			
	SIZE = std::stod(argv[1]);		
	N_x = std::stoi(argv[2]);
	DT = std::stod(argv[3]);	
	TEMP = std::stod(argv[4]);	
//	double eta = std::stod(argv[11]); // uncomment if Langevin eom is solved
	double m[1] = {1.0};
	double l[1] = {-1.0};
	DX = SIZE/N_x;
///////////////////////////////////////////////////// 
	std::vector<std::vector<double> > Phi;
	std::vector<std::vector<double> > Chi;

	Sphaleron sph(1, m, l, argv[5], argv[6]);	// KG eom
//  Sphaleron sph(eta, m, l, argv[5], argv[6]); // Langevin eom

	sph.history(Phi, Chi, std::stod(argv[7]), std::stod(argv[8]), std::stoi(argv[9]));	

	for (auto& time_slice : Phi) {
		save_data(time_slice, "Out/sph_profile/sph(t)", TEMP, std::stoi(argv[10])); }
	for (auto& time_slice : Chi) {
		save_data(time_slice, "Out/sph_profile/sph_dot(t)", TEMP, std::stoi(argv[10])); }

	return 0;
}