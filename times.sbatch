#!/bin/bash
#SBATCH --array=0-99
#SBATCH --account=def-matthewj
#SBATCH --time=0-23:59
#SBATCH --mem=4000
#SBATCH --cpus-per-task=1
#SBATCH --job-name=fvd_turnover
#SBATCH --output=./Out/slurm.out

# module load gcc/12.3
# module load fftw/3.3.10
# g++ -std=c++17 -o times_phi4.out times.cpp methods.cpp -lfftw3 -lm

SIZE=100                      # size of the box
N_x=4096                      # number of lattice points
DT=0.01                       # timestep
N_ENS=1000                    # number of realisations
TIME_SPAN=200                # evolution time
ETA=0.0                      # dissipation coefficient, eta=0 for conservative evolution
TEMP=0.1                     # temperature

./times_phi4.out $SIZE $N_x $DT $N_ENS $ETA $TIME_SPAN $TEMP >> out.txt
