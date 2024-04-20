#!/bin/bash

# Stochastic integrator test:
# Computes evolution of the power spectrum of phi and pi
# Measures its properties 

SIZE=100                        # size of the box (L)
N_x=8192                        # number of lattice points
DT=0.0025                       # time step (h)
N_ENS=100                       # number of realisations
N_SAMPLE=200                    # number of samples
TIME_SPAN=0.2                   # time between samples

DISTR="RJ_THERMALMASS"          # thermal in. distr. with thermal mass
INTEG="Neri3Stoch_LN"           # integration method

TEMP=0.1                        # temperature
SELFINT=1.0                     # self-interaction sign

ETA=1.0                         # dissipation coefficient

declare -a VARS=($SIZE $N_x $DT $N_ENS $N_SAMPLE $TIME_SPAN $DISTR $INTEG $TEMP $ETA $SELFINT)

#rm -rf ./Out/*
#g++ -std=c++17 -o test_Lang_ensemble.out test_Lang_ensemble.cpp methods_multi.cpp -lfftw3 -lm

./test_Lang_ensemble.out "${VARS[@]}"

python test_Lang_ensemble_PLOT.py "${VARS[@]}"