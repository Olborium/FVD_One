#!/bin/bash

# Records the evolution of the field across the barrier

SIZE=100                        # size of the box (L)
N_x=8192                        # number of lattice points
DT=0.01                         # time step (h)

TEMP=0.20                       # temperature
#ETA=0.100                      # dissipation coefficient

DISTR="RJ_THERMALMASS"          # thermal in. distr. with thermal mass
INTEG="RKN4-LN"                 # integration method

TIMESPAN=100                    # maximum evolution time
RECTIME=5                       # recording time
RECPOINTS=100                   # number of snapshots during the recording time

#g++ -std=c++17 -o sph_profile.out sph_profile.cpp methods_multi.cpp -lfftw3 -lm
#rm -rf ./Out/*

for ((i=1; i<300; i++)); do
    declare -a VARS=($SIZE $N_x $DT $TEMP $DISTR $INTEG $TIMESPAN $RECTIME $RECPOINTS $RANDOM)
    ./sph_profile.out "${VARS[@]}"
done

python sph_profile1.py "${VARS[@]}"    # process data, make some plots

python sph_profile2.py "${VARS[@]}"    # make more plots

python sph_profile3.py "${VARS[@]}"    # make animation