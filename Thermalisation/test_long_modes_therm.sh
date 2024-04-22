#!/bin/bash

SIZE=100			# size of the box
N_x=8192			# number of lattice points
DT=0.100			# timestep
N_ENS=100			# number of realisations
N_SAMPLE=500		# number of sample points
TIME_SPAN=100		# time between samples

DISTR="RJ_THERMALMASS_DISTORTED"	# lattice RJ with thermal mass and some of the long modes at a different temperature
INTEG="RKN4-LN"						# 4th order Runge-Kutta-Nystrom (LF4c from Mclachlan) with L-N splitting

TEMP=0.1
ETA=0.01

declare -a VARS=($SIZE $N_x $DT $N_ENS $N_SAMPLE $TIME_SPAN $DISTR $INTEG $TEMP $ETA)

#g++ -std=c++17 -o test_long_modes_therm.out test_long_modes_therm.cpp methods_multi.cpp -lfftw3 -lm
#rm -rf ./Out/*

./test_long_modes_therm_Lang_12.out "${VARS[@]}" &		# T_long = T/2
./test_long_modes_therm_Lang_22.out "${VARS[@]}" &		# T_long = 2T
wait

python test_long_modes_therm.py "${VARS[@]}"		# make plots