# FVD_One

Codes for the paper 2605.xxxx. 

## Evolution scripts:

methods.cpp, methods.h : main code

Prepares initial states corresponding to the excited critical bubble in the tree-level potential, pushed along the negative mode of the bubble towards false or true vacuum.
Evolves the states, records decay times (zero if no decay), records sloshing events (returns to the false vacuum). 
The equations of motion of modes are available for models with the quartic potential (phi4) and the Liouville potential.

times.cpp : example evolution routine

times.sbatch : example slurm script 

## Processing scripts:

times_phi4.py : makes Fig. 5

oscs_phi4.py : makes Fig. 6

times_Liouville.py : makes Fig. 7

times_phi4_diss.py : makes Fig. 8

times_phi4_stoch.py : makes Fig. 9
