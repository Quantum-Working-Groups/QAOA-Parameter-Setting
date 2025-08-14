# Scripts

This folder contains the scripts that we use to run the parameter setting methods on, for instance, classical compute clusters.
The repository contains the following scripts

* run_methods.sh: Runs the methods given as input in a certain directory using --method_dir. The instances used to run the methods are also given as input in another directory using --instance_dir. The directory to store results is given as --save_dir. The depths that are to be evaluated can be given with --depths as numbers separated by spaces. The script assumes the names of the method files include interp and transitionstates for the interpolation and transition states methods respectively. It also assumes the MaxCut problem is being evaluated, i.e., a pre-factor of -0.5.
