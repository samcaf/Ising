#!/bin/bash
#SBATCH --job-name all_thermal_test
#SBATCH -o logs/all_thermal_test-%j.out

set -e
echo "# ========================
# Performing Ising Tests
# ========================
Start time: "
date
echo "

# ------------------
Saving models
# ------------------
Starting at "
date
sbatch -W examples/slurm_scripts/thermalization/save_models.sh
echo "Complete at "
date
echo "

# ------------------
Plotting Basic Tests
# ------------------
Starting at "
date
sbatch -W examples/slurm_scripts/thermalization/test_basic.sh
echo "Complete at"
date
echo "


"
wait

echo "End time: "
date
echo "

All tasks complete."
