#!/bin/bash
#SBATCH --job-name all_test
#SBATCH -o logs/all_test-%j.out

set -e
date

echo "Saving models..."
sbatch -W examples/slurm_scripts/save_models.sh
echo "Complete"
echo ""
echo "Plotting from test_thermal_basic..."
sbatch -W examples/slurm_scripts/test_thermal_basic.sh
echo "Complete"
echo ""
wait

date
echo "All tasks complete."
