#!/bin/bash
#SBATCH --job-name test_thermal_basic
#SBATCH --exclusive
#SBATCH -c 24
#SBATCH --mem=0
#SBATCH -o logs/zlog-%j.out
#SBATCH -e logs/zlog-%j.err
#SBATCH --constraint=xeon-p8

###################################
# Preparation
###################################
# ============================
# Supercloud Preparation:
# ============================
# Preparation for running in supercloud cluster:
module load anaconda/2021b

# Linking log files to more precisely named logs
logfile='test_thermal_basic'
ln -f logs/zlog-${SLURM_JOB_ID}.out logs/$logfile.out.${SLURM_JOB_ID}
ln -f logs/zlog-${SLURM_JOB_ID}.err logs/$logfile.err.${SLURM_JOB_ID}

# ============================
# Path preparation:
# ============================
# Should be run from the root folder /Ising

# -------------------------
# PYTHONPATH:
# -------------------------
# Adding the Ising directory to the PYTHONPATH
# Must be used in the directory /path/to/Ising/
chmod +x prepare_path.sh
./prepare_path.sh


###################################
# Beginning to log workflow
###################################
printf "##################################################
# Generating Projectors, Models, and Hamiltonians
##################################################"
printf "
# ============================
# Date: "`date '+%F'`"-("`date '+%T'`")
# ============================"
python examples/thermalization/params.py

printf "
##################################################
"

python examples/thermalization/test_basic.py

printf "
##################################################
"
printf "# ============================
# End time: "`date '+%F'`"-("`date '+%T'`")
# ============================"

# Remove duplicate log files:
rm logs/zlog-${SLURM_JOB_ID}.out
rm logs/zlog-${SLURM_JOB_ID}.err

