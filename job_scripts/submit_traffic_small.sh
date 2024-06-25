#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=10:00:00
#SBATCH --job-name=traffic_small
#SBATCH --output=traffic_small__%j_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --profile=all
#SBATCH --array=0-15

PROJECT=$SCRATCH/df-analyze

cd "$PROJECT" || exit 1
bash run_traffic_data_small.sh