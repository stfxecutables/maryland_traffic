#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=24:00:00
#SBATCH --job-name=traffic
#SBATCH --output=traffic__%j_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --profile=all
#SBATCH --array=4,5,6,7

PROJECT=$SCRATCH/maryland_traffic

cd "$PROJECT" || exit 1
bash run_traffic_data.sh

# --array=0-15
