#!/bin/bash
#SBATCH --job-name=format_sft_data  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=debug       # Partition (queue) name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --cpus-per-task=128     # Number of cores (threads)
#SBATCH --time=00:30:00         # Run time (hh:mm:ss)
#SBATCH --account=project_462000353  # Project for billing


module use /appl/local/csc/modulefiles/
module load pytorch

python format_data.py \
        --src_path /scratch/project_462000444/finetuning_data/EuroParl/en-sv/Europarl.en-sv.sv \
        --trg_path /scratch/project_462000444/finetuning_data/EuroParl/en-sv/Europarl.en-sv.en \
        --src_lang swe \
        --trg_lang eng \
        --outfile /scratch/project_462000444/finetuning_data/SFTTrainer_format/xling/EuroParl/en-sv/train.jsonl \
        --task translation \
        --start_index 500000 \
        --end_index 1000000
