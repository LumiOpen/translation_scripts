#!/bin/bash
#SBATCH --job-name=eurollm_mt  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=dev-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000615 #353  # Project for billing

module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_462000353/mikakois/translation_scripts/.translation_bench_venv/bin/activate

export HF_HOME=/scratch/project_462000353/mikakois/hf_cache

#src=eng
for trg in bul hrv ces dan est nld fin fra deu ell hun isl gle ita lit lvs mlt nob pol por ron slk slv spa swe; 
    do python translate_benchmarking.py \
        --model /scratch/project_462000353/mikakois/models/EuroLLM-9B \
        --src_file /scratch/project_462000353/finetuning_data/FLORES-200/eng-devtest.txt \
        --trg_file /scratch/project_462000353/finetuning_data/FLORES-200/$trg-devtest.txt \
        --output_file /scratch/project_462000353/mikakois/translation_evals/test-euroLLM0-en-$trg.jsonl \
        --lang_pair eng-$trg --test_data flores-200 |tee output.txt;done

