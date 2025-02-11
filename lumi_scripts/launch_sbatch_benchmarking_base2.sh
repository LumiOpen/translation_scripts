#!/bin/bash
#SBATCH --job-name=europa_mt_eval  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
#SBATCH --time=10:00:00       # Run time (d-hh:mm:ss)
#SBATCH --mem=480G
#SBATCH --account=project_462000353  # Project for billing


module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_462000444/zosaelai2/.fastchat_venv/bin/activate
echo "$(python -c 'import torch; print(torch.cuda.is_available())')"

export HF_HOME=/scratch/project_462000444/cache

src="eng"

languages=(
    'bul'
    'hrv'
    'cze'
    'dan'
    'nld'
    'eng'
    'est'
    'fin'
    'fra'
    'deu'
    'ell'
    'hun'
    'isl'
    'gle'
    'ita'
    'lav'
    'lit'
    'mlt'
    'nob'
    'pol'
    'por'
    'ron'
    'slk'
    'slv'
    'spa'
    'swe'
)

for trg in "${languages[@]}"
do
        echo "Target lang: ${trg^^}"
        python translate_benchmarking.py \
                --model /scratch/project_462000353/converted-checkpoints/europa_7B_iter_0715255_bfloat16 \
                --src_file /scratch/project_462000444/finetuning_data/FLORES-101/eng-devtest.txt \
                --trg_file /scratch/project_462000444/finetuning_data/FLORES-101/$trg-devtest.txt \
                --output_file /scratch/project_462000444/zosaelai2/translation_evals/europa-FLORES-eng-$src-$trg-8-shot.jsonl \
                --lang_pair eng-$trg \
                --test_data flores-101 \

done
