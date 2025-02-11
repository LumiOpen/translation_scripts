#!/bin/bash
#SBATCH --job-name=eurollm_flores  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
#SBATCH --time=06:00:00       # Run time (d-hh:mm:ss)
#SBATCH --mem=480G
#SBATCH --account=project_462000353  # Project for billing

export HF_HOME=/scratch/project_462000444/cache

module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_462000444/zosaelai2/.fft_venv/bin/activate
echo "$(python -c 'import torch; print(torch.cuda.is_available())')"

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
        echo "Target lang: $trg"
        if [ $trg = "nor" ]; then
                trgfile="nob"
        else
                trgfile=$trg
        fi
    
        python translate_benchmarking.py \
                --model utter-project/EuroLLM-9B \
                --src_file /scratch/project_462000444/finetuning_data/FLORES-101/eng-devtest.txt \
                --trg_file /scratch/project_462000444/finetuning_data/FLORES-101/$trgfile-devtest.txt \
                --output_file /scratch/project_462000444/zosaelai2/translation_evals/viking_paper/eurollm-9b-eng-$trgfile-8-shot.jsonl \
                --lang_pair eng-$trg \
                --test_data flores-101 \
                
        python translate_benchmarking.py \
                --model utter-project/EuroLLM-9B \
                --src_file /scratch/project_462000444/finetuning_data/FLORES-101/$trgfile-devtest.txt \
                --trg_file /scratch/project_462000444/finetuning_data/FLORES-101/eng-devtest.txt \
                --output_file /scratch/project_462000444/zosaelai2/translation_evals/viking_paper/eurollm-9b-$trgfile-eng-8-shot.jsonl \
                --lang_pair $trg-eng \
                --test_data flores-101 \

done
