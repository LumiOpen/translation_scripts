#!/bin/bash
#SBATCH --job-name=llama70b_mt  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
#SBATCH --time=20:00:00       # Run time (d-hh:mm:ss)
#SBATCH --mem=480G
#SBATCH --account=project_462000353  # Project for billing



module use /appl/local/csc/modulefiles/
module load pytorch/2.4
source /scratch/project_462000444/zosaelai2/.fastchat_venv/bin/activate

echo "$(python -c 'import torch; print(torch.cuda.is_available())')"

export HF_HOME=/scratch/project_462000444/cache

src="eng"


languages=(
    'est'
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
    
        python translate_benchmarking.py \
                --model meta-llama/Llama-3.1-70B \
                --src_file /scratch/project_462000444/finetuning_data/FLORES-200/$src-devtest.txt \
                --trg_file /scratch/project_462000444/finetuning_data/FLORES-200/$trg-devtest.txt \
                --output_file /scratch/project_462000444/zosaelai2/translation_evals/Llama-3.1-70B-FLORES-$src-$trg-8-shot.txt \
                --lang_pair $src-$trg \
                --test_data flores-101 \

done
