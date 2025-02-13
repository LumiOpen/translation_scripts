#!/bin/bash
#SBATCH --job-name=translate_sft_prompts  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=dev-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
#SBATCH --time=03:00:00       # Run time (d-hh:mm:ss)
#SBATCH --mem=480G
#SBATCH --account=project_462000353  # Project for billing


module use /appl/local/csc/modulefiles/
module load pytorch/2.4
source /scratch/project_462000444/zosaelai2/.fastchat_venv/bin/activate
export HF_HOME="/scratch/project_462000444/cache"
echo "$(python -c 'import torch; print(torch.cuda.is_available())')"

src="eng"

languages=(
    'bul'
    'hrv'
    'ces'
    'dan'
    'nld'
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

# for trg in "${languages[@]}"

for trg in bul
do
        python translate_datasets.py \
                --model /scratch/project_462000353/converted-checkpoints/europa_7B_iter_0715255_bfloat16  \
                --filepath /scratch/project_462000444/finetuning_data/SFTTrainer_format/eng/instruction-collection/train.jsonl \
                --output_file /scratch/project_462000444/finetuning_data/SFTTrainer_format/$trg/instruction-collection-prompts/train.jsonl \
                --trg_lang $trg \
                --translate_roles user

done
