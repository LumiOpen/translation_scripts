#!/bin/bash
#SBATCH --job-name=viking_chat_mt  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=dev-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
#SBATCH --time=3:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000353  # Project for billing

module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_462000444/zosaelai2/.fastchat_venv/bin/activate

for trg in fin swe dan nor isl
do
    if [ $trg = "nor" ]; then
        trgfile="nob"
    else
        trgfile=$trg
    fi
    echo "src: ENG"
    echo "trg: $trg"
    python translate_benchmarking_chat.py \
                --model /scratch/project_462000444/zosaelai2/models/viking-33b-instruction-collection-packed-epochs-3 \
                --src_file /scratch/project_462000444/finetuning_data/FLORES-101/eng-devtest.txt \
                --trg_file /scratch/project_462000444/finetuning_data/FLORES-101/$trgfile-devtest.txt \
                --src_lang eng \
                --trg_lang $trg \
                --max_samples 100 \
                --outfile /scratch/project_462000444/zosaelai2/translation_evals/viking_chat/test-33b-eng-$trg-v2.txt \
    
    echo "src: $trg"
    echo "trg: ENG"
    python translate_benchmarking_chat.py \
                --model /scratch/project_462000444/zosaelai2/models/viking-33b-instruction-collection-packed-epochs-3 \
                --src_file /scratch/project_462000444/finetuning_data/FLORES-101/$trgfile-devtest.txt \
                --trg_file /scratch/project_462000444/finetuning_data/FLORES-101/eng-devtest.txt \
                --src_lang $trg \
                --trg_lang eng \
                --max_samples 100 \
                --outfile /scratch/project_462000444/zosaelai2/translation_evals/viking_chat/33b-$trg-eng-v2.txt \

done

