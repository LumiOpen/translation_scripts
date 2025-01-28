#!/bin/bash
#SBATCH --job-name=mtbench_translate  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=dev-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G 
#SBATCH --time=01:00:00       # Run time (d-hh:mm:ss) # 4 hours for everything
#SBATCH --account=project_462000353  # Project for billing

module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/.fastchat_venv/bin/activate

#Download the Comet model and put in in a folder named Unbabel
#wget https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt20/wmt20-comet-qe-da.tar.gz
#tar -xf /wmt20-comet-qe-da.tar.gz
#mkdir Unbabel
#mv wmt20-comet-qe-da Unbabel/
#m wmt20-comet-qe-da.tar.gz

#pip install sacrebleu
#pip install comet
#pip install deepl

#mkdir -p logs

# Define the arguments
SOURCE_FILE="/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
OUTPUT_DIR="/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/mt_bench/"
LANGUAGES=("bg" "hr" "cs" "nl" "et" "fr" "de" "el" "hu" "is" "ga" "it" "lv" "lt" "mt" "nn" "pl" "pt" "ro" "sk" "sl" "es")
LANGUAGES=("lv")

export deepl_auth_key="xxx"

# Iterate over each language and run the translation script
for TGT_LANG in "${LANGUAGES[@]}"; do
    echo "Translating to $TGT_LANG..."
    python translate_mtbench.py --source-file $SOURCE_FILE \
                                --output-dir $OUTPUT_DIR \
                                --tgt-lang $TGT_LANG \
                                --deepl True \
    
    echo "---"
done
