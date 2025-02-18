#!/bin/bash
#SBATCH --job-name=mtbench_translate  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=4
#SBATCH --mem=256G 
#SBATCH --time=04:00:00       # Run time (d-hh:mm:ss) # 4 hours for everything
#SBATCH --account=project_462000353  # Project for billing

module use /appl/local/csc/modulefiles/
module load pytorch
source /scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/.fastchat_venv/bin/activate
 
#pip install sacrebleu
#pip install comet
#pip install deepl
#pip install litellm

#install google-cloud-sdk and make sure you are signed in with the right account
# make sure to use the right project associated with the API key
gcloud config set project 880367206890
# On the first day, it didn't work. I got 429 ressource exhaused after only one prompt, sometimes two. 
#The next day, it worked without code changes.

#mkdir -p logs

# Define the arguments
SOURCE_FILE="/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
OUTPUT_DIR="/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/mt_bench/deepl"
OUTPUT_DIR="/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/mt_bench/gemini"
mkdir -p $OUTPUT_DIR
LANGUAGES=("bg" "hr" "cs" "nl" "et" "fr" "de" "el" "hu" "is" "ga" "it" "lv" "lt" "mt" "nn" "pl" "pt" "ro" "sk" "sl" "es")
LANGUAGES=("it" "lt" "pl" "pt" "ro" "sk" "sl" "es" "mt" "nn" "ga")

export deepl_auth_key=
export GEMINI_API_KEY=

export SSL_CERT_FILE="" # litellm does not run with this environment variable set. The value was /etc/ssl/ca-bundle.pem I restore it immediately after running the script

# Iterate over each language and run the translation script
for TGT_LANG in "${LANGUAGES[@]}"; do
    echo "Translating to $TGT_LANG..."
    python translate_mtbench.py --source-file $SOURCE_FILE \
                                --output-dir $OUTPUT_DIR \
                                --tgt-lang $TGT_LANG \
                                --gemini True \
    
    echo "---"
done

export SSL_CERT_FILE=/etc/ssl/ca-bundle.pem # restore the SSL_CERT_FILE environment variable