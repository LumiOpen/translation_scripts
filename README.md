# Translation scripts
Python scripts for LLM machine translation

## Translation benchmarking for base models
`translate_benchmarking.py`: This is for benchmarking the translation capability of base LLMs (e.g., Poro, Viking, Mistral, Llama) using a few-shot prompt. Translation examples in the prompt are randomly selected from the dev sets of FLORES-101 and Tatoeba. This script also benchmarks open-source MT models such as Opus and NLLB.

### Example usage
Translating the FLORES-101 devtest sentences from English to Finnish with Viking-7B using an 8-shot prompt:
```
python translate_benchmarking.py \
            --model LumiOpen/Viking-7B   \
            --src_file /scratch/project_462000444/finetuning_data/FLORES-101/eng-devtest.txt \
            --trg_file /scratch/project_462000444/finetuning_data/FLORES-101/fin-devtest.txt \
            --output_file /scratch/project_462000444/translation_evals/FLORES-101/viking-7b-eng-fin.jsonl \
            --lang_pair eng-fin \
            --test_data flores-101 \
            --format_type equals \
            --num_examples 8 \
```

## Translation benchmarking for chat models
`translate_benchmarking_chat.py`: This is for benchmarking the translation capability of chat-tuned models (e.g. Poro-34B-Chat, Llama-3.1-8B-Instruct). We can evaluate the chat model with zero-shot prompting (no in-context examples) or with few-shot prompting. The script automatically applies the chat template indicated in the tokenizer config (`tokenizer_config.json`).

### Example usage
```
    python translate_benchmarking_chat.py \
                --model LumiOpen/Poro-34B-chat \
                --src_file /scratch/project_462000444/finetuning_data/FLORES-101/eng-devtest.txt \
                --trg_file /scratch/project_462000444/finetuning_data/FLORES-101/fin-devtest.txt \
                --src_lang eng \
                --trg_lang fin \
                --max_samples 100 \
                --outfile /scratch/project_462000444/zosaelai2/translation_evals/poro-33b-chat-eng-fin.txt \
    
```

## Translating datasets

`translate_datasets.py`: This is for translating SFT datasets using Poro and Viking from English to some target language. This assumes the dataset is in the HF SFTTrainer conversational [format](https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support):

```
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "..."}]}
```

### Example usage
Translating our instruction-collection dataset from English to Swedish with Viking-33B
```
python translate_datasets.py \
        --model LumiOpen/Viking-33B  \
        --filepath /scratch/project_462000444/finetuning_data/SFTTrainer_format/eng/instruction-collection/train.jsonl \
        --output_file /scratch/project_462000444/finetuning_data/SFTTrainer_format/swe/instruction-collection-viking/train.jsonl \
        --trg_lang swe \
```
