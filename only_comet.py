from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
import argparse
from tqdm import tqdm
import os
import numpy as np
import math
from comet import load_from_checkpoint

tqdm.pandas() 

"""
This script runs comet on two dataframes of the same length where the each has a column called "turns".
Turns contains a list of items. The two df's need to have the same structure and list lenght in turns
One is a hardcoded path to the English reference document
It runs comet reference free 
Author: Maria
"""

def opus_model_exists(model_name: str) -> bool:
    try:
        MarianTokenizer.from_pretrained(model_name)
        MarianTokenizer.from_pretrained(model_name)
        return True
    except Exception as e:
        print(f"Model {model_name} does not exist")
        return False

def translate_opus(text: str, tokenizer, model) -> str:
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Generate translation
    translated = model.generate(**inputs)
    
    # Decode the generated tokens
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    
    return translated_text

def translate_nllb(text: str, tokenizer, model, src_lang:str, tgt_lang:str) -> str:
    translator = pipeline('translation', 
                model=model, 
                tokenizer=tokenizer, 
                src_lang=src_lang, 
                tgt_lang=tgt_lang, 
                max_length = 800)
    output = translator(text)
    translated_text = output[0]['translation_text']
    return translated_text

def load_iso2nllb_map(filepath):
    iso2nllb_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                two_letter_code, long_code = line.strip().split()
                iso2nllb_dict[two_letter_code] = long_code
    return iso2nllb_dict

def map_lang_code(two_letter_code, iso2nllb_dict):
    return iso2nllb_dict.get(two_letter_code, "Unknown code")

def run_comet(df: pd.DataFrame):
    """This function takes the list in turns and translated turns and explodes them 
    such that each list item gets its own row. Then runs comet on it.
    Using GPUs throws an error"""
    model = load_from_checkpoint('Unbabel/wmt20-comet-qe-da/checkpoints/model.ckpt')

    #reformat for Comet
    df_exploded = df.explode(['turns', 'translated_turns'])

    # Create the new DataFrame with the required structure
    new_df = df_exploded.rename(columns={'turns': 'src', 'translated_turns': 'mt', 'reference': 'ref'})
    data = new_df[['src', 'mt']].to_dict('records')
    model_output = model.predict(data, batch_size=4, gpus=0)

    print(f"Comet score: {model_output.system_score}")

def check_lists_and_lengths(df1, df2, column):
    """
    Check if all fields in a column contain lists and if the list lengths are the same for both DataFrames.
    
    Args:
    df1 (pd.DataFrame): First DataFrame.
    df2 (pd.DataFrame): Second DataFrame.
    column (str): Column name to check.
    
    Returns:
    bool: True if all fields contain lists and the list lengths are the same, False otherwise.
    """
    assert len(df1) == len(df2) 
    # Check if all fields in the column contain lists
    all_lists_df1 = df1[column].apply(lambda x: isinstance(x, list)).all()
    all_lists_df2 = df2[column].apply(lambda x: isinstance(x, list)).all()
    
    if not all_lists_df1 or not all_lists_df2:
        return False
    
    # Check if the list lengths are the same for both DataFrames
    same_lengths = (df1[column].apply(len) == df2[column].apply(len)).all()
    
    return same_lengths


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text using a specified model.")
    #parser.add_argument('--src_lang', type=str, required=False, default='en', help="Source language code - two letter iso. Only supports English at the moment")
    parser.add_argument('--tgt-lang', type=str, required=True, help="Target language code - two-letter iso")
    parser.add_argument('--source-file', type=str, required=False, default='/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl', help="Path to the English source file")
    parser.add_argument('--output-dir', type=str, required=True, help="Path to the output file")
    parser.add_argument('--max-samples', type=int, default=None, help="Only take top n rows")
    args = parser.parse_args()
    
    src_lang = 'en'
    # check if the opus model exists

    #open the English source file
    eng = pd.read_json(args.source_file, lines=True)
    if args.tgt_lang == 'nb':
        tgt = pd.read_json(os.path.join(args.output_dir, f"question_no.jsonl"), lines=True)
    else:
        tgt = pd.read_json(os.path.join(args.output_dir, f"question_{args.tgt_lang}.jsonl"), lines=True)
    
    column = 'turns'
    assert check_lists_and_lengths(eng, tgt, 'turns'), f"List lengths in {column} are not the same for both DataFrames."

    if args.max_samples:
        eng = eng.head(args.max_samples)
        tgt = tgt.head(args.max_samples)

    eng['translated_turns'] = tgt['turns']

    run_comet(eng)