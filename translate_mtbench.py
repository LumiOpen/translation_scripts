from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
import argparse
from tqdm import tqdm
import os
import numpy as np
import math
from comet import load_from_checkpoint, download_model
import deepl
#from google import genai
import time
import sys
from litellm import completion

tqdm.pandas() 

"""
This script translates an mt bench question file from English to a target language.
It assumes that the text to be translated is in the column turn and references and is a flat list. 
It checks whether an Opus model exist for the English -> target language and if not:
use NLLB instead. Or you can set the --deepl or gemini flag and use DeepL/Gemini for translation.
It runs Comet without a reference to get an estimate of the translation quality.
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

def translate_deepl(text:str, tgt_lang:str, translator) -> str:
    if len(text)<1:
        return text
    else:
        result = translator.translate_text(text, target_lang=tgt_lang)
        return(result.text)  
    
def get_lang_code_dict(value:str) -> dict:
    """
    Read in a csv file with alpha 2 and 3 language codes and the language name
    Given either a language name or an alpha 2 or alpha 3 code, return a dict of the row or None
    Query the dict with the desired value, either: alpha3-b, alpha3-t, alpha2, English, French
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv('/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/lang_codes.csv', 
                     comment='#')
    row = None     
    for i in range(df.shape[1]):
        # Check if the value is in either column
        if value in df.iloc[:, i].values:
            row = df[df.iloc[:, i] == value]
            # returns the first match - it will not work if there are ambiguities
            row_dict = row.to_dict(orient='records')[0]
            return row_dict
    if row == None:
        return {}

def translate_gemini(text:str, tgt_lang:str, translator) -> str:

    lang = get_lang_code_dict(tgt_lang)['English']

    instruction=f"Translate the following sentence to {lang}. Say nothing else than the translation, but keep any code or functions such that the question can be answered from your output alone. Never give the answer to the question or try to solve the problem - only translate. The sentence is: {text}"
    print(instruction)

    for attempt in range(8):
        try:
            response = completion(
                model="gemini/gemini-2.0-flash-001",
                #model="gemini/gemini-1.5-pro",
                #model="gemini/gemini-1.5-flash",
                messages=[{"role": "user", "content": instruction}],
            )
            if response and response.choices:
                answer = response.choices[0].message.content
                print(answer)
                return answer
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(10)
    
    raise Exception("Failed to translate after 8 attempts")
    

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

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    #model = load_from_checkpoint('Unbabel/wmt20-comet-qe-da/checkpoints/model.ckpt')

    #reformat for Comet
    df_exploded = df.explode(['turns', 'translated_turns'])

    # Create the new DataFrame with the required structure
    new_df = df_exploded.rename(columns={'turns': 'src', 'translated_turns': 'mt', 'reference': 'ref'})
    data = new_df[['src', 'mt']].to_dict('records')
    model_output = model.predict(data, batch_size=4, gpus=0)

    print(f"Comet score: {model_output.system_score}")

def check_api_key():
    """
    Check if the API key is available in the environment.
    """
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable with the API key.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text using a specified model.")
    #parser.add_argument('--src_lang', type=str, required=False, default='en', help="Source language code - two letter iso. Only supports English at the moment")
    parser.add_argument('--tgt-lang', type=str, required=True, help="Target language code - two-letter iso")
    parser.add_argument('--source-file', type=str, required=False, default='/scratch/project_462000353/maribarr/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl', help="Path to the English source file")
    parser.add_argument('--output-dir', type=str, required=True, help="Path to the output file")
    parser.add_argument('--max-samples', type=int, default=None, help="Only take top n rows")
    parser.add_argument('--deepl', type=bool, required=False, default=False, help="Whether to use DeepL for translation")
    parser.add_argument('--gemini', type=bool, required=False, default=False, help="Whether to use Gemini for translation")
    args = parser.parse_args()
    
    # Load the model and tokenizer
    model_name = f'Helsinki-NLP/opus-mt-en-{args.tgt_lang}'
    checkpoint = 'facebook/nllb-200-distilled-1.3B'
    #checkpoint = 'facebook/nllb-200-distilled-600M'
    output_file = os.path.join(args.output_dir, f"question_{args.tgt_lang}.jsonl")

    print(f"Translating MT bench questions from en to {args.tgt_lang}")
    print(f"Reading en questions from {args.source_file}")

    if sum([args.deepl, args.gemini]) > 1:
        print('Select either deepL or Gemini, not both')
        sys.exit(1)

    using_NLLB = False # flag to specify that we use NLLB
    src_lang = 'en'
    # check if the opus model exists
    if args.deepl:
        print("Using DeepL translator")
        auth_key =  os.getenv('deepl_auth_key')
        translator = deepl.Translator(auth_key)
        translate_func = translate_deepl

    if args.gemini:
        print("Using Gemini Flash")
        check_api_key()
        translate_func=translate_gemini
        translator=None

    elif opus_model_exists(model_name):
        print("Opus model found")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    else:
        using_NLLB = True
        print(f"No Opus model found for {args.tgt_lang} Using NLLB")
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # map 2 letter lang code to long lang code used by NLLB
        filepath_isomap = '/scratch/project_462000353/maribarr/translation_scripts/iso2nllb.map'
        iso2nllb_dict = load_iso2nllb_map(filepath_isomap)
        
        src_lang_long = map_lang_code('en', iso2nllb_dict)
        tgt_lang_long = map_lang_code(args.tgt_lang, iso2nllb_dict)
        print("NLLB language codes: ", src_lang_long, tgt_lang_long)
        #print("not running NLBB now - passing")
        
    #open the English source file
    df = pd.read_json(args.source_file, lines=True)

    if args.max_samples:
        df = df.head(args.max_samples)

    if args.deepl or args.gemini:

        df.loc[:, 'translated_turns'] = df.turns.progress_map(lambda x: [translate_func(text=sent,
                                                                                        translator=translator,
                                                                                        tgt_lang=args.tgt_lang) if len(sent) > 2 else sent for sent in x])
        df.loc[:, 'translated_reference'] = df.reference.progress_map(lambda x: [translate_func(text=sent,
                                                                                            translator=translator,
                                                                                            tgt_lang=args.tgt_lang)
                                                                                            for sent in x] if isinstance(x, list) else x)
    elif not using_NLLB:    #Then translate using Opus
        df.loc[:, 'translated_turns'] = df.turns.progress_map(lambda x: [translate_opus(sent,
                                                                                        tokenizer=tokenizer,
                                                                                        model=model) for sent in x])
        df.loc[:, 'translated_reference'] = df.reference.progress_map(
            lambda x: [translate_opus(sent,
                                      tokenizer=tokenizer,
                                      model=model) for sent in x] if type(x)==list else x )

    else: #then use nllb
        df.loc[:, 'translated_turns'] = df.turns.progress_map(lambda x: [translate_nllb(sent,
                                                                        tokenizer=tokenizer,
                                                                        model=model,
                                                                        src_lang=src_lang_long,
                                                                        tgt_lang=tgt_lang_long) 
                                                                        for sent in x ])
        df.loc[:, 'translated_reference'] = df.reference.progress_map(lambda x: [translate_nllb(sent,
                                                                        tokenizer=tokenizer,
                                                                        model=model,
                                                                        src_lang=src_lang_long,
                                                                        tgt_lang=tgt_lang_long) 
                                                                        for sent in x ] if type(x)==list else x )      
             
    run_comet(df)
    # save to file
    #drop the english question column
    df = df.loc[:, [c for c in df.columns if c not in ["turns", 'reference']]]
    df.rename(columns={'translated_turns': 'turns', 'translated_reference': 'reference'}, inplace=True)

    with open(output_file, "w") as f:
        f.write(df.to_json(orient='records', lines=True, force_ascii=False))

    print(f'Wrote to file {output_file}')
    print('Done')