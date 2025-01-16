import os
import json
import random
import sys
import glob
import pandas as pd
from argparse import ArgumentParser

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--src_path', default="src.txt", type=str)
    ap.add_argument('--trg_path', default="trg.txt", type=str)
    ap.add_argument('--distance_path', default="dist.jsonl", type=str)
    ap.add_argument('--min_neighbor_distance', default=None, type=float)
    ap.add_argument('--src_lang', default="eng", type=str)
    ap.add_argument('--trg_lang', default="fin", type=str)
    ap.add_argument('--outfile', default="out.jsonl", type=str)
    ap.add_argument('--max_samples', default=None, type=int)
    ap.add_argument('--start_index', default=None, type=int)
    ap.add_argument('--end_index', default=None, type=int)
    ap.add_argument('--task', default='translation', type=str)
    ap.add_argument("--concat_jsonl", type=str, default=None)
    ap.add_argument(
        "--data_langs",
        type=str,
        nargs="+",
        default=None,
        help="language for each dataset in data_paths",
    )
    ap.add_argument(
        "--data_paths",
        type=str,
        nargs="+",
        default=None,
        help="path to dataset for each language in data_langs",
    )
    return ap

TRANSLATE_TEMPLATES = {
    'eng': ["Translate into {trg_lang}: {src_sent}", 
            "Translate to {trg_lang}: {src_sent}",
            "Translate the following sentence into {trg_lang}: {src_sent}",
            "Can you translate this sentence into {trg_lang}: {src_sent}",
            ],
    'fin': ["Käännä {trg_lang}: {src_sent}", 
            "Käännä seuraava lause {trg_lang}: {src_sent}",
            "Käännä tämä lause {trg_lang}: {src_sent}",
            "Voitko kääntää seuraavan lauseen {trg_lang}: {src_sent}",
            "Voisitko kääntää tämän lauseen {trg_lang}: {src_sent}",
            "Olisitko ystävällinen ja kääntäisit tämän lauseen {trg_lang}?: {src_sent}",
            "Käännä ystävällisesti tämä lause {trg_lang}: {src_sent}",
            "Ole hyvä ja käännä tämä lause {trg_lang}: {src_sent}"
            ],
    'swe': ['Översätt till {trg_lang}: {src_sent}',
            "Översätt följande mening till {trg_lang}: {src_sent}",
            "Kan du översätta följande mening till {trg_lang}: {src_sent}",
            "Översätt denna mening till {trg_lang}: {src_sent}",
            "Skulle du kunna översätta följande mening till {trg_lang}: {src_sent}",
            ],
    'dan': ['Oversæt til {trg_lang}: {src_sent}',
            "Oversæt følgende sætning til {trg_lang}: {src_sent}",
            "Kan du oversætte denne sætning til {trg_lang}: {src_sent}",
            "Lav en {trg_lang} oversættelse af følgende sætning: {src_sent}",
            "Oversæt denne sætning til {trg_lang}: {src_sent}",
            "Vil du oversætte følgende sætning til {trg_lang}: {src_sent}"
    ],
    'isl': ['Þýða á {trg_lang}: {src_sent}',
            "Þýddu eftirfarandi setningu yfir á {trg_lang}: {src_sent}",
            "Getur þú þýtt þessa setningu yfir á {trg_lang}: {src_sent}",
            "Viltu þýða þessa setningu á {trg_lang}: {src_sent}",
            "Þýddu þessa setningu á {trg_lang}: {src_sent}",
    ],
    'nor': ["Oversett til {trg_lang}: {src_sent}",
            "Oversett følgende setning til {trg_lang}: {src_sent}",
            "Kan du oversette denne setningen til {trg_lang}: {src_sent}",
            "Lag en {trg_lang} oversettelse av følgende setning: {src_sent}",
            "Oversett denne setningen til {trg_lang}: {src_sent}",
            "Vennligst oversett følgende setning til {trg_lang}: {src_sent}"       
    ],
}

TRANSLATE_TRG_LANGUAGE = {
    'dan': {
        'dan': 'dansk',
        'eng': 'engelsk',
        'fin': 'finsk',
        'isl': 'islandsk',
        'nor': 'norsk',
        'swe': 'svensk',
    },
    'fin': {
        'dan': 'tanskaksi',
        'eng': 'englanniksi',
        'fin': 'suomeksi',
        'isl': 'islanniksi',
        'nor': 'norjaksi',
        'swe': 'ruotsiksi',
    },
    'eng': {
        'dan': 'Danish',
        'eng': 'English',
        'fin': 'Finnish',
        'isl': 'Icelandic',
        'nor': 'Norwegian',
        'swe': 'Swedish',
    },
    'isl': {
        'dan': 'dönsku',
        'eng': 'ensku',
        'fin': 'finnsku',
        'isl': 'íslensku',
        'nor': 'norsku',
        'swe': 'sænsku',
    },
    'nor': {
        'dan': 'dansk',
        'eng': 'engelsk',
        'fin': 'finsk',
        'isl': 'islandsk',
        'nor': 'norsk',
        'swe': 'svensk',
    },
    'swe': {
        'dan': 'danska',
        'eng': 'engelska',
        'fin': 'finska',
        'isl': 'isländska',
        'nor': 'norska',
        'swe': 'svenska',
    },
}

LANG_ID_TEMPLATES = {
    'eng': ['What language is this? {src_sent}', 
            'What language is this sentence: {src_sent}?',
            'What language is this written in? {src_sent}'],
    'fin': ['Mitä kieltä tämä on: {src_sent}?',
            'Mitä kieltä tämä on? {src_sent}',
            'Mitä kieltä tämä lause on? {src_sent}'],
    'dan': ['Hvilket sprog er denne sætning: {src_sent}?'],
    'isl': ['Hvaða tungumál er þetta: {src_sent}?'],
    'nor': ['Hvilket språk er denne setningen: {src_sent}?'],
    'swe': ['Vilket språk är denna mening: {src_sent}?'],
}

LANG_ID_TRG_LANGUAGE = {
    'eng': {
        'dan': 'Danish',
        'eng': 'English',
        'fin': 'Finnish',
        'isl': 'Icelandic',
        'nor': 'Norwegian',
        'swe': 'Swedish',
    },
    'fin': {
            'dan': 'Tanskaa',
            'eng': 'Englantia',
            'fin': 'Suomea',
            'isl': 'Islantia',
            'nor': 'Norjaa',
            'swe': 'Ruotsia',
    },
    'dan': {
        'dan': 'Dansk',
        'eng': 'Engelsk',
        'fin': 'Finsk',
        'isl': 'Islandsk',
        'nor': 'Norsk',
        'swe': 'Svensk',
    },
    'isl': {
        'dan': 'Dönsku',
        'eng': 'Ensku',
        'fin': 'Finnsku',
        'isl': 'Íslenska',
        'nor': 'Norsku',
        'swe': 'Sænsku',
    },
    'nor': {
        'dan': 'Dansk',
        'eng': 'Engelsk',
        'fin': 'Finsk',
        'isl': 'Islandsk',
        'nor': 'Norsk',
        'swe': 'Svensk',
    },
    'swe': {
        'dan': 'Danska',
        'eng': 'Engelska',
        'fin': 'Finska',
        'isl': 'Isländska',
        'nor': 'Norska',
        'swe': 'Svenska',
    },   
    
}

PARAPHRASE_TEMPLATES = {
    'fin': ['Kerro seuraava teksti eri sanoin: {input_text}',
            'Kerro seuraava eri tavalla: {input_text}',
            "Kerro seuraava teksti eri sanoin: \"{input_text}\"",
            "Kerro seuraava eri tavalla: \"{input_text}\"",
            ]
}

def flores_jsonl_to_txt(jsonl_path, txt_path, num_sentences=0):
    data = [json.loads(line) for line in open(jsonl_path)]
    with open(txt_path, 'a') as f:
        for i, d in enumerate(data):
            if num_sentences == 0 or i < num_sentences:
                if 'sentence' in d:
                    f.write(d['sentence'] + '\n')
                else:
                    f.write(d['sentence'] + '\n')

def select_random_flores_translation_samples(flores_path, src_lang, trg_lang, n_samples=8):
    src_sents = open(os.path.join(flores_path, src_lang + "-dev.txt")).readlines()
    src_sents = [sent.strip() for sent in src_sents]
    trg_sents = open(os.path.join(flores_path, trg_lang + "-dev.txt")).readlines()
    trg_sents = [sent.strip() for sent in trg_sents]
    indexes = random.sample(list(range(len(src_sents))), n_samples)
    print("FLORES", src_lang.upper(), "-", trg_lang.upper())
    selected_src = []
    selected_trg = []
    for i in indexes:
        selected_src.append(src_sents[i])
        selected_trg.append(trg_sents[i]) 
        #print(src_lang.upper(), ":", src_sents[i])
        # print(trg_lang.upper(), ":", trg_sents[i])
        # print("--"*20)
    print(src_lang.upper(), "\n", selected_src)
    print(trg_lang.upper(), "\n", selected_trg)

def select_random_flores_translation_samples_all_langs(flores_path, n_samples=8):
    langs = ['eng', 'fin', 'swe', 'dan', 'isl', 'nob']
    sentences = {}
    for lang in langs:
        sents = open(os.path.join(flores_path, lang + "-dev.txt")).readlines()
        sents = [sent.strip() for sent in sents]
        sentences[lang] = sents
    indexes = random.sample(list(range(len(sentences[langs[0]]))), n_samples)
    for j, index in enumerate(indexes):
        print("-"*20, "Sample", j+1, " - sentence", index+1, "-"*20)
        for lang in langs:
            print(lang.upper(), ":", sentences[lang][index])
        print("--"*30) 
        #print(src_lang.upper(), ":", src_sents[i])
        # print(trg_lang.upper(), ":", trg_sents[i])
        # print("--"*20)


def select_random_tatoeba_translation_samples(tatoeba_path, src_lang, trg_lang, n_samples=8):
    src_sents = open(os.path.join(tatoeba_path, src_lang + "-" + trg_lang, "dev.src")).readlines()
    src_sents = [sent.strip() for sent in src_sents]
    trg_sents = open(os.path.join(tatoeba_path, src_lang + "-" + trg_lang, "dev.trg")).readlines()
    trg_sents = [sent.strip() for sent in trg_sents]
    indexes = random.sample(list(range(len(src_sents))), n_samples)
    print("TATOEBA", src_lang.upper(), "-", trg_lang.upper())
    for i in indexes:
        print(src_lang.upper(), ":", src_sents[i])
        print(trg_lang.upper(), ":", trg_sents[i])
        print("--"*20)
    return src_sents, trg_sents


def create_language_alignment_data(data_all_langs, outpath):
    print("create language alignment DPO data")
    src_lang = "eng"
    langs = list(data_all_langs.keys())
    trg_langs = [lang for lang in langs if lang != src_lang]
    src_data = data_all_langs[src_lang]
    aligned_data = {lang: [] for lang in langs}
    for i in range(len(src_data)):
        messages_orig = src_data[i]['messages']
        prompt_orig = messages_orig[0]['content']
        messages_trg = []
        for trg_lang in trg_langs:
            trg_data = data_all_langs[trg_lang]
            for j in range(len(trg_data)):
                prompt = trg_data[j]['messages'][0]['orig_content']
                if prompt == prompt_orig:
                    messages_trg.append(trg_data[j]['messages'])
                    break
        if len(messages_trg) == len(trg_langs):
            aligned_data[src_lang].append(messages_orig)
            for k, lang in enumerate(trg_langs):
                aligned_data[lang].append(messages_trg[k])
    for lang in aligned_data:
        with open(os.path.join(outpath, lang + ".jsonl"), "w") as f:
            for entry in aligned_data[lang]:
                f.write(
                    json.dumps(entry, ensure_ascii=False)
                    + "\n"
                )

def create_language_switching_data(src_data, trg_data, src_lang, trg_lang, outfile):
    print("create language switching data")
    for i in range(len(src_data)):
        prompt_orig = src_data[i]['messages'][0]['content']
        messages_trg = []
        for j in range(len(trg_data)):
            prompt = trg_data[j]['messages'][0]['orig_content']
            if prompt == prompt_orig:
                messages_trg = trg_data[j]['messages']
                break
        if len(messages_trg) > 0:
            messages_src = src_data[i]['messages']
            if len(messages_src) >= 4 and len(messages_src) % 2 == 0 and len(messages_src) == len(messages_trg):
                switched_messages = {"messages": []}
                cur_lang = src_lang
                for k in range(len(messages_src)):
                    # odd turns are the start of user turns
                    if messages_src[k]['role'] == 'user' and messages_src[k+1]['role'] == 'assistant':
                        if cur_lang == src_lang:
                            switched_messages['messages'].append({'role': messages_src[k]['role'],
                                                      'content': messages_src[k]['content']})
                            switched_messages['messages'].append({'role': messages_src[k+1]['role'],
                                                      'content': messages_src[k+1]['content']})
                            cur_lang = trg_lang
                        else:
                            switched_messages['messages'].append({'role': messages_trg[k]['role'],
                                                      'content': messages_trg[k]['content']})
                            switched_messages['messages'].append({'role': messages_trg[k+1]['role'],
                                                      'content': messages_trg[k+1]['content']})
                            cur_lang = src_lang
                with open(outfile, "a") as f:
                    f.write(
                        json.dumps(switched_messages, ensure_ascii=False)
                        + 
                        "\n"
                    )
    print("Done! Created language switching dataset")
        

def create_translation_data(src_sents, trg_sents, src_lang, trg_lang, outfile, max_samples=None, start_index=None, end_index=None):
    if max_samples is not None:
        num_samples = max_samples
        indexes = list(range(max_samples))
        random.shuffle(indexes)
    else:
        num_samples = (end_index - start_index) 
        indexes = list(range(start_index, end_index))
    print("Creating translation dataset with", num_samples, "samples")
    print("src_lang:", src_lang)
    print("trg_lang:", trg_lang)
    with open(outfile, "a") as f:
        for i in indexes:
            # print("--- Sample", i+1, "---")
            src_sent = src_sents[i]
            trg_sent = trg_sents[i]
            # sample a template
            template = random.sample(TRANSLATE_TEMPLATES[src_lang], 1)[0]
            user_content = template.format(trg_lang=TRANSLATE_TRG_LANGUAGE[src_lang][trg_lang],
                                           src_sent=src_sent)
            # print("user:", user_content)
            assistant_content = trg_sent
            # print("assistant:", assistant_content)
            messages = {'messages':[]}
            messages['messages'].append({'role': 'user',
                                         'content': user_content})
            messages['messages'].append({'role': 'assistant',
                                         'content': assistant_content})
            f.write(
                json.dumps(messages, ensure_ascii=False) 
                    + "\n"
                    )
    print("Done! Created translation dataset with", num_samples, "samples")

def create_lang_identification_data(sents, sent_lang, outfile, max_samples=None):
    if max_samples is None:
        max_samples = len(sents)
    random.shuffle(sents) # shuffle sentences
    with open(outfile, "a") as f:
        for i in range(max_samples):
            # print("--- Sample", i+1, "---")
            # sample a language
            lang = random.sample(list(LANG_ID_TEMPLATES.keys()), 1)[0]
            # print("lang:", lang)
            # sample a template
            template = random.sample(LANG_ID_TEMPLATES[lang], 1)[0]
            user_content = template.format(src_sent=sents[i])
            assistant_content = LANG_ID_TRG_LANGUAGE[lang][sent_lang]
            # print("user:", user_content)
            # print("assistant:", assistant_content)
            messages = {'messages':[]}
            messages['messages'].append({'role': 'user',
                                         'content': user_content})
            messages['messages'].append({'role': 'assistant',
                                         'content': assistant_content})
            f.write(
                json.dumps(messages, ensure_ascii=False) 
                    + "\n"
                    )
    print("Done! Created language identification dataset with", max_samples, "samples")

def create_turku_paraphrase_data(filepath, outfile, max_samples=None):
    data = [json.loads(line) for line in open(filepath)]
    data = pd.DataFrame(data)
    # drop rows that share text1/text2 combinations
    data = data.drop_duplicates(subset='goeswith', keep="last")
    data = data.sample(frac=1) # shuffle dataframe
    input_sents = data.input.tolist()
    output_sents = data.output.tolist()
    if max_samples is None:
        max_samples = len(input_sents)
    with open(outfile, "w") as f:
        for i in range(max_samples):
            # sample paraphrase template
            template = random.sample(PARAPHRASE_TEMPLATES['fin'], 1)[0]
            user_content = template.format(input_text=input_sents[i])
            messages = {'messages':[]}
            messages['messages'].append({"role": "user",
                                         "content": user_content})
            messages['messages'].append({"role": "assistant",
                                         "content": output_sents[i]})
            f.write(
                json.dumps(messages, ensure_ascii=False)
                + "\n"
            )
    print("Done! Created paraphrase dataset with", max_samples, "samples")

def filter_data(data_filepath, distance_filepath, min_neighbor_distance, outfile):
    print("Filtering data", data_filepath, "with distances", distance_filepath)
    data = [json.loads(line) for line in open(data_filepath)]
    distances = [json.loads(line) for line in open(distance_filepath)]
    if min_neighbor_distance is not None:
        valid_prompts = [entry['instruction'] for entry in distances if entry['min_neighbor_distance'] >= min_neighbor_distance]
    else:
        valid_prompts = [entry['instruction'] for entry in distances if entry['repeat_count']==0]
    filtered_data = [entry for entry in data if entry['messages'][0]['content'] in valid_prompts]
    print("Filtered data:", len(filtered_data))
    with open(outfile, 'w') as f:
        for entry in filtered_data:
            f.write(
                json.dumps(entry, ensure_ascii=False)
                + "\n"
            )
    print("Done! Filtered data saved to", outfile)
    

def concat_datasets(path, outfile, with_source = False):
    with open(outfile, "a") as f:
        filepaths = glob.glob(path)
        for filepath in filepaths:
            print("Concatenating to collection:", filepath)
            data = [json.loads(line) for line in open(filepath)]
            source = ""
            if "dolly" in filepath:
                source = "databricks-dolly-15k"
            elif "oasst2" in filepath:
                source = "oasst2"
            elif "argilla" in filepath:
                source = "argilla-10k_prompts_ranked_mistral_large_responses"
            elif "FLORES-101" in filepath:
                source = "FLORES-101"
            elif "FLORES-200" in filepath:
                source = "FLORES-200"
            elif "paraphrase" in filepath:
                source = "turku_paraphrase_corpus"
            for entry in data:
                if with_source:
                    entry["source"] = source
                f.write(
                    json.dumps(entry, ensure_ascii=False)
                    + "\n"
                )
    print("Done! Concatenated data saved to", outfile)

def main(argv):
    args = argparser().parse_args(argv[1:])
    outfile = args.outfile
    max_samples = args.max_samples
    start_index = args.start_index
    end_index = args.end_index
    src_lang = args.src_lang
    trg_lang = args.trg_lang
    if src_lang == "nob":
        src_lang = "nor"
    if trg_lang == "nob":
        trg_lang = "nor"
    if args.task == 'translation':
        src_sents = open(args.src_path).readlines()
        trg_sents = open(args.trg_path).readlines()
        src_sents = [sent.strip() for sent in src_sents]
        trg_sents = [sent.strip() for sent in trg_sents]
        create_translation_data(src_sents, trg_sents, src_lang, trg_lang, outfile, max_samples=max_samples, start_index=start_index, end_index=end_index)
    elif args.task == 'lang_identification':
        src_sents = open(args.src_path).readlines()
        src_sents = [sent.strip() for sent in src_sents]
        create_lang_identification_data(src_sents, src_lang, outfile, max_samples)
    elif args.task == 'paraphrase':
        create_turku_paraphrase_data(args.src_path, outfile, max_samples=max_samples)
    elif args.task == 'select_tatoeba_samples':
        select_random_tatoeba_translation_samples(args.src_path, args.src_lang, args.trg_lang)
    elif args.task == 'select_flores_samples':
        # select_random_flores_translation_samples(args.src_path, args.src_lang, args.trg_lang)
        select_random_flores_translation_samples_all_langs(args.src_path)
    elif args.task == 'language_switching':
        src_data = [json.loads(line) for line in open(args.src_path)]
        trg_data = [json.loads(line) for line in open(args.trg_path)]
        create_language_switching_data(src_data=src_data, trg_data=trg_data, src_lang=args.src_lang, trg_lang=args.trg_lang, outfile=args.outfile)
    elif args.task == 'language_alignment_data':
        data_all_langs = {}
        for i, lang in enumerate(args.data_langs):
            data_all_langs[lang] = [json.loads(line) for line in open(args.data_paths[i])]
        create_language_alignment_data(data_all_langs, outfile)
    elif args.task == 'jsonl_to_txt':
        flores_jsonl_to_txt(args.src_path, args.trg_path, num_sentences=args.max_samples)
    elif args.task == 'filtering':
        filter_data(args.src_path, args.distance_path, args.min_neighbor_distance, outfile)
    else:
        if 'concat' in args.task and args.concat_jsonl is not None:
            concat_datasets(args.concat_jsonl, outfile, with_source=False)
        else:
            print("Specify another task or provide files to concatenate")


if __name__ == '__main__':
    sys.exit(main(sys.argv))
