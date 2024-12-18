#!/usr/bin/env python3
import os.path
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from argparse import ArgumentParser
from logging import warning

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    M2M100ForConditionalGeneration, 
    M2M100Tokenizer,
    MarianMTModel, 
    MarianTokenizer,
    StoppingCriteriaList,
    StoppingCriteria,
)

from utils import timed
from collections import Counter

from sacrebleu.metrics import BLEU

DTYPE_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

DMAP_CHOICES = ['auto', 'sequential']

# M2M language map
M2M_LANG_MAP = {
    "eng": "en",
    "fin": "fi",
    "swe": "sv",
    "dan": "da",
    "isl": "is",
    "nor": "no",
    "nob": "no",
}

# Default user and assistant token strings
DEFAULT_USER_START = '<|im_start|>user\n'
DEFAULT_ASST_START = '<|im_start|>assistant\n'
DEFAULT_END  = '<|im_end|>\n'

# Poro user and assistant token strings
PORO_USER_START = '<|user|>'
PORO_ASST_START = '<|assistant|>'
PORO_END  = 'END'

# AI-Sweden models
LLAMA_3_END = "<|end_of_text|>"
GPT_SW3_END = "<|endoftext|>"

# TRANSLATION TEMPLATES

# Language-agnostic templates
SRC_EQUALS_TRG_TEMPLATE = '{src}={trg}\n{end}\n'
LLAMA3_SRC_EQUALS_TRG_TEMPLATE = '{src}={trg}{end}\n'
USER_ASSISTANT_TEMPLATE = '{user}{src}\n{asst}{trg}\n{end}\n'

# GPT-SW3 translator template
GPT_SW3_TRANSLATOR_TEMPLATES = {
    'eng-swe': '<|endoftext|><s>User: Översätt till Svenska från Engelska\n{text}<s>Bot:',
    'swe-eng': '<|endoftext|><s>User: Översätt till Engelska från Svenska\n{text}<s>Bot:'
}

# Viking Templates by language
VIKING_TEMPLATES = {
    'dan': [
        '{user}Oversæt til {trg_lang}: {src}{end}{asst}{trg}{end}'
    ],
    'eng': [
        '{user}Translate into {trg_lang}: {src}{end}{asst}{trg}{end}'
    ],
    'fin': [
        '{user}Käännä {trg_lang}: {src}{end}{asst}{trg}{end}'
    ],
    'isl': [
        '{user}Þýða á {trg_lang}: {src}{end}{asst}{trg}{end}'
    ],
    'nor': [
        '{user}Oversett til {trg_lang}: {src}{end}{asst}{trg}{end}'
    ],
    'swe': [
        '{user}Översätt till {trg_lang}: {src}{end}{asst}{trg}{end}'
    ]
}

# EuroLLM template
EUROLLM_TEMPLATE = "{src_lang}: {src} {trg_lang}: {trg}"

# EMMA-500 template
EMMA_500_TEMPLATE_SYSTEM_PROMPT = "Translate the following sentence from {src_lang} to {trg_lang}\n"
EMMA_500_TEMPLATE_EXAMPLES = "[{src_lang}]: {src}\n[{trg_lang}]: {trg}"

EMMA_500_TEMPLATE = "Translate the following sentence from {src_lang} to {trg_lang}\n[{src_lang}]: {src}\n[{trg_lang}]: {trg}"

# CPT templates
CPT_ENG_TEMPLATE = "Translate into {trg_lang}: {src}\n{trg}"
CPT_FIN_TEMPLATE = "Käännä {trg_lang}: {src}\n{trg}"

# General template
GENERAL_PURPOSE_TEMPLATE = "## {src_lang}: {src}\n## {trg_lang}: {trg}"

TRG_LANGUAGE = {
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

# ICL examples per language (samples from FLORES-101 and Tatoeba dev sets)
ICL_EXAMPLES = {
    'tatoeba': {
            'eng-fin': {
                'src': ["Jesus was a carpenter.",
                        "According to some historians, Napoleon was an enlightened despot because of improvements he made in the social institutions of France. Others denounce him as an egocentric dictator because of the large number of people who died in his wars.",
                        "My grandmother ran off with a cowboy.",
                        "There must be something in the box.",
                        "This animal is the size of a beaver.",
                        "A lot of people around here like country music.",
                        "It's just a minor problem.",
                        "Tom lives in a brown house."],

                'trg': ["Jeesus oli puuseppä.",
                        "Toisten historijoitsijoiden mukaan Napoleon oli valistunut yksinvaltias, sillä hän teki useita parannuksia Ranskan yhteiskuntarakenteeseen. Toiset julistavat hänet itsekeskeiseksi hirmuvaltiaaksi hänen sodissaan kuolleen ihmismäärän johdosta.",
                        "Isoäitini karkasi lehmipojan matkaan.",
                        "Laatikossa on oltava jotakin.",
                        "Eläin on majavan kokoinen.",
                        "Monet ihmiset täällä päin pitävät country-musiikista.",
                        "Se on vain vähäpätöinen ongelma.",
                        "Tomi asuu ruskeassa talossa."]
            },
            'eng-swe': {
                'src': [
                    "Everyone wants to meet you. You're famous!",
                    "We'll win.",
                    "He likes animals.",
                    "You have got to be kidding me.",
                    "Columbus discovered America in 1492.",
                    "My wedding has to be perfect.",
                    "Melanie thinks that the situation is very bad.",
                    "Give me a bottle of wine."
                ],
                'trg': [
                    "Alla vill träffa dig. Du är känd!",
                    "Vi ska vinna.",
                    "Han tycker om djur.",
                    "Du måste skämta!",
                    "Columbus upptäckte Amerika 1492.",
                    "Mitt bröllop måste vara perfekt.",
                    "Melanie tycker att situationen är mycket dålig.",
                    "Ge mig en flaska vin."                
                    ]
            },
            "eng-dan": {
                "src": [
                    "I have two books.",
                    "The weather being fine, we went on a picnic.",
                    "Will you be ready by 2:30?",
                    "My father was about to leave when the phone rang.",
                    "He had a near-death experience.",
                    "You didn't vote, did you?",
                    "Blah.",
                    "Isn't that unconstitutional?"
                ],
                "trg": [
                    "Jeg har to bøger.",
                    "Fordi vejret var godt, tog vi på picnic.",
                    "Vil du være klar klokken halv tre?",
                    "Min far skulle til at gå da telefonen ringede.",
                    "Han havde en nærdødsoplevelse.",
                    "Du stemte ikke, gjorde du vel?",
                    "Bla.",
                    "Er det ikke forfatningsstridigt?"
                ]
            },
            "eng-isl": {
                "src": [
                    "I can't keep up with you if you walk so fast.",
                    "Stop beating on the door!",
                    "I will finish it by tomorrow afternoon.",
                    "I can't stand it anymore.",
                    "This tea is called green tea.",
                    "The traveler reached his destination at last.",
                    "I'll never tell this to anyone.",
                    "Thanks for the dinner."
                ],
                "trg": [
                    "Ég get ekki haldið í við þig ef þú gengur svona hratt.",
                    "Hættu að berja á dyrnar!",
                    "Ég mun klára það fyrir eftirmiðdaginn á morgun.",
                    "Ég þoli það ekki lengur!",
                    "Þetta te kallast grænt te.",
                    "Ferðalangurinn náði að lokum áfangastað sínum.",
                    "Ég mun aldrei segja neinum þetta.",
                    "Takk fyrir matinn."
                ]
            },
            "eng-nor": {
                "src": [
                    "She made her mother happy.",
                    "Can you take me to Boston with you?",
                    "She lived a lonely life.",
                    "Tom doesn't know the answer yet.",
                    "I don't like early morning meetings.",
                    "Last year, he was at sea for three months.",
                    "I'm proud to be a part of this project.",
                    "I will miss you all."
                ],
                "trg": [
                    "Hun gjorde sin mor glad.",
                    "Kan du ta meg med til Boston?",
                    "Hun levde et ensomt liv.",
                    "Tom vet ikke svaret ennå.",
                    "Jeg liker ikke tidlige morgenmøter.",
                    "I fjor var han tre måneder på sjøen.",
                    "Jeg er stolt av å være en del av dette prosjektet.",
                    "Jeg vil savne dere alle sammen."
                ]
            }
    },
    'flores-101': {
        'eng': ["The player who takes the fewest strokes, or swings of the club, to complete the course wins.",
                "The algae produces a neurotoxin that can disable nerves in both humans and fish.",
                "In April this year, a temporary restaining order was issued by Judge Glynn against the facility to enforce the release of those held more than 24 hours after their intake who did not receive a hearing by a court commissioner.",
                "Greed and selfishness will always be with us and it is the nature of cooperation that when the majority benefit there will always be more to gain in the short term by acting selfishly",
                "This causes the skater to turn. If the skates tilt to the right, the skater turns right, if the skates tilt to the left, the skater turns left.",
                "The game is based on the Second Battle of Fallujah, a vicious battle between American and Iraqi forces.",
                "Virgin have only purchased the \'good bank\' of Northern Rock, not the asset management company.",
                "England had experienced a long period of peace after the reconquest of the Danelaw."],
        'fin': ["Se pelaaja voittaa, joka tarvitsee vähiten lyöntejä radan suorittamiseen.",
                "Levät tuottavat hermomyrkkyä, joka voi lamauttaa sekä ihmisten että kalojen hermot.",
                "Tämän vuoden huhtikuussa tuomari Glynn määräsi poliisivankilalle tilapäisen kiellon. Sen tarkoituksena oli vapauttaa pidätetyt, joita tuomioistuimen edustajat eivät olleet kuulleet 24 tunnin kuluessa pidättämisestä.",
                "Ahneutta ja itsekkyyttä tulee olemaan aina. Yhteistyölle on myös luonteenomaista, että kun toimitaan enemmistön etujen puolesta, lyhyellä aikavälillä itsekäs toiminta tuottaa yksilölle eniten hyötyä.",
                "Sen takia luistelija kääntyy. Jos luistimet kallistuvat oikealle, luistelija kääntyy oikealle, jos luistimet kallistuvat vasemmalle, luistelija kääntyy vasemmalle.",
                "Peli perustuu Fallujahin toiseen taisteluun. Kyseessä oli väkivaltainen taistelu Yhdysvaltojen ja Irakin joukkojenvälillä.",
                "Virgin on ostanut vain Northern Rockin \"hyvän pankin\", ei Asset Managementia eli omaisuudenhoitopuolta.",
                "Englanti oli kokenut pitkän rauhanajan sen jälkeen, kun Danelagen oli valloitettu takaisin."],
        'swe': ["Vinner gör den spelare som behöver minst antal slag, eller svingar med klubban, för att slutföra banan.",
                "Algerna producerar ett nervgift som kan förstöra nerverna hos både människor och fiskar.",
                "I april i år utfärdade domare Judge Glynn ett tillfälligt besöksförbud för anläggningen, för att genomföra frigivningen av de som hållits i mer än 24 timmar efter sin intagning, och som inte blivit hörda av en domstolskommissionär.",
                "Girighet och själviskhet kommer alltid att finnas hos oss, och det är samarbetets natur att det när majoriteten gynnas alltid finns mer att vinna på kort sikt genom att agera själviskt",
                "Detta gör att skridskoåkaren svänger. Om skridskorna lutar till höger svänger åkaren åt höger, om skridskorna lutar till vänster svänger åkaren åt vänster.",
                "Spelet är baserat på det andra slaget om Fallujah, en grym strid mellan amerikanska och irakiska styrkor.",
                "Virgin har bara köpt den \"\"goda banken\"\" av Northern Rock, inte kapitalförvaltningsföretaget.",
                "England upplevde en lång period av fred efter återerövringen av Danelagen."],
        'dan': ["Den spiller, der bruger færrest slag eller sving med køllen for at fuldføre banen, vinder.",
                "Algerne producerer et neurotoksin, som kan lamme nerverne i både mennesker og fisk.",
                "I april i år blev der udstedt et midlertidigt polititilhold af dommer Glynn mod faciliteten, for at håndhæve løsladelsen af dem der blev tilbageholdt i mere end 24 timer, efter de blev anholdt, og som ikke blev afhørt af en kommissær.",
                "Grådighed og egoisme vil altid findes, og det er en del af den måde samarbejde fungerer på, at når flertallet har overtaget, vil det altid være muligt at vinde mere på kort sigt ved at handle egoistisk.",
                "Dette får skøjteløberen til at dreje. Hvis skøjterne hælder til højre, drejer skøjteløberen til højre, og hvis skøjterne hælder til venstre, drejer skøjteløberen til venstre.",
                "Spillet er baseret på Det Andet Slag ved Fallujah, en grusom kamp mellem de amerikanske og irakiske styrker.",
                "Virgin har kun købt den \"gode bank\" i Northern Rocks, og ikke kapitalforvaltningsselskabet.",
                "England havde gennemlevet en lang periode med fred ovenpå generobringen af Danelagen."],
        'isl': ["Sá leikmaður sem slær sjaldnast, eða sveiflar kylfunni sjaldnast, til að ljúka hringnum vinnur.",
                "Þörungarnir mynda taugaeitur sem getur gert taugar bæði hjá mönnum og fiskum óvígar.",
                "Í apríl í ár setti Glynn dómari tímabundið nálgunarbann á stofnunina. Tilgangurinn var að frelsa alla fangana sem höfðu setið í haldi í meira en 24 klukkustundir eftir handtöku og höfðu ekki fengið áheyrn hjá héraðsdómara.",
                "Græðgi og eigingirni munu alltaf fylgja okkur og það er eðlislægur hluti allrar samvinnu að þegar meirihlutinn græðir verður alltaf hægt að græða meira til skamms tíma með því að sýna eigingirni",
                "Þetta veldur því að skautarinn þarf að snúa sér. Ef skautarnir halla til hægri, þá beygir skautarinn til hægri, ef skautarnir halla til vinstri, þá beygir skautarinn til vinstri.",
                "Leikurinn er byggist á síðari orrustunni um Fallujah, sem var hræðilegur bardagi á milli bandarískra og íraskra hersveita.",
                "Virgin fyrirtækið keypti aðeins ,\'góða\' Northern Rock bankann, ekki eignastýringafyrirtækið.",
                "England upplifði langan friðartíma eftir endurupptöku Danalaga."],
        'nor': ["Spilleren som fullfører banen med færrest slag, eller svinger golfkøllen færrest ganger, vinner.",
                "Algene produserer en nervegift som kan skade nerver hos både mennesker og fisk.",
                "I april i år ble det utstedt et midlertidig påbud av dommer Glynn for å forhindre løslatelse av de som var anholdt mer enn 24 timer, som ikke hatt fått en høring av en rettskommisær.",
                "Grådighet og egoisme vil alltid være en del av oss, og det er samarbeidets natur at med en gang flertallet har en fordel, vil man alltid få mer igjen på kort sikt dersom man handler egoistisk.",
                "Dette gjør at skøyteløperen svinger. Hvis skøytene vippes mot høyre, så svinger de til høyre, hvis skøytene vippes mot venstre, så svinger de til venstre.",
                "Spillet tar utgangspunkt i det andre slaget om Fallujah, en brutal kamp mellom amerikanske og irakiske styrker.",
                "Virgin har bare kjøpt banken Northern Rock, ikke ressursforvaltningsselskapet.",
                "Etter gjenerobringen av Danelaw hadde England hatt en lang tid med fred."]
    }
}


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--min_new_tokens', default=10, type=int)
    ap.add_argument('--max_new_tokens', default=1024, type=int)
    ap.add_argument('--temperature', default=1.0, type=float)
    ap.add_argument('--memory-usage', action='store_true')
    ap.add_argument('--show-devices', action='store_true')    
    ap.add_argument('--dtype', choices=DTYPE_MAP.keys(), default='bf16')
    ap.add_argument('--device-map', choices=DMAP_CHOICES, default='auto')
    ap.add_argument('--trust-remote-code', default=None, action='store_true')
    # ap.add_argument('--transformers_cache',type=str, default="/scratch/project_462000319/transformers_cache")
    ap.add_argument('--model', type=str)
    ap.add_argument('--tokenizer', type=str)
    ap.add_argument('--src_file', type=str)
    ap.add_argument('--trg_file', type=str)
    ap.add_argument('--max_sentences', type=int, default=None)
    ap.add_argument('--output_file', type=str, default=None)
    ap.add_argument('--lang_pair', default="eng-fin", type=str)
    ap.add_argument('--test_data', default="tatoeba", type=str)
    ap.add_argument('--format_type', default="equals", type=str, help="chatml, user_assistant, equals")
    ap.add_argument('--num_examples', type=int, default=8, help="number of ICL examples to include in prompt")
    ap.add_argument('--skip_lines', type=int, default=None)
    return ap

class Llama3StopOnTokenCriteria(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id

class EuroLLMStoppingCriteria(StoppingCriteria):
    def __init__(self, stops = []):
      StoppingCriteria.__init__(self), 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
      self.stops = stops
      for i in range(len(stops)):
        self.stops = self.stops[i]

def report_memory_usage(message, out=sys.stderr):
    print(f'max memory allocation {message}:', file=out)
    total = 0
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.max_memory_allocated(i)
        print(f'  cuda:{i}: {mem/2**30:.1f}G', file=out)
        total += mem
    print(f'  TOTAL: {total/2**30:.1f}G', file=out)


def generate(prompt, model, tokenizer, args, end_token, skip_special_tokens=False):
    eos_token_id = tokenizer.eos_token_id
    if end_token:
        if "emma-500" in args.model.lower():
            eos_token_id = [eos_token_id, tokenizer.encode(end_token)[0], 29961]
        elif "salamandra" in args.model.lower() or "mistral" in args.model.lower() or "llama" in args.model.lower():
            eos_token_id = [eos_token_id]
            end_token_ids = tokenizer.encode(end_token)
            eos_token_id.extend(end_token_ids)
            print("end_token:", end_token)
            print("end_token_ids:", end_token_ids)
            print("eos_token_id:", eos_token_id)
            print("eos_token_id:", eos_token_id)
        else:
            end_token_ids = tokenizer.encode(end_token)[0]
            eos_token_id = [eos_token_id]
            eos_token_id.append(end_token_ids)
            print("end_token:", end_token)
            print("end_token_ids:", end_token_ids)
            print("eos_token_id:", eos_token_id)
            print("eos_token_id:", eos_token_id)
    
    print("--"*30)
    print(f"PROMPT:\n{prompt}")
    print("--"*30)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    if "llama-3-8b" in args.model.lower():
        stop_on_token_criteria = Llama3StopOnTokenCriteria(stop_token_id=tokenizer.bos_token_id)
        input_tokens = tokenizer(prompt, return_tensors="pt").input_ids
        dynamic_max_length = 4096 - input_tokens.shape[1]
        output = model.generate(
            input_ids,
            eos_token_id=eos_token_id,
            # max_new_tokens=args.max_new_tokens,
            max_length=dynamic_max_length,
            # truncation=True,
            stopping_criteria=StoppingCriteriaList([stop_on_token_criteria])
        )
    # elif "eurollm" in args.model.lower():
    #     # print("Using EuroLLM stopping criteria")
    #     # stop_on_token_criteria = EuroLLMStoppingCriteria(stops=[4])
    #     output = model.generate(
    #         input_ids,
    #         max_new_tokens=args.max_new_tokens,
    #         eos_token_id=eos_token_id,
    #         # max_new_tokens=args.max_new_tokens,
    #         # stopping_criteria=StoppingCriteriaList([stop_on_token_criteria])
    #     )
    # elif "mistral" in args.model.lower():
    #     stop_on_token_criteria = Llama3StopOnTokenCriteria(stop_token_id=tokenizer.eos_token_id)
    #     output = model.generate(
    #         input_ids,
    #         eos_token_id=eos_token_id,
    #         max_new_tokens=args.max_new_tokens,
    #         stopping_criteria=StoppingCriteriaList([stop_on_token_criteria])
    #     )
    else:
        output = model.generate(
            input_ids,
            eos_token_id=eos_token_id,
            max_new_tokens=args.max_new_tokens,
        )
    result =  tokenizer.decode(output[0], skip_special_tokens=skip_special_tokens)
    # print(f"\nRAW RESULT:\n{result}")
    if "gpt-sw3" in args.model:
        # print("GPT-SW3 RESULT 1:", result)
        result = result.split("<s> Bot: ")[-1].strip()
        if result.endswith(end_token):
            result = result[:-len(end_token)]
    # elif "salamandra" in args.model.lower():
    #     # result = result.strip().split("\n")[-1]
    #     lines = result.strip().split("\n")
    #     print("LINES:")
    #     for line_index, line in enumerate(lines):
    #         print(f"{line_index}: {line}")
    elif "llama" in args.model.lower() or "salamandra" in args.model.lower():
        # lines = result.strip().split("\n")
        # print("LINES:")
        # for line_index, line in enumerate(lines):
        #     print(f"{line_index+1}: {line}")
        result = result.strip().split("\n")[25].strip()
    else:
        # lines = result.strip().split("\n")
        # print("LINES:")
        # for line_index, line in enumerate(lines):
        #     print(f"{line_index+1}: {line}")
        if result.endswith(end_token):
            result = result[:-len(end_token)]
        if result.endswith(tokenizer.eos_token):
            result = result[:-len(tokenizer.eos_token)]
        # result includes the prompt, remove the prompt from the output
        result = result.replace(prompt, '', 1).strip()
    print(f"\nRESULT:\n{result}")
    return result

@timed
def translate_sentences(model, tokenizer, src_sentences, trg_sentences, args):
    if args.max_sentences is not None:
        src_sentences = src_sentences[:args.max_sentences]
    elif args.skip_lines is not None:
        src_sentences = src_sentences[args.skip_lines:]
        trg_sentences = trg_sentences[args.skip_lines:]
    src_lang = args.lang_pair.split("-")[0]
    trg_lang = args.lang_pair.split("-")[1]
    if src_lang == "nob":
        src_lang = "nor"
    if trg_lang == "nob":
        trg_lang = "nor"
    predictions = []
    print(f"trg_lang: {trg_lang}")
    print(f"src_lang: {src_lang}")
    for i, src_sent in enumerate(src_sentences):
        result = ""
        if "nllb" in args.model:
            if trg_lang == "nor":
                trg_lang = "nob"
            input_ids = tokenizer.encode(src_sent, return_tensors="pt", padding = True).to('cuda')
            output = model.generate(input_ids=input_ids,
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.pad_token_id,
                                     max_new_tokens=args.max_new_tokens,
                                     forced_bos_token_id=tokenizer.lang_code_to_id[trg_lang + "_Latn"])
            result = tokenizer.decode(output[0], skip_special_tokens=True)
            print(i+1, "src:", src_sent)
            print("pred:", result)
            print("-"*50)
        elif "m2m" in args.model:
            tokenizer.src_lang = M2M_LANG_MAP[src_lang]
            input_ids = tokenizer.encode(src_sent, return_tensors="pt").to('cuda')
            output = model.generate(input_ids=input_ids,
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.pad_token_id,
                                     max_new_tokens=args.max_new_tokens,
                                     forced_bos_token_id=tokenizer.get_lang_id(M2M_LANG_MAP[trg_lang]))
            result = tokenizer.decode(output[0], skip_special_tokens=True)
            print(i+1, "src:", src_sent)
            print("\npred:", result)
            print("-"*50)           
        elif "opus" in args.model:
            input_ids = tokenizer.encode(src_sent, return_tensors="pt").to('cuda')
            output = model.generate(input_ids=input_ids,
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.pad_token_id,
                                     max_new_tokens=args.max_new_tokens)
            result = tokenizer.decode(output[0], skip_special_tokens=True)
            print(i+1, "src:", src_sent)
            print("\npred:", result)
            print("-"*50) 
        else:
        # LLMs
            # Tatoeba examples
            if args.test_data == "tatoeba":
                if args.lang_pair == "eng-fin":
                    src_sents = ICL_EXAMPLES[args.test_data]['eng-fin']['src']
                    trg_sents = ICL_EXAMPLES[args.test_data]['eng-fin']['trg']
                elif args.lang_pair == "fin-eng":
                    src_sents = ICL_EXAMPLES[args.test_data]['eng-fin']['trg']
                    trg_sents = ICL_EXAMPLES[args.test_data]['eng-fin']['src']
                elif args.lang_pair == "eng-swe":
                    src_sents = ICL_EXAMPLES[args.test_data]['eng-swe']['src']
                    trg_sents = ICL_EXAMPLES[args.test_data]['eng-swe']['trg']
                elif args.lang_pair == "swe-eng":
                    src_sents = ICL_EXAMPLES[args.test_data]['eng-swe']['trg']
                    trg_sents = ICL_EXAMPLES[args.test_data]['eng-swe']['src']
                elif args.lang_pair == "eng-dan":
                    src_sents = ICL_EXAMPLES[args.test_data]['eng-dan']['src']
                    trg_sents = ICL_EXAMPLES[args.test_data]['eng-dan']['trg']
                elif args.lang_pair == "dan-eng":
                    src_sents = ICL_EXAMPLES[args.test_data]['eng-dan']['trg']
                    trg_sents = ICL_EXAMPLES[args.test_data]['eng-dan']['src']
                elif args.lang_pair == "eng-isl":
                    src_sents = ICL_EXAMPLES[args.test_data]['eng-isl']['src']
                    trg_sents = ICL_EXAMPLES[args.test_data]['eng-isl']['trg']
                elif args.lang_pair == "isl-eng":
                    src_sents = ICL_EXAMPLES[args.test_data]['eng-isl']['trg']
                    trg_sents = ICL_EXAMPLES[args.test_data]['eng-isl']['src']
                elif args.lang_pair == "eng-nor":
                    src_sents = ICL_EXAMPLES[args.test_data]['eng-nor']['src']
                    trg_sents = ICL_EXAMPLES[args.test_data]['eng-nor']['trg']
                elif args.lang_pair == "nor-eng":
                    src_sents = ICL_EXAMPLES[args.test_data]['eng-nor']['trg']
                    trg_sents = ICL_EXAMPLES[args.test_data]['eng-nor']['src']
                elif args.lang_pair == "fin-swe":
                    src_sents = ICL_EXAMPLES[args.test_data]['fin-swe']['src']
                    trg_sents = ICL_EXAMPLES[args.test_data]['fin-swe']['trg']
                elif args.lang_pair == "swe-fin":
                    src_sents = ICL_EXAMPLES[args.test_data]['fin-swe']['trg']
                    trg_sents = ICL_EXAMPLES[args.test_data]['fin-swe']['src']
            else:
                # FLORES examples
                src_sents = ICL_EXAMPLES[args.test_data][src_lang]
                trg_sents = ICL_EXAMPLES[args.test_data][trg_lang]
            src_sents = src_sents[:args.num_examples]
            trg_sents = trg_sents[:args.num_examples]
            # Viking / Poro
            if "viking" in args.model.lower() or "33B" in args.model.lower():
                if args.format_type == "chatml":
                    prompt = format_prompt_chatml(src_sent, src_sents, trg_sents, src_lang, trg_lang)
                    result = generate(prompt, model, tokenizer, args, end_token="<|im_end|>", skip_special_tokens=False)
                elif args.format_type == "user_assistant":
                    prompt = format_prompt_user_assistant(src_sent, src_sents, trg_sents)
                    result = generate(prompt, model, tokenizer, args, end_token="END", skip_special_tokens=True)
                else:
                    prompt = format_prompt_src_equals_trg(src_sent, src_sents, trg_sents, template=SRC_EQUALS_TRG_TEMPLATE, end_token="END")
                    result = generate(prompt, model, tokenizer, args, end_token="END", skip_special_tokens=False)

            else:
                # AI-Sweden models
                if "AI-Sweden" in args.model.lower():
                    if "gpt-sw3-6.7b-v2-translator" in args.model:
                        prompt = format_prompt_gpt_swe_translator(src_sent, src_lang, trg_lang)
                        result = generate(prompt, model, tokenizer, args, end_token="<s>", skip_special_tokens=False)
                    else:
                        prompt = format_prompt_src_equals_trg(src_sent, src_sents, trg_sents, template=LLAMA3_SRC_EQUALS_TRG_TEMPLATE, end_token = LLAMA_3_END)
                        result = generate(prompt, model, tokenizer, args, end_token=LLAMA_3_END, skip_special_tokens=True)
                # EuroLLM models
                elif "eurollm" in args.model.lower():
                    # print("EuroLLM model -- use EuroLLM prompt format")
                    prompt = format_prompt_eurollm(src_sent, src_sents, trg_sents, src_lang, trg_lang)
                    result = generate(prompt, model, tokenizer, args, end_token="<s>", skip_special_tokens=True)
                elif "emma-500" in args.model.lower():
                    prompt = format_prompt_emma500(src_sent, src_sents, trg_sents, src_lang, trg_lang)
                    result = generate(prompt, model, tokenizer, args, end_token="[", skip_special_tokens=True)
                elif "cpt" in args.model.lower() or "mistral" in args.model.lower():
                    prompt = format_prompt_cpt(src_sent, src_sents, trg_sents, src_lang, trg_lang)
                    result = generate(prompt, model, tokenizer, args, end_token="\n", skip_special_tokens=True)
                elif "salamandra" in args.model.lower():
                    prompt = format_prompt_cpt(src_sent, src_sents, trg_sents, src_lang, trg_lang)
                    result = generate(prompt, model, tokenizer, args, end_token="\n\n", skip_special_tokens=True)
                # elif "mistral" in args.model.lower():
                #     prompt = format_prompt_mistral_cpt(src_sent, src_sents, trg_sents, src_lang, trg_lang)
                #     result = generate(prompt, model, tokenizer, args, end_token="\n\n", skip_special_tokens=True)                    
            # print(i+1, "src:", src_sent)
            # print("\n pred:", result)
            # print("\n  trg:", trg_sentences[i])
            print("-"*50)
        with open(args.output_file, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "src": src_sent,
                        "trg": trg_sentences[i],
                        "prediction": result,
                    },
                    ensure_ascii=False
                )
                + "\n"
            )
        # remove \n from translation
        result = result.replace("\n", " ")
        with open(args.output_file.replace(".jsonl", ".txt"), "a", encoding="utf-8") as f:
             f.write( result + "\n")
        predictions.append(result)
    return predictions

def format_general_prompt(new_sent, src_sents, trg_sents, src_lang, trg_lang):
    template = GENERAL_PURPOSE_TEMPLATE
    prompts = []
    for src, trg in zip(src_sents, trg_sents):
        text = template.format(src_lang=TRG_LANGUAGE['eng'][src_lang],
                                trg_lang=TRG_LANGUAGE['eng'][trg_lang],
                               src=src,
                               trg=trg
                               ).strip()
        prompts.append(text)
    assistant_prompt = template.format(src_lang=TRG_LANGUAGE['eng'][src_lang],
                                       trg_lang=TRG_LANGUAGE['eng'][trg_lang],
                                       src=new_sent, 
                                       trg=""
                                       ).strip()
    prompt = "\n\n".join(prompts).strip()
    prompt += "\n\n" + assistant_prompt + "\n"
    return prompt

def format_prompt_cpt(new_sent, src_sents, trg_sents, src_lang, trg_lang):
    # if src_lang == "eng":
    template = CPT_ENG_TEMPLATE
    # else:
    #     template = MISTRAL_CPT_FIN_TEMPLATE
    prompts = []
    for src, trg in zip(src_sents, trg_sents):
        text = template.format(trg_lang=TRG_LANGUAGE['eng'][trg_lang],
                               src=src,
                               trg=trg
                               ).strip()
        prompts.append(text)
    assistant_prompt = template.format(trg_lang=TRG_LANGUAGE['eng'][trg_lang], 
                                       src=new_sent, 
                                       trg=""
                                       ).strip()
    prompt = "\n\n".join(prompts).strip()
    prompt += "\n\n" + assistant_prompt + "\n"
    return prompt

def format_prompt_emma500(new_sent, src_sents, trg_sents, src_lang, trg_lang):
    system_template = EMMA_500_TEMPLATE_SYSTEM_PROMPT
    examples_template = EMMA_500_TEMPLATE_EXAMPLES
    prompt = system_template.format(src_lang=src_lang, 
                                    trg_lang=trg_lang,
                                    ).strip()
    examples_prompts = []
    for src, trg in zip(src_sents, trg_sents):
        text = examples_template.format(src_lang=src_lang, 
                               src=src, 
                               trg_lang=trg_lang,
                               trg=trg
                               ).strip()
        examples_prompts.append(text)
    assistant_prompt = examples_template.format(src_lang=src_lang, 
                                       src=new_sent, 
                                       trg_lang=trg_lang,
                                       trg=""
                                       ).strip()
    examples_prompts.append(assistant_prompt)
    examples_prompt = "\n".join(examples_prompts).strip()
    prompt += "\n" + examples_prompt
    return prompt

def format_prompt_eurollm(new_sent, src_sents, trg_sents, src_lang, trg_lang):
    template = EUROLLM_TEMPLATE
    prompts = []
    # for src, trg in zip(src_sents, trg_sents):
    #     text = template.format(src_lang=TRG_LANGUAGE['eng'][src_lang], 
    #                            src=src, 
    #                            trg_lang=TRG_LANGUAGE['eng'][trg_lang],
    #                            trg=trg
    #                            ).strip()
    #     prompts.append(text)
    assistant_prompt = template.format(src_lang=TRG_LANGUAGE['eng'][src_lang], 
                                       src=new_sent, 
                                       trg_lang=TRG_LANGUAGE['eng'][trg_lang],
                                       trg=""
                                       ).strip()
    # prompts.append(assistant_prompt)
    # prompt = "\n".join(prompts).strip()
    return assistant_prompt

def format_prompt_gpt_swe(new_sent, src_sents, trg_sents, src_lang, trg_lang):
    template = VIKING_TEMPLATES[src_lang][0]
    prompts = []
    for src, trg in zip(src_sents, trg_sents):
        text = template.format(
                        src=src,
                        trg=trg,
                        trg_lang=TRG_LANGUAGE[src_lang][trg_lang],
                        user=DEFAULT_USER_START,
                        asst=DEFAULT_ASST_START,
                        end=DEFAULT_END,
                    )
        prompts.append(text)
    assistant_prompt = template.format(
        src=new_sent, 
        trg="", 
        trg_lang=TRG_LANGUAGE[src_lang][trg_lang], 
        user=DEFAULT_USER_START, 
        asst=DEFAULT_ASST_START,
        end=DEFAULT_END)
    prompt = "\n".join(prompts)
    # print("--"*20)
    # print("prompt:", prompts)
    # print("--"*20)
    # print("assistant_prompt:", assistant_prompt)
    assistant_prompt = assistant_prompt[:-len(DEFAULT_END)]
    # print("--"*20)
    prompt = prompt + "\n" + assistant_prompt
    # print("GPT-SWE PROMPT:", prompt)
    # print("--"*20)
    return prompt

def format_prompt_gpt_swe_translator(new_sent, src_lang, trg_lang):
    template = GPT_SW3_TRANSLATOR_TEMPLATES[f"{src_lang}-{trg_lang}"]
    prompt = template.format(text=new_sent)
    return prompt

def format_prompt_chatml(new_sent, src_sents, trg_sents, src_lang, trg_lang):
    template = VIKING_TEMPLATES[src_lang][0]
    prompts = []
    for src, trg in zip(src_sents, trg_sents):
        text = template.format(
                        src=src,
                        trg=trg,
                        trg_lang=TRG_LANGUAGE[src_lang][trg_lang],
                        user=DEFAULT_USER_START,
                        asst=DEFAULT_ASST_START,
                        end=DEFAULT_END,
                    )
        prompts.append(text)
    assistant_prompt = template.format(
        src=new_sent, 
        trg="", 
        trg_lang=TRG_LANGUAGE[src_lang][trg_lang], 
        user=DEFAULT_USER_START, 
        asst=DEFAULT_ASST_START,
        end=DEFAULT_END)
    prompt = "\n".join(prompts)
    assistant_prompt = assistant_prompt[:-len(DEFAULT_END)]
    prompt = prompt + "\n" + assistant_prompt
    # print("--"*20)
    # print("VIKING PROMPT:", prompt)
    # print("--"*20)
    return prompt

def format_prompt_src_equals_trg(new_sent, src_sents, trg_sents, template, end_token="END"):
    # template = SRC_EQUALS_TRG_TEMPLATE
    prompts = []
    for src, trg in zip(src_sents, trg_sents):
        text = template.format(
                        src=src,
                        trg=trg,
                        end=end_token,
                    )
        prompts.append(text)
    assistant_prompt = template.format(src=new_sent,
                                       trg="",
                                       end="")  
    prompt = "\n".join(prompts)
    prompt = prompt + "\n" + assistant_prompt.strip()
    return prompt

def format_prompt_user_assistant(new_sent, src_sents, trg_sents):
    template = USER_ASSISTANT_TEMPLATE
    prompts = []
    for src, trg in zip(src_sents, trg_sents):
        text = template.format(
                        src=src,
                        trg=trg,
                        user=PORO_USER_START,
                        asst=PORO_ASST_START,
                        end=PORO_END
                )
        prompts.append(text)
    assistant_prompt = template.format(src=new_sent,
                                       trg="",
                                        user=PORO_USER_START,
                                        asst=PORO_ASST_START,
                                       end="")  
    prompt = "\n".join(prompts)
    prompt = prompt + "\n" + assistant_prompt.strip()
    return prompt

def compute_spbleu_score(preds, refs, tokenize='flores101'):
    refs = [refs]
    bleu = BLEU(tokenize=tokenize)
    bleu_score = bleu.corpus_score(preds, refs)
    bleu_score = np.round(bleu_score.score, 1)
    return bleu_score

@timed
def load_model(args):
    print("Loading model:", args.model)
    if "opus" in args.model:
        model = MarianMTModel.from_pretrained(
            args.model,
            # cache_dir=args.transformers_cache,
        )
        model.to('cuda')
    elif "nllb" in args.model or "m2m" in args.model:
        model = M2M100ForConditionalGeneration.from_pretrained(
            args.model,
            # cache_dir=args.transformers_cache
        ) 
        model.to('cuda')       
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=args.device_map,
            torch_dtype=DTYPE_MAP[args.dtype],
            trust_remote_code=args.trust_remote_code,
            # cache_dir=args.transformers_cache,
        )
        #model.to('cuda')
    model.eval()
    return model


def check_devices(model, args):
    if args.show_devices:
        print(f'devices:', file=sys.stderr)
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if args.show_devices:
                print(f'  {name}.{param_name}:{param.device}', file=sys.stderr)
            elif param.device.type != 'cuda':
                warning(f'{name}.{param_name} on device {param.device}')

def main(argv):
    args = argparser().parse_args(argv[1:])
    if args.tokenizer is None:
        args.tokenizer = args.model
    if "opus" in args.tokenizer:
        tokenizer = MarianTokenizer.from_pretrained(args.tokenizer)
    elif "m2m" in args.tokenizer:
        tokenizer = tokenizer = M2M100Tokenizer.from_pretrained(args.tokenizer)
    else:
        if "nllb" in args.model:
            src_lang = args.lang_pair.split("-")[0]
            if src_lang == "nor":
                src_lang = "nob"
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=True, src_lang=src_lang + "_Latn")
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_model(args)
    if args.memory_usage:
        report_memory_usage('after model load')
    # load src sentences
    if ".jsonl" in args.src_file:
        src_sentences = [json.loads(line) for line in open(args.src_file)]
        src_sentences = [sent['sentence'].rstrip() for sent in src_sentences]
    else:
        src_sentences = open(args.src_file).readlines()
        src_sentences = [src.rstrip() for src in src_sentences]
    print("src_sentences:", len(src_sentences))
    # load target sentences
    if ".jsonl" in args.trg_file:
        trg_sentences = [json.loads(line) for line in open(args.trg_file)]
        trg_sentences = [sent['sentence'].rstrip() for sent in trg_sentences]
    else:
        trg_sentences = open(args.trg_file).readlines()
        trg_sentences = [trg.rstrip() for trg in trg_sentences]
    print("trg_sentences:", len(trg_sentences))
    predictions = translate_sentences(model, tokenizer, src_sentences, trg_sentences, args)
    # print("Model:", args.model)
    print("--- Done translating. Outputs saved to", args.output_file, "---")
    print("--- Computing spBLEU score ---")
    print("predictions:", len(predictions))
    print("trg_sentences:", len(trg_sentences))
    spbleu_score = compute_spbleu_score(predictions, trg_sentences[:len(predictions)])
    print("-"*20)
    print("| spBLEU score:", spbleu_score, "|")
    print("-"*20)
    if args.test_data == "flores-101" and len(predictions) > 100:
        spbleu_score = compute_spbleu_score(predictions[:100], trg_sentences[:100])
        print("-"*20)
        print("| First 100 sentences |")
        print("| spBLEU score:", spbleu_score, "|")
        print("-"*20)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
