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

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import timed
from sacrebleu.metrics import BLEU
# from comet import load_from_checkpoint
import random

DTYPE_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

DMAP_CHOICES = ['auto', 'sequential']

TRANSLATE_TEMPLATES = {
    'eng': ['Translate into {trg_lang}: {src_sent}', ],
    'fin': ['Käännä {trg_lang}: {src_sent}', ],
    'swe': ['Översätt till {trg_lang}: {src_sent}'],
    'dan': ['Oversæt til {trg_lang}: {src_sent}'],
    'isl': ['Þýða á {trg_lang}: {src_sent}'],
    'nor': ['Oversett til {trg_lang}: {src_sent}'],
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
        'bul': 'Bulgarian',
        'hrv': 'Croatian',
        'ces': 'Czech',
        'dan': 'Danish',
        'nld': 'Dutch',
        'eng': 'English',
        'est': 'Estonian',
        'fin': 'Finnish',
        'fra': 'French',
        'deu': 'German',
        'ell': 'Greek',
        'hun': 'Hungarian',
        'gle': 'Irish',
        'isl': 'Icelandic',
        'ita': 'Italian',
        'lav': 'Latvian',
        'lit': 'Lithuanian',
        'mlt': 'Maltese',
        'nor': 'Norwegian',
        'pol': 'Polish',
        'por': 'Portuguese',
        'ron': 'Romanian',
        'slk': 'Slovak',
        'slv': 'Slovenian',
        'spa': 'Spanish',
        'swe': 'Swedish'
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

# PROMPT EXAMPLES per language 
PROMPT_EXAMPLES = {
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
    ap.add_argument('--max_samples', default=None, type=int)
    ap.add_argument('--temperature', default=1.0, type=float)
    ap.add_argument('--memory-usage', action='store_true')
    ap.add_argument('--show-devices', action='store_true')    
    ap.add_argument('--dtype', choices=DTYPE_MAP.keys(), default='bf16')
    ap.add_argument('--device-map', choices=DMAP_CHOICES, default='auto')
    ap.add_argument('--trust-remote-code', default=None, action='store_true')
    ap.add_argument('--transformers_cache',type=str, default="/scratch/project_462000444/cache")
    ap.add_argument('--model', type=str)
    ap.add_argument('--tokenizer', type=str)
    ap.add_argument('--src_file', type=str)
    ap.add_argument('--trg_file', type=str)
    ap.add_argument('--src_lang', type=str)
    ap.add_argument('--trg_lang', type=str)
    ap.add_argument('--outfile', type=str)
    ap.add_argument('--n_shot', type=int, default=None)
    ap.add_argument('--example_data', type=str, default=None)
    return ap


def report_memory_usage(message, out=sys.stderr):
    print(f'max memory allocation {message}:', file=out)
    total = 0
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.max_memory_allocated(i)
        print(f'  cuda:{i}: {mem/2**30:.1f}G', file=out)
        total += mem
    print(f'  TOTAL: {total/2**30:.1f}G', file=out)

def compute_spbleu_score(preds, refs, tokenize='flores101'):
    print("Computing BLEU score")
    refs = [refs]
    bleu = BLEU(tokenize=tokenize)
    bleu_score = bleu.corpus_score(preds, refs)
    bleu_score = np.round(bleu_score.score, 1)
    # print("BLEU score:", bleu_score)
    return bleu_score

def compute_comet_score(preds, refs, sources, model_path):
    print("Computing COMET score")
    # print("Model path:", model_path)
    model = load_from_checkpoint(model_path)
    data = [{'src': src, 'mt': pred, 'ref': ref} for src, pred, ref in zip(sources, preds, refs)]
    model_output = model.predict(data, batch_size=16, gpus=2)
    # print (model_output) 
    print("-"*5, "COMET score:", model_output['system_score'], "-"*5)
    return model_output 

def create_translation_prompts(src_sents, src_lang, trg_lang, args):
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = len(src_sents)
    examples_src_sents = None
    examples_trg_sents = None
    if args.n_shot is not None and args.n_shot > 0:
        if args.example_data == 'flores-101' or args.example_data is None:
            examples_src_sents = PROMPT_EXAMPLES['flores-101'][src_lang]
            examples_trg_sents = PROMPT_EXAMPLES['flores-101'][trg_lang]
        else:
            print(f"src_lang: {src_lang}, trg_lang: {trg_lang}")
            if f'{src_lang}-{trg_lang}' in PROMPT_EXAMPLES['tatoeba']:
                examples_src_sents = PROMPT_EXAMPLES['tatoeba'][f'{src_lang}-{trg_lang}']['src']
                examples_trg_sents = PROMPT_EXAMPLES['tatoeba'][f'{src_lang}-{trg_lang}']['trg']
            elif f'{trg_lang}-{src_lang}' in PROMPT_EXAMPLES['tatoeba']:
                examples_src_sents = PROMPT_EXAMPLES['tatoeba'][f'{trg_lang}-{src_lang}']['trg']
                examples_trg_sents = PROMPT_EXAMPLES['tatoeba'][f'{trg_lang}-{src_lang}']['src']
            else:
                print(f"No examples found for {src_lang}-{trg_lang} in Tatoeba dataset")                
    print("Creating translation dataset with", max_samples, "samples")
    print("src_lang:", src_lang)
    print("trg_lang:", trg_lang)
    indexes = list(range(max_samples))
    prompts = []
    for i in indexes:
        src_sent = src_sents[i]
        template = TRANSLATE_TEMPLATES[src_lang][0]
        if examples_src_sents is not None:
            user_prompts = []
            for j in range(args.n_shot):
                example_src = examples_src_sents[j]
                example_trg = examples_trg_sents[j]
                user_prompt = template.format(trg_lang=TRANSLATE_TRG_LANGUAGE[src_lang][trg_lang], src_sent=example_src)
                user_prompts.append([user_prompt, example_trg])
            user_prompt = template.format(trg_lang=TRANSLATE_TRG_LANGUAGE[src_lang][trg_lang], src_sent=src_sent)
            user_prompts.append([user_prompt, None])
            prompts.append(user_prompts)
        else:
            user_prompt = template.format(trg_lang=TRANSLATE_TRG_LANGUAGE[src_lang][trg_lang],
                                            src_sent=src_sent)
            prompts.append(user_prompt)
    return prompts

@timed
def generate(prompts, tokenizer, model, args):
    print(f"Generating translations for {len(prompts)} sentences")
    eos_token_id = tokenizer.eos_token_id
    pipe = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            eos_token_id=[eos_token_id],
        )
    responses = []
    for i, prompt in enumerate(prompts):
        if args.n_shot is not None:
            messages = []
            for p in prompt:
                messages.append({"role": "user", "content": p[0]})
                if p[1] is not None:
                    messages.append({"role": "assistant", "content": p[1]})
        else:
            prompt = prompt.rstrip('\n')
            messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        print("-"*10, f"CHAT PROMPT {i+1} of {len(prompts)}", "-"*10)
        print(formatted_prompt)
        print("--"*20)
        generated = pipe(formatted_prompt)
        for g in generated:
            text = g['generated_text']
            # print(f"RAW RESPONSE: {text}")
            # print("--"*20)
            text = text.replace(formatted_prompt, '', 1)
            print(f"FORMATTED RESPONSE: {text}")
            print("--"*20)
            text = text.replace('\n', ' ')
            responses.append(text)
    if args.outfile is not None:
        with open(args.outfile, 'w') as f:
            for response in responses:
                f.write(response + '\n')
    print("Saved responses to", args.outfile)
    return responses

@timed
def load_model(args):
    print("Loading model:", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        # trust_remote_code=args.trust_remote_code,
        cache_dir=args.transformers_cache,
        attn_implementation='flash_attention_2',
    )
    # print("Done loading!")
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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_model(args)
    # if args.memory_usage:
    # report_memory_usage('after model load')
    src_sents = open(args.src_file).readlines()
    src_sents = [line.strip() for line in src_sents]
    trg_sents = open(args.trg_file).readlines()
    trg_sents = [line.strip() for line in trg_sents]
    formatted_prompts = create_translation_prompts(src_sents, args.src_lang, args.trg_lang, args)
    responses = generate(formatted_prompts, tokenizer, model, args)
    # compute translation metrics
    bleu_score = compute_spbleu_score(responses, trg_sents)
    print(f"{args.src_lang.upper()}-{args.trg_lang.upper()} BLEU score: {bleu_score}")
    print("="*40)
    if args.memory_usage:
        report_memory_usage('after generation')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
