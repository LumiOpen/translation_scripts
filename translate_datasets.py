#!/usr/bin/env python3
import os
import sys
import json
import torch
from argparse import ArgumentParser
from logging import warning
from utils import timed

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    MarianMTModel, 
    MarianTokenizer,
    M2M100ForConditionalGeneration, 
    M2M100Tokenizer,
)

# language detection
import fasttext
fasttext.FastText.eprint = lambda x: None
FASTTEXT_LID_BINARY = "/scratch/project_462000444/zosaelai2/lid.176.bin"
LID_MODEL = fasttext.load_model(FASTTEXT_LID_BINARY)
LANG_THRESHOLD = 0.5

NLLB_LANG_MAP = {
    "fi": "fin_Latn",
    "fin": "fin_Latn",
    "sv": "swe_Latn",
    "swe": "swe_Latn",
    "da": "dan_Latn",
    "dan": "dan_Latn",
    "nob": "nob_Latn",
    "nno": "nno_Latn",
    "is": "isl_Latn",
    "isl": "isl_Latn",
}

LANG_CODE_MAP = {
    "fin": "fi",
    "swe": "sv",
    "dan": "da",
    "nob": "no",
    "nno": "no",
    "isl": "is",
}

# Default user and assistant token strings
DEFAULT_USER_START = '<|im_start|>user\n'
DEFAULT_ASST_START = '<|im_start|>assistant\n'
DEFAULT_END  = '<|im_end|>\n'

# Poro user and assistant token strings
PORO_USER_START = '<|user|>'
PORO_ASST_START = '<|assistant|>'
PORO_END  = 'END'

# Language-agnostic templates
SRC_EQUALS_TRG_TEMPLATE = '{src}={trg}\n{end}\n'
USER_ASSISTANT_TEMPLATE = '{user}{src}\n{asst}{trg}\n{end}\n'

# ICL examples per language (samples from FLORES-101 and Tatoeba dev sets)
FLORES_SENT_INDICES = [532, 136, 51, 587, 356, 119, 152, 381]
ICL_EXAMPLES = {
    'tatoeba': {
            'eng-fin': {
                'src': [
                        "Jesus was a carpenter.",
                        "My grandmother ran off with a cowboy.",
                        "There must be something in the box.",
                        "This animal is the size of a beaver.",
                        "A lot of people around here like country music.",
                        "It's just a minor problem.",
                        "Tom lives in a brown house.",
                        "According to some historians, Napoleon was an enlightened despot because of improvements he made in the social institutions of France. Others denounce him as an egocentric dictator because of the large number of people who died in his wars."
                        ],

                'trg': [
                        "Jeesus oli puuseppä.",
                        "Isoäitini karkasi lehmipojan matkaan.",
                        "Laatikossa on oltava jotakin.",
                        "Eläin on majavan kokoinen.",
                        "Monet ihmiset täällä päin pitävät country-musiikista.",
                        "Se on vain vähäpätöinen ongelma.",
                        "Tomi asuu ruskeassa talossa.",
                        "Toisten historijoitsijoiden mukaan Napoleon oli valistunut yksinvaltias, sillä hän teki useita parannuksia Ranskan yhteiskuntarakenteeseen. Toiset julistavat hänet itsekeskeiseksi hirmuvaltiaaksi hänen sodissaan kuolleen ihmismäärän johdosta.",
                        ]
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
        'eng-fin': {
            'src': ["They also said in a statement, \"The crew is currently working to determine the best method of safely extracting the ship\".",
                    "Each country also has unique laws requiring what emergency items need to be in the car.",
                    "The pit is either heated with hot stones from a fire, or in some places geothermal heat makes areas of ground naturally hot.",
                    "Heels should be low and wide. Sand, gravel or salt (calcium chloride) is often scattered on roads or paths to improve traction.",
                    "Entering Southern Africa by car is an amazing way to see all the region's beauty as well as to get to places off the normal tourist routes.",
                    "Jardín de la Unión. This space was built as the atrium for a 17th-century convent, of which the Templo de San Diego is the sole surviving building.",
                    "Its second claw was larger, giving rise to the name Hesperonychus which means \"western claw.\"",
                    "The steel needle floats on top of the water because of surface tension.",
                    ],
            'trg': ["Se ilmoitti lausunnossaan myös, että \"miehistö tekee parhaillaan töitä löytääkseen parhaan tavan irrottaa laiva turvallisesti\".",
                    "Jokaisella maalla on myös omat lakinsa siitä, mitä pakollisia varusteita autossa tulee olla.",
                    "Kuoppa lämmitetään joko nuotiosta otetuilla kuumilla kivillä, tai joissakin paikoissa maalämpö kuumentaa joitakin maaperän kohtia luonnollisesti.",
                    "Koron tulisi olla matala ja leveä. Hiekkaa, soraa tai suolaa (kalsiumkloridi) on usein siroteltu teille ja poluille pidon parantamiseksi.",
                    "Etelä-Afrikkaan matkustaminen autolla on mahtava tapa nähdä koko alueen kauneus ja päästä paikkoihin tavallisten turistireittien ulkopuolella.",
                    "Jardín de la Unión. Tila rakennettiin atriumiksi 1600-luvun luostariin, jonka ainoa jäljellä oleva rakennus on Templo de San Diego.",
                    "Sen toinen kynsi oli suurempi, mistä juontuu \"läntistä kynttä\" tarkoittava nimi hesperonychus.",
                    "Teräsneula kelluu veden päällä pintajännityksen ansiosta.",
                    ]
        },
        'eng-swe': {
            'src': [
                    "Many exotic animals are hard to find, and parks sometimes have rules about taking photographs for commercial purposes.",
                    "The Sun doesn't have a crust like the Earth that you can stand on. The whole Sun is made out of gases, fire, and plasma.",
                    "One of the world's richest people, Allen has reportedly invested much of his wealth in marine exploration and began his quest to find the Musashi out of a lifelong interest in the war.",
                    "Day hiking involves distances of less than a mile up to longer distances that can be covered in a single day.",
                    "The Cook Islands are an island country in free association with New Zealand, located in Polynesia, in the middle of the South Pacific Ocean.",
                    "The Global Running Tours successor, Go Running Tours networks dozens of sightrunning providers on four continents.",
                    "A tornado is a spinning column of very low-pressure air, which sucks the surrounding air inward and upward.",
                    "Before the Spanish arrived in the 16th century, northern Chile was under Inca rule while the indigenous Araucanians (Mapuche) inhabited central and southern Chile."
            ],
            'trg': [
                    "Många exotiska djur är svåra att hitta, och parker har ibland regler kring fotografering för kommersiella ändamål.",
                    "Solen har ingen skorpa likt Jorden som du kan stå på. Hela solen är uppbyggd av gaser, eld och plasma.",
                    "En av världens rikaste personer, Allen, rapporteras ha investerat mycket av sin förmögenhet i marin utforskning, och inledde sitt sökande efter Musashi på grund av ett livslångt intresse för kriget.",
                    "Dagsvandring innefattar distanser på mindre än en mile upp till längre distanser som kan avklaras på en och samma dag.",
                    "Cooköarna är ett öland fritt förbundet med Nya Zeeland, beläget i Polynesien, mitt i södra Stilla havet.",
                    "\"Global Running Tours efterträdare, Go Running Tours, kopplar samman dussintals leverantörer av så kallad \'sightrunning\' på fyra kontinenter.\"",
                    "En tornado är en snurrande pelare med mycket lågt lufttryck som suger den omgivande luften inåt och uppåt.",
                    "Innan spanjorerna anlände på 1500-talet var norra Chile under Inka-styre medan de inhemska araukanerna (Mapuche) bebodde de centrala och södra delarna av Chile."
            ]
        },
        'eng-dan': {
            'src':[
                "In the nomadic phase, army ants march at night and stop to camp during the day.",
                "Learning to create interactive media requires conventional and traditional skills, as well as tools mastered in interactive classes (storyboarding, audio and video editing, story telling, etc.)",
                "The Cook Islands are an island country in free association with New Zealand, located in Polynesia, in the middle of the South Pacific Ocean.",
                "The whole district is designated as a UNESCO World Heritage Site for its unique cultural and historical value, and its property values are among the highest of the country.",
                "Brazil is the largest Roman Catholic country on Earth, and the Roman Catholic Church has consistently opposed the legalization of same-sex marriage in the country.",
                "One of the world's richest people, Allen has reportedly invested much of his wealth in marine exploration and began his quest to find the Musashi out of a lifelong interest in the war.",
                "The use of the Internet and the World Wide Web allows learners to have access to information at all times.",
                "Vautier's achievements outside of directing include a hunger strike in 1973 against what he viewed as political censorship."
            ],
            'trg':[
                "I nomadefasen marcherer hærmyrer om natten og stopper for at slå lejr om dagen.",
                "At lære at skabe interaktive medier kræver konventionelle og traditionelle færdigheder samt værktøjer, der er lært i interaktive kurser (storyboards, lyd- og videoredigering, historiefortælling osv.)",
                "Cook-øerne er en østat i fri tilknytning til New Zealand, Øerne er beliggende i Polynesien, midt i det sydlige Stillehav.",
                "Hele distriktet er udpeget som et UNESCO verdensarvssted for dens unikke kulturelle og historiske værdi, og dens ejendomsværdier er blandt de højeste i hele landet.",
                "Brasilien er det største romersk-katolske land i verden, og den romersk-katolske kirke har konsekvent modsat sig legaliseringen af homoseksuelle ægteskaber i landet.",
                "Allen, som er en af verdens rigeste personer, har efter sigende investeret en stor del af sin formue i havudforskning og begyndte sin søgen efter Musashi på grund af sin livslange interesse i krigen.",
                "Brugen af internettet og World Wide Web giver elever adgang til information på et hvilket som helst tidspunkt.",
                "Vautiers bedrifter ud over at være filminstruktør inkluderer en sultestrejke i 1973 mod, hvad han betragtede som politisk censur."
            ]
        },
        'eng-isl': {
            'src': [
                "Regarding the global financial situation, Zapatero continued by saying that \"the financial system is a part of the economy, a crucial part.\"",
                "Fred currently has winds of 105 miles per hour (165 km/h) and is moving towards the northwest.",
                "Sikhs consider their faith to be a separate religion from Hinduism though they acknowledge its Hindu roots and traditions.",
                "They normally offer higher bandwidth and better quality of service. They are encrypted and thus harder to spy on.",
                "Former U.S. Speaker of the House Newt Gingrich came in second with 32 percent.",
                "Interactive design requires that you re-assess your assumptions about media production and learn to think in a non-linear ways.",
                "It is the biggest acquisition in eBay's history.",
                "Stewart, Gordon, Kenseth, and Harvick round out the top-ten positions for the Drivers' Championship with four races remaining in the season."
            ],
            'trg': [
                "Varðandi fjármálaaðstæður á heimsvísu hélt Zapatero áfram með því að segja að \"fjármálakerfið væri hluti af hagkerfinu, mikilvægur hluti.\"",
                "Fred er sem stendur með vindhraða upp á 105 mílur á klukkustund (165 km/klst.) og stefnir í norðvestur.",
                "Síkar telja trú sína vera aðskilda trú frá hindúatrú þó að þeir viðurkenni hindúarætur þess og venjur.",
                "Þeir bjóða venjulega meiri bandvídd og betri gæði þjónustu. Þau eru dulkóðuð og því erfiðari að njósna um.",
                "Fyrrum þingforseti fulltrúadeildarinnar, Newt Gingrich, var annar með 32 prósent.",
                "Gagnvirk hönnun kallar á endurmat þitt á fyrri ályktunum hvað varðar framleiðslu margmiðlunarefnis og þú lærir að hugsa á ólínulegan hátt.",
                "Þetta er stærsta yfirtaka í sögu eBay.",
                "Stewart, Gordon, Kenseth og Harvick eru í tíu efstu sætunum hvað ökumannameistaratitillinn varðar þegar fjórar keppnir eru eftir á tímabilinu."
            ]
        },
        'eng-nor': {
            'src': [
                "RSPCA New South Wales chief inspector David O'Shannessy told the ABC that surveillance and inspections of abattoirs should be commonplace in Australia.",
                "An immigration checkpoint is usually the first stop when disembarking from a plane, a ship, or another vehicle.",
                "Hsieh also argued that the photogenic Ma was more style than substance.",
                "The man allegedly drove a three-wheeled vehicle armed with explosives into a crowd.",
                "The two sides would meet in the major semi final where Noosa ran out winners by 11 points.",
                "Dark clouds unrelated to any volcanic activity were reported at the base of the mountain.",
                "To return to their previous energy level, they must get rid of the extra energy they got from the light.",
                "These agents are responsible for providing government and judicial services under Article 247 of the Pakistani Constitution."
            ],
            'trg': [
                "Sjefinspektør for RSPCA New South Wales, David O'Shannessy, fortalte ABC at overvåkning og inspeksjoner av slakteriene bør være en selvfølge i Australia.",
                "Punktet for innvandringskontroll er som regel 1. stopp når man forlater et fly, et skip eller et annet kjøretøy.",
                "Hsieh argumenterte også for at Ma som var veldig fotogen, hadde mer stil enn substans.",
                "Mannen hadde med seg sprengstoff og skal ha kjørt inn i en folkemengde med en trehjuling.",
                "Begge gruppene treffes etter hvert i den store semifinalen der Noosa overkjørte vinnere med 11 poeng.",
                "Det ble rapportert mørke skyer ved foten av fjellet, uten tilknytning til vulkansk aktivitet.",
                "For å komme tilbake til sitt tidligere energinivå, må de kvitte seg med den ekstra energien de fikk fra lyset.",
                "Ansvaret til disse agentene er å tilby offentlige- og rettslige tjenester etter artikkel 247 i den pakistanske grunnloven."
            ]
        },
        'fin-swe': {
            'src':  [
                'Ihmisen käsi on lyhyempi kuin jalka, ja sen sormiluut ovat suoremmat.', 
                'Kun taistelu Ranskasta oli päättynyt, Saksa alkoi valmistautua Brittein saarille hyökkäämiseen.', 
                'Lisäksi huipputuomari Evangelos Kalousis on tuomittu vankeuteen korruptiosta ja sopimattomasta käytöksestä.', 
                'Silmän rakenteita on useita erilaisia, ja niiden monimutkaisuus vaihtelee eliön tarpeiden mukaan.', 
                'Ota vakuutuksistasi kopiot ja pidä niitä sekä vakuutusyhtiösi yhteystietoja mukanasi.', 
                'Abu Ghraibin vankila Irakissa on sytytetty palamaan mellakan yhteydessä.', 
                'Artikkeli osoitti testitulosten parantuneen epätodennäköiseltä tuntuvan nopeasti ja esitti koulun havainneen ongelmia sisäisissä tarkastuksissa mutta jättäneen reagoimatta niihin.', 
                'Mystiikka on pyrkimystä päästä yhteyteen korkeimman todellisuuden, jumaluuden, hengellisen todellisuuden tai Jumalan kanssa, samastua tähän tai saavuttaa tietoisuus tästä.'
            ],
            'trg': [
                'Den mänskliga handen är kortare än foten, med rakare falanger.', 
                'När slaget om Frankrike var över, började Tyskland förbereda invasionen av ön Storbritannien.', 
                'Dessutom fängslas toppdomaren Evangelos Kalousis efter att ha befunnits skyldig till korruption och omoraliskt beteende.', 
                'Ögon är konstruerade på många olika sätt, med olika komplexitet beroende på vad organismen kräver.', 
                'Gör kopior av din policy och ditt försäkringsbolags kontaktuppgifter och bär dem med dig.', 
                'Iraks Abu Ghraib-fängelse började brinna under ett upplopp.', 
                'Rapporten visade att provresultaten hade förbättrats osannolikt snabbt, och gjorde gällande att skolan hade upptäckt problem internt, utan att göra något åt dem.', 
                'Mysticism är strävan efter gemenskap med, identitet med, eller medveten medvetenhet om en högsta verklighet, gudomlighet, andlig sanning, eller Gud.'
            ]
        },
    }
}

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--min_new_tokens', default=10, type=int)
    ap.add_argument('--max_new_tokens', default=2048, type=int)
    ap.add_argument('--num_return_sequences', default=1, type=int)
    ap.add_argument('--temperature', default=1.0, type=float)
    ap.add_argument('--memory-usage', action='store_true')
    ap.add_argument('--show-devices', action='store_true')    
    ap.add_argument('--dtype', default='bf16')
    ap.add_argument('--device-map', default='auto')
    ap.add_argument('--trust-remote-code', default=None, action='store_true')
    ap.add_argument('--transformers_cache',type=str, default="/scratch/project_462000319/transformers_cache")
    ap.add_argument('--model', type=str)
    ap.add_argument('--tokenizer', type=str)
    ap.add_argument('--output_file', type=str, default=None)
    ap.add_argument('--filepath', type=str, default=None)
    ap.add_argument('--skip_lines', type=int, default=None)
    ap.add_argument('--last_line', type=int, default=None)
    ap.add_argument('--src_lang', type=str, default='eng', help='eng, fin, swe, dan, isl, nor')
    ap.add_argument('--trg_lang', type=str, default='fin', help='eng, fin, swe, dan, isl, nor')
    ap.add_argument('--icl_data', type=str, default='flores-101', help='tatoeba or flores-101')
    ap.add_argument('--format_type', type=str, default='user_assistant', help='equals, user_assistant')
    ap.add_argument('--num_examples', type=int, default=5)
    ap.add_argument('--flores_path', type=str, default="/scratch/project_462000444/finetuning_data/FLORES-200", help="path to FLORES-200 dev sents")
    ap.add_argument(
        "--translate_roles",
        type=str,
        nargs="+",
        default=None,
        help="SFT roles to translate",
    )
    return ap

def detect_language(sent: str):
    # remove \n from sentences because fasttext processes by line
    sent = sent.replace("\n", " ") 
    pred = LID_MODEL.predict(sent)
    # get top language
    lang = pred[0][0].split("__")[-1] 
    # get prob of top language
    prob = pred[1][0]
    return lang, prob

def generate(prompt, model, tokenizer, args, end_token="END", skip_special_tokens=False):
    # print("PROMPT:")
    # print(prompt)
    eos_token_id = tokenizer.eos_token_id
    end_tokens = ["<|user|>", "<|assistant|>", "<|im_start|>", "<|im_end|>"]
    if end_token:
        end_tokens.append(end_token)
    eos_token_id = [eos_token_id]
    for tok in end_tokens:
        eos_token_id.append(tokenizer.encode(tok)[0])
    # print("temperature:", args.temperature)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    output = model.generate(
        input_ids,
        eos_token_id=eos_token_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    result =  tokenizer.decode(output[0], skip_special_tokens=skip_special_tokens)
    if result.endswith(end_token):
        result = result[:-len(end_token)]
    if result.endswith(tokenizer.eos_token):
        result = result[:-len(tokenizer.eos_token)]
    # result includes the prompt, remove the prompt from the output
    result = result.replace(prompt, '', 1).strip()
    if "\n" in result:
        result = result.split("\n")
        result = result[0]
    # print("RESULT:")
    # print(result)
    return result

def translate_content(content, model, tokenizer, args, trg_lang=None, remove_periods=False):
    # print("--- translate_content ---")
    paragraphs_double_newline = content.split("\n\n")
    translated_content = []
    is_code = False
    for paragraph_double in paragraphs_double_newline:
        translated_paragraphs = []
        if len(paragraph_double) > 0:
            paragraphs_single_newline = paragraph_double.split("\n")
            # print("paragraphs_single_newline:", len(paragraphs_single_newline))
            # if len(paragraphs_single_newline) > 30 and "```" not in paragraph_double:
            #     print("Cannot translate current content. Skipping.")
            #     valid_entry = False
            #     break
            # else:
            for paragraph in paragraphs_single_newline:
                if len(paragraph) > 0:
                    if "```" in paragraph and is_code is False:
                        is_code = True
                    elif "```" in paragraph and is_code is True:
                        is_code = False
                    if is_code is True:
                        if "```" in paragraph:
                            paragraph = "\n" + paragraph
                        result = paragraph
                    else:  
                        if paragraph == "```":
                            result = paragraph
                        else:
                            if "opus" in args.model:
                                input_ids = tokenizer.encode(paragraph, return_tensors="pt").to('cuda')
                                output = model.generate(input_ids=input_ids,
                                            eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            max_new_tokens=args.max_new_tokens
                                            )
                                result = tokenizer.decode(output[0], skip_special_tokens=True) 
                            elif "nllb" in args.model:
                                input_ids = tokenizer.encode(paragraph, return_tensors="pt", padding=True).to('cuda')
                                output = model.generate(
                                                input_ids=input_ids,
                                                eos_token_id=tokenizer.eos_token_id,
                                                pad_token_id=tokenizer.pad_token_id,
                                                max_new_tokens=args.max_new_tokens,
                                                forced_bos_token_id=tokenizer.lang_code_to_id[trg_lang])
                                result = tokenizer.decode(output[0], skip_special_tokens=True)
                            elif "m2m" in args.model:
                                tokenizer.src_lang = "en"
                                input_ids = tokenizer.encode(paragraph, return_tensors="pt").to('cuda')
                                output = model.generate(input_ids=input_ids,
                                                eos_token_id=tokenizer.eos_token_id,
                                                pad_token_id=tokenizer.pad_token_id,
                                                max_new_tokens=args.max_new_tokens,
                                                forced_bos_token_id=tokenizer.get_lang_id(trg_lang))
                                result = tokenizer.decode(output[0], skip_special_tokens=True)
                            else:
                                if len(paragraph.split()) < 20:
                                    icl_data = "tatoeba"
                                else:
                                    icl_data = "flores-101"
                                if args.trg_lang == "fin":
                                    src_sents = ICL_EXAMPLES[icl_data]['eng-fin']['src']
                                    trg_sents = ICL_EXAMPLES[icl_data]['eng-fin']['trg']
                                elif args.trg_lang == "swe":
                                    src_sents = ICL_EXAMPLES[icl_data]['eng-swe']['src']
                                    trg_sents = ICL_EXAMPLES[icl_data]['eng-swe']['trg']
                                elif args.trg_lang == "dan":
                                    src_sents = ICL_EXAMPLES[icl_data]['eng-dan']['src']
                                    trg_sents = ICL_EXAMPLES[icl_data]['eng-dan']['trg']
                                elif args.trg_lang == "isl":
                                    src_sents = ICL_EXAMPLES[icl_data]['eng-isl']['src']
                                    trg_sents = ICL_EXAMPLES[icl_data]['eng-isl']['trg']
                                elif args.trg_lang == "nor":
                                    src_sents = ICL_EXAMPLES[icl_data]['eng-nor']['src']
                                    trg_sents = ICL_EXAMPLES[icl_data]['eng-nor']['trg']
                                else:
                                    src_lang = "eng"
                                    flores_src_sentences = open(os.path.join(args.flores_path, src_lang+"-dev.txt")).readlines()
                                    flores_trg_sentences = open(os.path.join(args.flores_path, trg_lang+"-dev.txt")).readlines()
                                    src_sents = [flores_src_sentences[sent_index].strip() for sent_index in FLORES_SENT_INDICES]
                                    trg_sents = [flores_trg_sentences[sent_index].strip() for sent_index in FLORES_SENT_INDICES]
                                src_sents = src_sents[:args.num_examples]
                                trg_sents = trg_sents[:args.num_examples]
                                if remove_periods is True:
                                    for sent_index in range(len(src_sents)):
                                        src_sents[sent_index] = src_sents[sent_index].replace(".", "")
                                        trg_sents[sent_index] = trg_sents[sent_index].replace(".", "")
                                if args.format_type == "user_assistant":
                                    prompt = format_prompt_user_assistant(paragraph, src_sents, trg_sents)
                                    if "viking" in args.model:
                                        result = generate(prompt, model, tokenizer, args, end_token="END", skip_special_tokens=True)
                                    else:
                                        result = generate(prompt, model, tokenizer, args, end_token="END", skip_special_tokens=False)
                                else:
                                    prompt = format_prompt_src_equals_trg(paragraph, src_sents, trg_sents, template=SRC_EQUALS_TRG_TEMPLATE, end_token="END")
                                    result = generate(prompt, model, tokenizer, args, end_token="END", skip_special_tokens=False)
                        # print("\nRESULT:\n", result)
                    translated_paragraphs.append(result)
        translated_paragraphs = "\n\n".join(translated_paragraphs)
        translated_content.append(translated_paragraphs)
    translated_content = "\n\n".join(translated_content)
    return translated_content

def report_memory_usage(message, out=sys.stderr):
    print(f'max memory allocation {message}:', file=out)
    total = 0
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.max_memory_allocated(i)
        print(f'  cuda:{i}: {mem/2**30:.1f}G', file=out)
        total += mem
    print(f'  TOTAL: {total/2**30:.1f}G', file=out)


@timed
def translate_sft_dataset(model, tokenizer, args, trg_lang=None):
    print("--- translate_sft_dataset ---")
    print("Source filepath:", args.filepath)
    print("Output filepath:", args.output_file)
    print("trg_lang:", trg_lang)
    print("skip_lines:", args.skip_lines)
    print("translate_roles:", args.translate_roles)
    if "nllb" in args.model and trg_lang is not None:
        trg_lang = NLLB_LANG_MAP[trg_lang]
    data = [json.loads(line) for line in open(args.filepath)]
    for index, line in enumerate(data): 
        print("--"*10, "Sample", index+1, "--"*10)
        if (args.skip_lines is None or index >= args.skip_lines) and (args.last_line is None or index < args.last_line):
            translated_messages = {'messages': [] }
            messages = line['messages']
            valid_entry = True
            for message in messages: 
                role = message['role']
                content = message['content']
                if role in args.translate_roles:
                    print("\nROLE:", role)
                    print("\nSRC:\n", content)
                    translated_content = translate_content(content, model, tokenizer, args, trg_lang=trg_lang, remove_periods=False)
                    print("\nPRED:\n", translated_content)
                    translated_messages['messages'].append({"role": role, 
                                                        "content": translated_content,
                                                        "orig_content": content})
            if valid_entry is True:
                with open(args.output_file, "a") as f:
                    f.write(
                        json.dumps(
                            translated_messages,
                            ensure_ascii=False
                            )
                        + "\n"
                    )


def check_needs_retranslation(content, trg_lang, model, tokenizer, args):
    # check if content is already translated
    lang, prob = detect_language(content)
    if lang != LANG_CODE_MAP[trg_lang] or (lang == LANG_CODE_MAP[trg_lang] and prob < LANG_THRESHOLD):
        print(f"CONTENT:\n{content}")
        print(f"\nLANG: {lang} PROB: {prob}")
        print("CONTENT NEEDS RETRANSLATION")
        # args.icl_data = "tatoeba"
        # args.num_examples = 5
        if content[-1] != ".": 
            translated_output = translate_content(content, model, tokenizer, args, trg_lang=trg_lang, remove_periods=True)
        else:
            translated_output = translate_content(content, model, tokenizer, args, trg_lang=trg_lang, remove_periods=False)
        # check language of translated output
        lang, prob = detect_language(translated_output)
        if lang != LANG_CODE_MAP[trg_lang] or (lang == LANG_CODE_MAP[trg_lang] and prob < LANG_THRESHOLD):
            print(f"TRANSLATION ERROR\nlang: {lang} prob: {prob}\n")
            print("\nINPUT:\n", content)
            print("\nOUTPUT:\n", translated_output)
            print("--"*20)
        else:
            print(f"TRANSLATION SUCCESS\nlang: {lang} prob: {prob}\n")
            print("\nINPUT:\n", content)
            print("\nOUTPUT:\n", translated_output)
            print("--"*20)
    else:
        print("Content already translated. Skipping.")
        print("--"*20)
        translated_output = content
    return translated_output

@timed
def translate_dpo_dataset(model, tokenizer, args, trg_lang=None):
    print("--- translate_dpo_dataset ---")
    print("Source filepath:", args.filepath)
    print("Output filepath:", args.output_file)
    print("trg_lang:", trg_lang)
    if "nllb" in args.model and trg_lang is not None:
        trg_lang = NLLB_LANG_MAP[trg_lang]
    data = [json.loads(line) for line in open(args.filepath)]
    for index, entry in enumerate(data): 
        print("--"*10, "Sample", index+1, "--"*10)
        if (args.skip_lines is None or index >= args.skip_lines) and (args.last_line is None or index < args.last_line):
            translated_entry = {'prompt': [],
                                'chosen': '', 
                                'orig_chosen': '',
                                'rejected': '',
                                'orig_rejected': ''
                                }
            valid_entry = True
            for response_type in ['prompt', 'chosen', 'rejected']: 
                if response_type == 'prompt':
                    for message in entry[response_type]:
                        role = message['role']
                        content = message['content']
                        translated_output = check_needs_retranslation(content, trg_lang, model, tokenizer, args)
                        if len(translated_output) > 0:
                            translated_entry['prompt'].append({"role": role,
                                                            "content": translated_output,
                                                            "orig_content": content})
                        else:
                            valid_entry = False
                            break
                else:
                    content = entry[response_type]
                    translated_output = check_needs_retranslation(content, trg_lang, model, tokenizer, args)
                    if len(translated_output) > 0:
                        translated_entry[response_type] = translated_output
                        translated_entry[f"orig_{response_type}"] = content
                    else:
                        valid_entry = False
                        break
            if valid_entry is True:
                with open(args.output_file, "a") as f:
                    f.write(
                        json.dumps(
                            translated_entry,
                            ensure_ascii=False
                            )
                        + "\n"
                    )



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

@timed
def load_model(args):
    print("Loading model:", args.model)
    if "opus" in args.model:
        model = MarianMTModel.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
            # cache_dir=args.transformers_cache,
        )        
        model.to('cuda')
    elif "nllb" in args.model or "m2m" in args.model:
        model = M2M100ForConditionalGeneration.from_pretrained(
            args.model,
            # device_map=args.device_map,
            # torch_dtype=DTYPE_MAP[args.dtype],
            # trust_remote_code=args.trust_remote_code,
            # cache_dir=args.transformers_cache
        ) 
        model.to('cuda')  
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
        )
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
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_model(args)
    if args.memory_usage:
        report_memory_usage('after model load')
    if "DPO" in args.filepath:
        translate_dpo_dataset(model, tokenizer, args, trg_lang=args.trg_lang)
    else:
        translate_sft_dataset(model, tokenizer, args, trg_lang=args.trg_lang)
    print("Model:", args.model)
    print("--- Done translating. Outputs saved to", args.output_file, "---")
    if args.memory_usage:
        report_memory_usage('after translation')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
