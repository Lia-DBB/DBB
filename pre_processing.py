# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:06:59 2024

@author: Lia Ahrens
"""

import torch
import numpy as np
#import torchvision.transforms.functional as T
import math
import pandas as pd
import datetime

from load_data import questions
from load_data import tags

tag_lists = tags.groupby('Id')['Tag'].apply(list) # result uses 'Id' as index
tags_max_len = tag_lists.apply(len).max() 
set_diff = set(tags['Id']) - set(questions['Id'])
# tags[tags['Id']==40115300]
questions_tags = questions.set_index('Id').join(pd.DataFrame(tag_lists)) #df
# default: leftjoin
input_strings = (questions_tags['Title'] 
                 + '\n' + questions_tags['Body']) # pandas series
#len(input_strings[index<-random]) -> pos encoding

id_to_tag = [""] + sorted(set(tags['Tag'])) # zahl -> Tag, id=Index
tag_to_id = {tag: i for i, tag in enumerate(id_to_tag)} # dictionary
# tag_to_id['python']
dim_tag = len(id_to_tag) # num of tags =dim_ouput

def tokenize(string):
    return [ord(x) for x in string]
# ord: zeichen zu Zahl in [0, 256)

# wie behandeln wir die Fragen
# wie machen wir die Klassifizierung, damit wir mehrere Klassen als Output bekommen


# Ansatz: NN für Sequence classification (bidirectional LSTM, 
# Transformer: 1)En-De: input: Frage & Überschrift, output: sequenz von tags 
#              2) GPT-var.: input: Frage & Überschrift + ein extra Token=pos
# Frage eingeben -> mehrere Tokens generieren, iterative bis End-Token generiert wird)
        # trainieren: Fragen & Überschrift -> Tags als Label -> cat + End-Token
                      # wie Autoencoder mit kausale Maske (GPT: 1-Schritt-Vorhersage-Modell)
                      # in Loss: nur Antwort-Teil
# sparlsg mit Chat-GPT: pre-/Post-processing

# wie behandeln Fragen: tokeniser(Text->tokens): input dafür: 
    # nur Überschrift
    # Überschrift + 2xleerzeilen als Trennzeichen + Frage (alle cat)
# tokeniser: LookupTab: Buchstaben=Character
                        #(-> weniger verschiedene Tokens, dafür längere Seq.)
                         #/Wörter
                        #(-> mehr Kombis, dafür weniger pro Satz)
                        # Mittelweg bei GPT: Buchstaben-Bündel
                            # häufige Wörter/Silben & Buchstab. bei seltenen
                #->Indizes
                # ->embedding layer (-> 2048 dim)

# Labels:
# für LSTM: multi-hot
# bei Transformer En-De: 0-1-Seq. (binary classifier output)
# bei GPT: input=output=label  = output von tokeniser
            # bzw. tokeniser modifizieren, s.d. tags extra Tokens bekommen
            # (Überschrift + Trennzeichen + Frage + :/... + Tags mit ,/leer)
            
# im Falle ohne Training: wie bei GPT-Fall, mehr Trennzeichen & Prefix
            # Prefix: ("please tag the following question: \n")
            # Trennzeichen: "use one or more of the following tags: tag, tag..."
        # bei GPT iengeben: prefix + Frage + Trennzeichen (als ein String)
                            # -> Tokeniser -> GPT
        # bei Datenbank: Zugang zu API = application programming interface
                 # -> Frage-String an API senden -> Antwort zurückbekommen
                 
# decoder: embedding shape = #batch x #pos, Werte=(indices für tag<-LT)
# From: LT<->LT für Tags, Tokeniser(Str from df[..+\n+.. -> Liste ints)
# torch.embed