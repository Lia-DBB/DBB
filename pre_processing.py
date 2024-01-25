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

