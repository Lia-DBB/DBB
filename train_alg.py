# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 23:37:32 2024

@author: Lia Ahrens
"""

import torch
import numpy as np
# import math
import pandas as pd
import datetime

# from load_data import questions
# from load_data import tags

from pre_processing import tags_max_len # max num of tags per question
from pre_processing import questions_tags # df with questions + tags(grouped)
from pre_processing import input_strings # pandas series {titlle+\n+quest}
from pre_processing import id_to_tag # LUT index -> tag
from pre_processing import tag_to_id # LUT tag -> index
from pre_processing import tokenize # str -> int in [0, 256)

from NN_train_evaluate import Transf_Predict_Class
from NN_train_evaluate import pos_encode
from NN_train_evaluate import train_evaluate

#DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda')

dict_size = 256 # size of dictionary for str->int, depending on tokenizer
tag_size = len(id_to_tag) # total number of tags
token_size = 512
predict_class = Transf_Predict_Class(dim_dict=dict_size, 
                                     tag_dim=tag_size,
                                     embed_dim=token_size,
                              ).to(device=DEVICE)
optimizer = torch.optim.Adam(params=predict_class.parameters(),
                              #lr=0.0001)
                              lr=0.00001)

train_data_size = questions_tags.shape[0] //6 *5 # first 5/6 of entire dataset
# valid_data_size = questions_tags.shape[0] //6 *1 
# test_data_size = questions_tags.shape[0] - train_data_size - valid_data_size

n_epochs = 100000
batch_size = 1 # currently unbatchable due to various #pos_in (length of str)

losses = []

for n in range(n_epochs):
    index_question = torch.randint(train_data_size, ()).item() 
                    # simplified, only considering the case of batch_size=1 
    input_str = input_strings.iloc[index_question]
    num_pos_in = len(input_str)
    inputs_en = torch.tensor(tokenize(input_str), device=DEVICE)[None, ...]
                        # shape = (#batch=1, #pos_in) 
    pos_encode_en = pos_encode(chn_s=token_size, 
                               pos_s=torch.arange(num_pos_in, device=DEVICE))
    pos_predict = torch.randint(tags_max_len, ()).item() # int<5
    input_tags_total = questions_tags.iloc[index_question]['Tag'] 
                        # list of tags
    input_tags_used = [0 for j in range(pos_predict)] 
                        # len = #pos_de, without pred
    for i in range(pos_predict):
        if i<len(input_tags_total):
            input_tags_used[i] = tag_to_id[input_tags_total[i]]
        else:
            input_tags_used[i] = tag_to_id[""]
    input_tags_used = torch.tensor(input_tags_used,
                                   dtype=torch.int64,
                                   device=DEVICE)[
            None, ...]     
                        # shape = (#batch=1, #pos_de_without_pred) 
    if pos_predict<len(input_tags_total):
        tag_label = torch.tensor(tag_to_id[input_tags_total[pos_predict]],
                                 device=DEVICE)[None, ...]
    else:
        tag_label = torch.tensor(tag_to_id[""],
                                 device=DEVICE)[None, ...]
        # shape = (#batch=1, )
    pos_encode_de = pos_encode(chn_s=token_size, 
                               pos_s=torch.arange(pos_predict+1,
                                                  device=DEVICE))
    run_loss = train_evaluate(
                   inputs_en=inputs_en, 
                   inputs_de=input_tags_used,
                   targets=tag_label, 
                   pos_encode_en=pos_encode_en, 
                   pos_encode_de=pos_encode_de, 
                   en_de=predict_class, 
#                   loss_fct=torch.nn.CrossEntropyLoss(), 
                   optimizer=optimizer)
    losses.append(run_loss)
    torch.save(losses, 'losses.pth')
    #if (n+1)%100==0:  
    if (n+1)%1000==0:  
        timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        state_dict = predict_class.state_dict()
        torch.save(state_dict,  f'predict_class_{timestamp}.pth')
        torch.save(state_dict, 'predict_class.pth')  ###fast
                                       
        
        
    
           
    

