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
state_dict = torch.load('predict_class.pth')
predict_class.load_state_dict(state_dict)
predict_class.eval()
# optimizer = torch.optim.Adam(params=predict_class.parameters(),
#                               lr=0.0001)

train_data_size = questions_tags.shape[0] //6 *5 # first 5/6 of entire dataset
#valid_data_size = questions_tags.shape[0] //6 *1 
test_data_size = (questions_tags.shape[0] - train_data_size 
#                  - valid_data_size
                    )  # last 1/6 of entire dataset for test
#n_epochs = 300
#batch_size = 1 # currently unbatchable due to various #pos_in (length of str)

#losses = []
error_rates = []
for n in range(test_data_size):
    index_question = train_data_size + n # only in case no validation
    input_str = input_strings.iloc[index_question]
    num_pos_in = len(input_str)
    inputs_en = torch.tensor(tokenize(input_str), device=DEVICE)[None, ...]
                        # shape = (#batch=1, #pos_in) 
    pos_encode_en = pos_encode(chn_s=token_size, 
                               pos_s=torch.arange(num_pos_in, device=DEVICE))
    input_tags_total = questions_tags.iloc[index_question]['Tag'] 
                        # list of tags
    input_tags_used_list = []
    for k in range(tags_max_len):
        pos_predict = k
    #    pos_predict = torch.randint(tags_max_len, (1,)).item() # int<5
        # input_tags_used_list = [0 for j in range(pos_predict)] 
        #                     # len = #pos_de, without pred
        #for i in range(pos_predict):
        #    if i<len(input_tags_total):
        #        input_tags_used_list[i] = tag_to_id[input_tags_total[i]]
        #    else:
        #        input_tags_used_list[i] = tag_to_id[""]
        input_tags_used = torch.tensor(input_tags_used_list,
                                       dtype=torch.int64,
                                       device=DEVICE)[None, ...]     
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
        predicted_class = train_evaluate(
                       inputs_en=inputs_en, 
                       inputs_de=input_tags_used,
                       targets=tag_label, 
                       pos_encode_en=pos_encode_en, 
                       pos_encode_de=pos_encode_de, 
                       en_de=predict_class, 
    #                   loss_fct=torch.nn.CrossEntropyLoss(), 
                       optimizer=None)
        input_tags_used_list.append(predicted_class.item())
    predict_class_total = [id_to_tag[i] for i in input_tags_used_list
                           if i != tag_to_id[""]] # len<=5
    wrong_tags = set(predict_class_total) ^ set(input_tags_total)
                        #+ (tags_max_len-len(input_tags_total)) * [""]))
    #error_rate = len(wrong_tags) / (tags_max_len + len(input_tags_total))
    error_rate = min(len(wrong_tags) / len(input_tags_total), 1.0)
    error_rates.append(error_rate)
    print(error_rate, input_tags_total, predict_class_total)

error_rates = torch.tensor(error_rates)
average_error_rates = torch.mean(error_rates, dim=-1)
torch.save(error_rates, 'error_rates.pth')
torch.save(error_rates, 'average_error_rates.pth')
print('average error rate:', average_error_rates.item())
    
    
        
                                       
        
        
    
           
    

