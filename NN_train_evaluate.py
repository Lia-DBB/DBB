# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:02:42 2024

@author: Lia Ahrens
"""

import torch
# import numpy as np
# import math
# import pandas as pd
# import datetime

# from load_data import questions
# from load_data import tags

# from pre_processing import questions_tags # df with questions + tags(grouped)
# from pre_processing import input_strings # pandas series {questions+\n+tags}
# from pre_processing import id_to_tag # LUT index -> tag
# from pre_processing import tag_to_id # LUT tag -> index
# from pre_processing import tokenize # str -> int in [0, 256)


class Transf_Predict_Class(torch.nn.Module):
    def __init__(self, dim_dict, # size dictionary for str
                      tag_dim, # total num tags
                      embed_dim=512, # size token
                      num_heads=8, 
                      num_encoder_layers=2,
                      num_decoder_layers=2,
                      #num_encoder_layers=6,
                      #num_decoder_layers=6,
                      ):
        super().__init__()
        self.dim_dict = dim_dict
        self.tag_dim = tag_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        self.embed_en = torch.nn.Embedding(
            num_embeddings=self.dim_dict, 
            embedding_dim=self.embed_dim)        
        self.embed_de = torch.nn.Embedding(
            num_embeddings=self.tag_dim, 
            embedding_dim=self.embed_dim)        
        self.transformer_en_de = torch.nn.Transformer(
            d_model=self.embed_dim, 
            nhead=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            batch_first=True) # (batch, pos, chn)
        self.out_layer = torch.nn.Linear(in_features=self.embed_dim, 
                                         out_features=self.tag_dim)
    def forward(self, x_en, pos_encode_en, y_de, pos_encode_de):
        """
        x_en: torch tensor (batch, pos_in), tokenized str = seq of int<256
        pos_encode_en: (chn, pos_in) ! output of pos_encode for pos_in
        y_de: torch tensor (batch, pos_out), seq of tags = seq of int<#tags
        pos_encode_de: (chn, pos_out_with_pred), 
                            #pos_out ! with pos to be predicted 
                        ! output of pos encode for pos_out_with_pred
        return: torch tensor, shape = (#batch, tag_dim), 
                prediction for last pos of out_seq
        """
        x_embed = self.embed_en(x_en) # (batch, pos_in, chn)
        x_en_input = x_embed + pos_encode_en
        y_embed = self.embed_de(y_de) # (batch, pos_out, chn)
        y_embed = torch.cat([y_embed, torch.zeros(
                                (*y_embed.shape[:-2], 1, y_embed.shape[-1]),
                                device=y_embed.device)],
                            dim=-2) # add pos to be predicted
        y_de_input = y_embed + pos_encode_de 
                    # shape = (#batch, #pos_out_with_pred, embed_dim)
        y_de_output = self.transformer_en_de(x_en_input, y_de_input)
        y_predict = y_de_output[:, -1, :] # shape =(#batch, embed_dim)
        y_predict_tag = self.out_layer(y_predict) # shape =(#batch, tag_dim)
        return y_predict_tag 
        
def pos_encode(chn_s, pos_s, delta_s=1,
                 ):
    """
    positional encoding for d-many chn and s=[seq of pos]: 
        {exp(i*alpha*(beta**k)*s)}_{k<d/2}
    chn_s: total number of channels for encoding pos_s
    pos_s: torch tensor, shape=(#pos,), seq of pos
    delta_s: unit distance along s-axis, for NLP: delta_s=1
    return: torch tensor (pos, chn) 
    """        
    alpha = 1 / delta_s # alpha=f_max_x=1 in NLP
    beta = 10**(-4 / (chn_s//2))
    embed_s = torch.cat((torch.stack(
                            [torch.cos(alpha * beta**k * pos_s)
                             for k in range(chn_s//2)], dim=0),
                         torch.stack(
                             [torch.sin(alpha * beta**k * pos_s)
                              for k in range(chn_s//2)], dim=0)),
                        dim=0) # (chn, pos)
    embed_s = torch.movedim(embed_s, -2, -1) #(pos, chn)
    return embed_s

def train_evaluate(inputs_en, 
                   inputs_de,
                   targets, 
                   pos_encode_en, pos_encode_de, 
                   en_de, 
                   loss_fct=torch.nn.CrossEntropyLoss(), 
                   optimizer=None):
    """
    inputs_en: batch of tokenised str (title + \n + body), 
                                    with values int<256 
               shape = (#batch, #pos_in)
    inputs_de: batch of seq of tags (known) = seq of int<#tags
    targets: batch of tag for last pos of transformer output
            shape = (#batch, ), tag = int < tag_dim=#tags
    pos_encode_en: (chn, pos_in) ! output of pos_encode for pos_in
    pos_encode_de: (chn, pos_out_with_pred), 
                        #pos_out ! with pos to be predicted 
                    ! output of pos encode for pos_out_with_pred
    en_de: NN_predicter with transformer & embedding & outlayer
    loss_fct: loss for classification, default: cross entropy
            ! to be refined: weight ! proportional to 1/(# train examples)
                                                      for each class
    optimizer: train if not None, else evaluate
    """
    with torch.set_grad_enabled(optimizer is not None):
        predict = en_de(x_en=inputs_en,
                        pos_encode_en=pos_encode_en, 
                        y_de=inputs_de, 
                        pos_encode_de=pos_encode_de)    
        #shape =(#batch, tag_dim)   
        loss = loss_fct(predict, targets)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = (loss.item())
            print('running_losses:', running_loss)
            return running_loss
        else:
            predicted_class = torch.argmax(predict, dim=-1)
                        # shape=(#batch, ), pred_tag = int < tag_dim=#tags
#             evaluation = predict_class[predict_class==targets]
# !check # not quite right yet
#             num_right = torch.sum(evaluation, dim=0).item()
            return predicted_class

        


