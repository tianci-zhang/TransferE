#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:04:55 2019

@author: ztc
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Modules import clones,Predict
from SubLayers import MultiHeadedAttention,PositionwiseFeedForward,SublayerConnection

                       
class TrmE(nn.Module):
    '''
    a standard encoder architecture 
    '''
    def __init__(self,num_entity,num_relation,
                h=8,N=6,d_model=512,d_per=64,d_ff=2048,dropout=0.1):
        super(TrmE,self).__init__()
        self.num_entity_class = num_entity+1
        self.num_relation_class = num_relation+1 # add 1 for MASK
        self.block = Block(h,d_model,d_per,d_ff,dropout,N)
        
        bound = 6./(d_model ** 0.5)
        self.Entity_Embedding = nn.Embedding(self.num_entity_class, d_model)
        self.Entity_Embedding.weight.data = torch.FloatTensor(self.num_entity_class, d_model).uniform_(-bound, bound)
        self.Entity_Embedding.weight.data = F.normalize(self.Entity_Embedding.weight.data,2,1)
        self.Relation_Embedding = nn.Embedding(self.num_relation_class,d_model)
        self.Relation_Embedding.weight.data = torch.FloatTensor(self.num_relation_class, d_model).uniform_(-bound,bound)
        self.Relation_Embedding.weight.data = F.normalize(self.Relation_Embedding.weight.data,2,1)  
        # entity and relation embedding matrices
        self.predict = Predict(self.Entity_Embedding, self.Relation_Embedding,
                 d_model,self.num_entity_class,self.num_relation_class,dropout)
        
        
    def forward(self,x):
        '''
        Take in and process masked sequences
        x/output: [B,L,3,dim]
        '''
        output = self.block(self.Embedding(x))  
        return output  


    def Embedding(self,x):
        '''
        getting the embedding of x
        shape of input x:[B,L,3]
        shape of return: [B,L,3,dim]
        '''
        Heads = x[:,:,0]#[B,L]
        Tails = x[:,:,1]
        Relations = x[:,:,2]
        heads = self.Entity_Embedding(Heads).unsqueeze(-2) #[B,L,1,dim]
        relations = self.Relation_Embedding(Relations).unsqueeze(-2) 
        tails = self.Entity_Embedding(Tails).unsqueeze(-2) 
        triples = torch.cat((heads,tails,relations),-2) #[B,L,3,dim]
        return triples #[B,L,3,dim]    
    

    

class Block(nn.Module): 
    '''TrmE block, a stack of N layers, with layer norm after each block'''
    def __init__(self,h,d_model,d_per,d_ff,dropout,N):
        super(Block,self).__init__()
        self.layers = clones(BlockLayer(h,d_model,d_per,d_ff,dropout),N)
        self.norm = nn.LayerNorm(d_model)
    def forward(self,x):
        '''pass the input and mask through each layer in ture'''
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
    
class BlockLayer(nn.Module):
    def __init__(self,h,d_model,d_per,d_ff,dropout):
        super(BlockLayer,self).__init__()
        self.self_attn = MultiHeadedAttention(h,d_model,d_per,dropout) # multi-head layer of block
        self.feed_forward = PositionwiseFeedForward(d_model,d_ff,dropout) # FFN of block
        self.sublayer = clones(SublayerConnection(d_model,dropout),2) # Add and Layer Norm
    def forward(self,x):
        x = self.sublayer[0](x, lambda x:self.self_attn(x))
        # x: [B,L,3,d_model]
        '''
        input of sublayer:x,sublayer
        x1 = norm(x), layer norm
        x2 = self_attn(x1)
        x3 = dropout(x2)+x
        x4 = norm(x3)
        x5 = feed_forward(x4)
        x6 = dropout(x5)+x3, output      
        '''
        return self.sublayer[1](x,self.feed_forward)
    







