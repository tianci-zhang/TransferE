#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:43:08 2019

@author: ztc
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F
import math
import numpy as np
from Modules import clones


class MultiHeadedAttention(nn.Module):
    '''take in model size and number of heads'''
    def __init__(self,h,d_model,d_per,dropout):
        super(MultiHeadedAttention,self).__init__()
        self.d_per = d_per
        self.h = h
        self.linears = clones(nn.Linear(d_model, int(self.d_per*self.h)), 3)
        self.linear = nn.Linear(int(self.d_per*self.h) ,d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def forward(self,x,mask=None):
        '''
        calculate query1,key1,query2,key2,value
        getting the attention representation of heads, tails and relations respectively
        '''
        #x[B,L,3,dim]
        H = x[:,:,0,:]
        T = x[:,:,1,:]
        R = x[:,:,2,:] #[B,L,dim]
        H_new = self.Multi_Head_Attn(T, T, R, R, H)
        T_new = self.Multi_Head_Attn(H, H, R, R, T)
        # H_new = H.unsqueeze(-2)
        # T_new = T.unsqueeze(-2)
        R_new = self.Multi_Head_Attn(H, H, T, T, R) #[B,L,1,dim]
        return torch.cat((H_new,T_new,R_new),-2)
        
    def Multi_Head_Attn(self, Q1, K1, Q2, K2, V):
        # here Q1 or Q2 is equal, hence them sharing parameters.
        # K1 and K2 are similar.
        batch_size = Q1.size(0)
        Q1 = self.linears[0](Q1) #linear projection, getting [B,L,d_per*h]
        query1 = Q1.view(batch_size,-1,self.h,self.d_per).transpose(1,2) #[B,h,L,d_per]
        K1 = self.linears[1](K1)
        key1 = K1.view(batch_size,-1,self.h,self.d_per).transpose(1,2) #[B,h,L,d_per]
        Q2 = self.linears[0](Q2)
        query2 = Q2.view(batch_size,-1,self.h,self.d_per).transpose(1,2) #[B,h,L,d_per]
        K2 = self.linears[1](K2)
        key2 = K2.view(batch_size,-1,self.h,self.d_per).transpose(1,2) #[B,h,L,d_per]
        V = self.linears[2](V)
        value = V.view(batch_size,-1,self.h,self.d_per).transpose(1,2)  #[B,h,L,d_per]
        # attention
        x_return = attention(query1, key1, query2, key2, value, mask=None, dropout=self.dropout)
        # x: [B,h,L,d_per]
        x_return = x_return.transpose(1,2).contiguous().view(batch_size,-1,
                       int(self.h*self.d_per))
        x_return = self.linear(x_return).unsqueeze(-2)
        return x_return #[B,L,1,d_model]




class PositionwiseFeedForward(nn.Module):
    '''feed forward layer of encoder block'''
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))




class SublayerConnection(nn.Module):
    '''a residual connection followed by a layer norm'''
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,sublayer):
        '''apply residual connection to any sublayer with 
        the same size'''
        return x + self.dropout(sublayer(self.norm(x)))




def attention(query1, key1, query2, key2, value, mask=None, dropout=None):
    ''' 
    size of each input: [B,h,L,d_per] 
    size of output: [B,h,L,d_per]
    '''
    d_per = query1.size(-1) #[B,h,L,d_per]
    score_1 = torch.matmul(query1,key1.transpose(-2,-1))/math.sqrt(d_per) #[B,h,L,L]
    score_2 = torch.matmul(query2,key2.transpose(-2,-1))/math.sqrt(d_per) #[B,h,L,L]
    scores = torch.div(torch.add(score_1, score_2), 2) # F_1
    if mask is not None:
        scores = scores.masked_fill(mask==0,-np.inf)
    p_attn = F.softmax(scores,dim=-1) #[B,h,L,L]
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn,value) 









