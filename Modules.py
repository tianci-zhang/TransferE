#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:35:59 2019

@author: ztc
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import copy,math
import numpy as np


class Predict(nn.Module):
    '''define stantard linear + softmax generation step for prediction'''
    def __init__(self,Entity_Embedding,Relation_Embedding,
                 d_model,num_entity_class,num_relation_class,dropout):
        super(Predict,self).__init__()
        self.linear_head = nn.Linear(d_model,num_entity_class,bias=True)
        self.linear_head.weight = Entity_Embedding.weight
        self.linear_tail = nn.Linear(d_model,num_entity_class,bias=True)
        self.linear_tail.weight = Entity_Embedding.weight
        self.linear_relation = nn.Linear(d_model,num_relation_class,bias=True)
        self.linear_relation.weight = Relation_Embedding.weight
        # here the weights of linear layer is entity/relation embedding matrices
        self.dropout = nn.Dropout(dropout)
    def forward(self,output,task,mask_index):
        '''
        getting logits according to different task
        size of input: [B,L,3,dim]
        size of output: [B,num_mask,num_class]
        '''
        if task == "head-pre":
            output_h = output[:,mask_index,0,:] #[B,num_mask,dim]
            logits = self.dropout(output_h)
            logits = self.linear_head(logits)
        elif task == "tail-pre":
            output_t = output[:,mask_index,1,:] #[B,num_mask,dim]
            logits = self.dropout(output_t)
            logits = self.linear_tail(logits)
        elif task == "relation-pre":
            output_r = output[:,mask_index,2,:] #[B,num_mask,dim]
            logits = self.dropout(output_r)
            logits = self.linear_relation(logits)
        logits = F.log_softmax(logits,dim=-1)#[B,num_mask,dim]
        return logits


def clones(module,N):
    '''produce N identical layers'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])




class LabelSmoothing(nn.Module):
    '''implement label smoothing'''
    def __init__(self, entity_size, relation_size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing  # if i=y
        self.smoothing = smoothing
        self.entity_size = entity_size  
        self.relation_size = relation_size
        self.ture_dist = None
        self.size = 0

    def forward(self, x, target, task):
        # x[B,num_mask,num_class], target:[B,num_mask] 
        batch_size, num_mask, num_class = x.size()
        target = target.reshape(-1)
        if task == "head-pre" or task == "tail-pre":
            self.size = self.entity_size
        elif task == "relation-pre":
            self.size = self.relation_size
        assert num_class == self.size
        true_dist = x.data.clone()
        true_dist = true_dist.reshape(-1,num_class)
        true_dist.fill_(self.smoothing / (self.size - 1))  # when i!=y
        true_dist.scatter_(1, target.data.unsqueeze(-1), self.confidence)  # when i=y
        true_dist = true_dist.reshape(batch_size, num_mask, num_class)
        self.true_dist = Variable(true_dist, requires_grad=False)
        return self.criterion(x, self.true_dist)/(batch_size*num_mask)
        # after this, we can get loss, for each sample in each position


class NoamOpt:
    '''optim wrapper that implementss rate'''
    def __init__(self,model_size,factor,warmup,optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    def step(self):
        '''update parameters and rate'''
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr']=rate
            # print("step:", self._step)
            # print("the learning rate is:", p['lr'])
        self._rate = rate
        self.optimizer.step()
    def rate(self,step=None):
        '''implement 'lrate' above'''
        if step is None:
            step = self._step
        return self.factor*(self.model_size**(-0.5)*min(step**(-0.5), step*self.warmup**(-1.5)))