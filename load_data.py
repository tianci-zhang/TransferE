#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 10:50:23 2019

@author: ztc
"""


import os
import pandas as pd
import numpy as np
import random
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import copy


def load_data(data_dir):
	entity_dict_file   = 'entity2id.txt'
	relation_dict_file = 'relation2id.txt'
	print('----------Loading entity dict----------')
	entity_df   = pd.read_table(os.path.join(data_dir,entity_dict_file),header=None)
	entity_dict = dict(zip(entity_df[0], entity_df[1]))
	num_entity = len(entity_dict)
	print('num of entity: {}'.format(num_entity))
	print('----------Loading relation dict----------')
	relation_df   = pd.read_table(os.path.join(data_dir,relation_dict_file),header=None)
	relation_dict = dict(zip(relation_df[0], relation_df[1]))
	num_relation = len(relation_dict)
	print('num of relation: {}'.format(num_relation))
	File = ['train','test','valid']
	all_triples = dict()  
	for file in File:
	    print(('----------Loading {} triples----------').format(file))
	    data_df = pd.read_table(os.path.join(data_dir,file+'.txt'),header=None)
	    data = list(zip([entity_dict[h] for h in data_df[0]],
                      [entity_dict[t] for t in data_df[1]],
                         [relation_dict[r] for r in data_df[2]]
                         ))
	    all_triples[file] = data
	    print('num of {} triples: {}'.format(file, len(data)))
	return all_triples, num_entity, num_relation



def Data_Loader(train_set, data, triple_length, batch_size):
    '''
    creating  input sequences of model from train set, no matter for training or test.
    shape of train_set: [N,3]
    shape of data_set: [N,triple_length,3] 
    '''
    data_set = []
    for sample in data:
        samples = random.sample(train_set,triple_length-1)
        data_set.append(samples+[sample])
    data_set = np.array(data_set)
    data_set = Variable(torch.from_numpy(data_set),requires_grad=False).long()
    data_loader = Data.DataLoader(data_set, batch_size, shuffle=False)
    return data_loader




def X_MASK_Train(X, num_entity, num_relation, mask_percent, task):
    '''Generating masked X, it is not need setting masked matrix, replacing by one-matrix'''
    MASK_Entity_index = num_entity # index of [MASK] in entity
    MASK_Relation_index = num_relation  # index of [MASK] in relation
    X_new = copy.deepcopy(X) #[B,L,3]
    triple_length = X.size(1)
    length_index_list = [i for i in range(triple_length)]
    number_mask = max(int(triple_length*mask_percent),1) # at least mask one triple for each sequence 
    if task == "head-pre":
        mask_index = random.sample(length_index_list, number_mask)
        X_new[:,mask_index,0] = MASK_Entity_index
        X_labels = X[:,mask_index,0]
    elif task == "tail-pre":
        mask_index = random.sample(length_index_list, number_mask)
        X_new[:,mask_index,1] = MASK_Entity_index
        X_labels = X[:,mask_index,1]
    elif task == "relation-pre":
        mask_index = random.sample(length_index_list, number_mask) # define where need to be masked 
        X_new[:,mask_index,2] = MASK_Relation_index # replacing the entity/relation to MASK which choosed to mask.
        X_labels = X[:,mask_index,2]
    return X_new, X_labels, mask_index
  # the new X contains MASK, which need to be predicted after training, shape: [B,L,3]
  # X_labels: truth entity/relation of MASK, shape:[B,num_mask]
  # mask_index: where masked of sequence, shape: [num_mask] 




def X_MASK_Test(X, num_entity, num_relation, task):
    '''Generating masked X, it is not need setting masked matrix, replacing by one-matrix'''
    MASK_Entity_index = num_entity # index of [MASK]
    MASK_Relation_index = num_relation  
    X_new = copy.deepcopy(X) #[B,L,3]
    if task == "head-pre":
        X_new[:,-1,0] = MASK_Entity_index
        X_labels = X[:,-1,0]
    elif task == "tail-pre":
        X_new[:,-1,1] = MASK_Entity_index
        X_labels = X[:,-1,1]
    elif task == "relation-pre":
        X_new[:,-1,2] = MASK_Relation_index
        X_labels = X[:,-1,2]
    mask_index = [-1]
    return X_new, X_labels, mask_index
    # for test sequence, only the last triple is test data. 
    # Hence we only need to mask the last entity/relation of sequence.
    # X_new:[B,L,3], X_labels:[B] 





