#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:05:02 2019

@author: ztc
"""
import torch
import torch.nn as nn
import time
import random
import numpy as np


from Model import TrmE
from Modules import LabelSmoothing,NoamOpt
from load_data import Data_Loader,X_MASK_Train,X_MASK_Test
from load_data import load_data
from Evaluation import Evaluation,Print_mean_rank


def train(task, model, train_data, test_data, triples_set, criterion, optimizer,
          num_epoch, triple_length, num_entity, num_relation, mask_percent, batch_size,
          batch_split, device, result_path, checkpoints_path):
    print()
    print('----------start training----------')
    print("Task:", task)
    results = [] # ["train_loss", "test_loss", "MeanRank", "MRR", "Hits@1", "Hits@3", "Hits@10"]
    batch_inner_size = batch_size // batch_split
    for epoch in range(num_epoch):
        print()
        train_loader = Data_Loader(train_data, train_data, triple_length, batch_inner_size)
        model.train()
        start = time.time()
        total_loss = 0
        num_batch = 0
        for i,batch in enumerate(train_loader):
            X = batch #[B,L,3]
            X_new,X_labels,MASK_index = X_MASK_Train(X, num_entity, num_relation, mask_percent, task)  #[B,L,3]
            X_new = X_new.to(device)
            output = model.forward(X_new)
            label_pre = model.predict(output, task, MASK_index)#[B,num_mask,num_class]
            X_labels = X_labels.to(device)  #[B,num_mask]
            losses = criterion(label_pre,X_labels,task)
            losses.backward()
            if (i+1) % batch_split == 0:
                optimizer.step()
                optimizer.optimizer.zero_grad()
            losses = losses.item() # scalar
            total_loss += losses
            num_batch = num_batch + 1
            if i % 50 == 0:
                print('[%ds] Epoch %d Step: %d  Loss:%f'
                      %(time.time()-start, epoch, i, losses))
        train_loss = total_loss/num_batch
        print('Training Loss of Epoch {}: {:.5f}'.format(epoch, train_loss))
        print()
        #start test
        test_loader = Data_Loader(train_data, test_data, triple_length, batch_inner_size)
        test_loss, result = test(model, test_loader, criterion, triples_set, task, num_entity, num_relation, device)
        results.append([round(train_loss,3), round(test_loss,3)] + result)
        if epoch % 1 == 0:
            results_ = np.array(results)
            np.savetxt(result_path + "_epoch_" + str(epoch) + ".txt", results_, fmt="%s")
        if epoch % 1 == 0:
            state = {"net": model.state_dict(), "optimizer": optimizer.optimizer.state_dict(), "epoch": epoch}
            torch.save(state, checkpoints_path + "_epoch_" + str(epoch))
    results = np.array(results)
    np.savetxt(result_path + ".txt", results, fmt="%s")


def test(model, data_iter, criterion, triples_set, task, num_entity, num_relation, device):
     model.eval()
     with torch.no_grad():
        start = time.time()
        rank_result_list = dict()
        rank_result_list[task] = []
        total_loss = 0
        num_batch = 0
        for i,batch in enumerate(data_iter):
            X = batch
            X_new,X_labels,MASK_index = X_MASK_Test(X,num_entity,num_relation,task)#[B,L,3]
            #actually here MASK_index=[-1], the last triple of input sequence, only test triple.
            X_new = X_new.to(device) #[B,L,3]
            output = model.forward(X_new)
            label_pre = model.predict(output, task, MASK_index)
            # getting test loss 
            X_labels = X_labels.to(device)
            losses = criterion(label_pre,X_labels,task)
            total_loss += losses.item()
            num_batch += 1
            label_pre = label_pre.cpu() #[B,num_mask,num_class]=[B,1,num_class]
            label_pre = label_pre.reshape(-1,label_pre.size(-1))#[B,num_class]
            # different task has different number of class
            H_n = X[:,-1,0] #[B]
            T_n = X[:,-1,1] #[B]
            R_n = X[:,-1,2] #[B]
            # here only last triple is the test triple
            rank_result = Evaluation(label_pre, H_n.numpy(), T_n.numpy(), 
                                     R_n.numpy(),triples_set,task)
            rank_result_list[task] = rank_result_list[task] + rank_result
        test_loss = total_loss/num_batch
        print('----------Evaluation----------')
        print("total time: {:.3f}s.".format(time.time()-start))
        print("prediction result:")
        result = Print_mean_rank(rank_result_list[task], task)
        return test_loss, result  #[mm, mmr, hits@1, hits@3, hits@10]

def main():
    # begin to train
    task = "relation-pre"
    # ["head-pre","tail-pre","relation-pre"]
    d_model = 256
    d_ff = 512 # dimension of Feed Forward Layer
    dropout = 0.1 
    N = 6  # number of blocks
    h = 4  # number of multi-head 
    triple_length = 64 # sequence length of input 
    d_per = d_model//h # dimension of attention layer in each head 
    num_epoch = 50
    batch_size = 256
    batch_split = 1 
    data_dir = "./data/FB15k" # choose dataset to train 
    checkpoints_path = "/home1/zhangtc/trme/FB15k-try/relation-pre_2" # save the parameters during training
    result_path = "/home/zhangtc/TrmE/trme/FB15k-try/relation-pre_2" # save the results of each epoch
    label_smoothing = 0.0
    seed = 0
    warmup = 10000 # 10000
    factor = 1
    mask_percent = 0.15

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print('parameters:')
    print(data_dir)
    print("num_epoch:",num_epoch)
    print("triple_length:",triple_length)
    print("batch_size:",batch_size)
    print("batch_split:", batch_split)
    print("dropout:",dropout)
    print("warmup:", warmup)
    print("factor:", factor)
    print("label_smoothing:",label_smoothing)
    print('d_model:',d_model)
    print("d_ff:",d_ff)
    print("d_per:",d_per)
    print("mask_percent:",mask_percent)
    print("N:",N)
    print("h:",h)
    print()


    data_raw, num_entity, num_relation = load_data(data_dir)  
    '''
    data_raw["train"]: a list of train data, each row represents a triple.
    data_raw["test"]: a list of test data
    data_raw["valid"]: a list of valid data
    num_entity: number of entities in whole dataset
    num_relation: number of relations in whole dataset
    '''
    triples_set = set(data_raw["train"]+data_raw["test"]+data_raw["valid"]) # store all triples used for filterd evaluation
    train_data = data_raw["train"]
    test_data = data_raw["test"]
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(device)
    model = TrmE(num_entity,num_relation,h,N,d_model,d_per,d_ff,dropout) # define the model
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model = model.to(device)
    criterion = LabelSmoothing(entity_size=num_entity+1, relation_size=num_relation+1,
                                      padding_idx=0, smoothing=label_smoothing)
    model_opt = NoamOpt(d_model,factor,warmup,
                    torch.optim.Adam(model.parameters(),lr=0.0,
                    betas=(0.9,0.98),eps=1e-9))
    train(task, model, train_data, test_data, triples_set, criterion, model_opt,
          num_epoch, triple_length, num_entity, num_relation, mask_percent, batch_size,
          batch_split, device, result_path, checkpoints_path)

if __name__ == "__main__":
    main()



