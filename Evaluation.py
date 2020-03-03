# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:54:33 2019

@author: ztc
"""
import torch
import numpy as np


def Evaluation(label_pre,H_n,T_n,R_n,triples_set,task): 
    '''calculate evaluation according to different task '''
    if task == "head-pre":
        return Evaluation_Head(label_pre,H_n,T_n,R_n,triples_set)
    elif task == "tail-pre":
        return Evaluation_Tail(label_pre,H_n,T_n,R_n,triples_set)
    elif task == "relation-pre":
        return Evaluation_Relation(label_pre,H_n,T_n,R_n,triples_set)


def Evaluation_Head(label_pre,H_n,T_n,R_n,triples_set): 
    #label_pre:[B,num_class], entity classification
    #T,H,R:[B]    
    '''getting the ranked list of all entities, according to their probability value.'''
    _, rank_head = torch.topk(label_pre, label_pre.size(-1), dim=1) 
    rank_result_list = []
    rank_head = rank_head.numpy() #[B,num_class]
    for i in range(len(label_pre)):
        '''getting the rank of truth head for each test triple'''
        label = H_n[i]
        t_n = T_n[i]
        r_n = R_n[i]
        h_n_list = rank_head[i]
        head_rank_raw = 0
        head_rank_filter = 0
        for candidate in h_n_list:
            if candidate == label:
                break
            else:
                head_rank_raw += 1
                if (candidate,t_n,r_n) in triples_set:
                    continue
                else:
                    head_rank_filter += 1
        rank_result_list.append([head_rank_raw,head_rank_filter])
    return rank_result_list # [B,2], each row contains the raw and filter rank of truth head for the triple of this row.


def Evaluation_Tail(label_pre,H_n,T_n,R_n,triples_set): 
    #label_pre:[B,num_class], entity classification
    #T,H,R:[B]    
    '''getting the ranked list of all entities, according to their probability value.'''
    _, rank_tail = torch.topk(label_pre, label_pre.size(-1), dim=1)
    rank_result_list = []
    rank_tail = rank_tail.numpy()
    # rank the entities according to the probability
    for i in range(len(label_pre)):
    '''getting the rank of truth tail for each test triple'''
        label = T_n[i]
        h_n = H_n[i]
        r_n = R_n[i]
        t_n_list = rank_tail[i]
        tail_rank_raw = 0
        tail_rank_filter = 0
        for candidate in t_n_list:
            if candidate == label:
                break
            else:
                tail_rank_raw += 1
                if (h_n,candidate,r_n) in triples_set:
                    continue
                else:
                    tail_rank_filter += 1
        rank_result_list.append([tail_rank_raw,tail_rank_filter])
    return rank_result_list # [B,2], each row contains the raw and filter rank of truth tail for the triple of this row.



def Evaluation_Relation(label_pre,H_n,T_n,R_n,triples_set): 
    #label_pre:[B,num_class], relation classification
    #T,H,R:[B]    
    '''getting the ranked list of all relations, according to their probability value.'''
    _, rank_relation = torch.topk(label_pre, label_pre.size(-1), dim=1)
    rank_result_list = []
    rank_relation = rank_relation.numpy()
    # rank the entities according to the probability
    for i in range(len(label_pre)):
        '''getting the rank of truth relation for each test triple'''
        label = R_n[i]
        h_n = H_n[i]
        t_n = T_n[i]
        r_n_list = rank_relation[i]
        relation_rank_raw = 0
        relation_rank_filter = 0
        for candidate in r_n_list:
            if candidate == label:
                break
            else:
                relation_rank_raw += 1
                if (h_n,t_n,candidate) in triples_set:
                    continue
                else:
                    relation_rank_filter += 1
        rank_result_list.append([relation_rank_raw,relation_rank_filter])
    return rank_result_list # [B,2], each row contains the raw and filter rank of truth relation for the triple of this row.




def Print_mean_rank(rank_result_list, task):
    '''calculating the evaluation results according to different metrics '''
    if task == "relation-pre":
        mr_raw = 0
        mr_filter = 0
        hits_1_raw = 0
        hits_1_filter = 0
        for rank_result in rank_result_list:
            rank_raw, rank_filter = rank_result   
            mr_raw += rank_raw
            mr_filter += rank_filter
            if rank_raw < 1:
                hits_1_raw += 1
            if rank_filter < 1:
                hits_1_filter += 1
        mr_raw /= len(rank_result_list)
        mr_filter /= len(rank_result_list)
        hits_1_raw /= len(rank_result_list)
        hits_1_filter /= len(rank_result_list)
        print("RawMeanRank:{:.3f}, RawHits@1: {:.3f}, FilterMeanRank:{:.3f}, FilterHits@1: {:.3f}".format(
                mr_raw, hits_1_raw, mr_filter, hits_1_filter))
        return [round(mr_raw, 3), round(hits_1_raw, 3), round(mr_filter, 3), round(hits_1_filter, 3)]
    else:
        mean_rank_filter = 0
        hits_1_filter = 0
        hits_3_filter = 0
        hits_10_filter = 0
        mrr_filter = 0
        for rank_result in rank_result_list:
            _, rank_filter = rank_result  #actually here _ is rank_raw
            mean_rank_filter += rank_filter
            mrr_filter += 1/(1+rank_filter)
            if rank_filter < 1:
                hits_1_filter += 1
            if rank_filter < 3:
                hits_3_filter += 1
            if rank_filter < 10:
                hits_10_filter += 1
        mean_rank_filter /= len(rank_result_list)
        hits_1_filter /= len(rank_result_list)
        hits_3_filter /= len(rank_result_list)
        hits_10_filter /= len(rank_result_list)
        mrr_filter /= len(rank_result_list)
        print('MeanRank: {:.3f}, MeanReciprocalRank:{:.3f}, Hits@{}: {:.3f}, Hits@{}: {:.3f}, Hits@{}: {:.3f}'.format(
                mean_rank_filter, mrr_filter, 1, hits_1_filter, 3, hits_3_filter, 10, hits_10_filter))
        return [round(mean_rank_filter,3), round(mrr_filter,3),
        round(hits_1_filter,3),round(hits_3_filter,3), round(hits_10_filter,3)]
