# -*- coding: utf-8 -*-
# @Author: hliu.Luke
# @Date:   2020-07-31 10:00:26
# @Last Modified by:   hliu.Luke
# @Last Modified time: 2020-08-06 13:17:39
import torch
import math
import torch.nn as nn


class BPR(nn.Module):
    def __init__(self, num_users, num_items, rate_mean, embed_dim, rate_max = 5, rate_min = 0, dropout = 0.0):
        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number embedding dimension;
        rate_mean: mean of rate;
        rate_max : max  of rate;
        dropout: we tried dropout method but it did'nt work
        """     

        self.num_users = num_users
        self.num_items = num_items
        self.rate_mean = rate_mean
        self.embed_dim = embed_dim

        self.rate_max = rate_max
        self.rate_min = rate_min

        self.embed_user = nn.Embedding(self.num_users,  self.embed_dim)
        self.embed_item = nn.Embedding(self.num_items,  self.embed_dim)

        self.bias_user = nn.Embedding(self.num_users, 1)
        self.bias_item = nn.Embedding(self.num_items, 1)

        nn.init.xavier_uniform_(self.embed_user.weight)
        nn.init.xavier_uniform_(self.embed_item.weight)


        nn.init.zeros_(self.bias_user.weight)
        nn.init.zeros_(self.bias_item.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, user, item):

        embed_u = self.embed_user(user)
        embed_v = self.embed_item(item)
        bias_u  = self.bias_user(user)
        bias_v  = self.bias_item(item)

        predicted = (embed_u * embed_v).sum(1, keepdim = True) + bias_u + bias_v +  self.rate_mean
        predicted = torch.clamp(predicted, self.rate_min, self.rate_max)
        
        return predicted.squeeze()
        