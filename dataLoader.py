# -*- coding: utf-8 -*-
# @Author: LH
# @Date:   2020-07-31 00:01:53
# @Last Modified by:   hliu.Luke
# @Last Modified time: 2020-08-03 15:55:16
import numpy as np 
import pandas as pd 
import scipy.sparse as sp
import torch.utils.data as data

from sklearn.model_selection import train_test_split
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

from tqdm import tqdm

class BPRData(data.Dataset):
    def __init__(self, datasets, num_users, num_items, rate_max):
        super(BPRData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """

        self.datasets  = datasets
        self.num_users = num_users
        self.num_items = num_items
        self.rate_max  = rate_max

        temp = self.datasets.tocoo()
        self.item = list(temp.col.reshape(-1))
        self.user = list(temp.row.reshape(-1))
        self.rate = list(temp.data)
        
        print('len item = {}, len user = {}, len rate = {}'.format(len(self.item), len(self.user), len(self.rate)))

    def __len__(self):
        return len(self.user)

    def __getitem__(self, idx):
        user = self.user[idx]
        item = self.item[idx]
        rate = self.rate[idx] # Ground Truth

        return user, item, rate, self.rate_max
        

def load_rating_data(path="../data/u.data", header = ['user_id', 'item_id', 'rating', 'category'], test_size = 0.1, num_negatives= 0, sep="\t"):
    print('Load Datasets Started...')
    df = pd.read_csv(path, sep=sep, names=header, engine='python')
    print('Load Datasets Ended...')
    # n_users = df.user_id.unique().shape[0]
    # n_items = df.item_id.unique().shape[0]
    n_users = df.user_id.max()
    n_items = df.item_id.max()
    rate_max = df.rating.max()

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []
    train_rating_1= []

    train_dict = {}
    for line in tqdm(df.itertuples()):
        u = line[1] - 1
        i = line[2] - 1
        r = line[3]
        if (u,i) in test_data:
            continue
        train_dict[(u, i)] = r

    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(line[3])
        train_rating_1.append(1)
        for t in range(num_negatives):
            j = np.random.randint(n_items)
            while (u, j) in train_dict.keys():
                j = np.random.randint(n_items)
            train_row.append(u)
            train_col.append(j)
            train_rating.append(0)

    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))
    all_items = set(np.arange(n_items))
    train_user_item_matrix = []
    neg_user_item_matrix = {}
    for u in range(n_users):
        neg_user_item_matrix[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))
        train_user_item_matrix.append(list(train_matrix.getrow(u).toarray()[0]))

    test_row = []
    test_col = []
    test_rating = []
    unique_users = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        unique_users.append(line[1] - 1)
        test_rating.append(line[3])
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
    test_user_item_matrix = {}
    for u in range(n_users):
        test_user_item_matrix[u] = test_matrix.getrow(u).nonzero()[1]

    return train_matrix.todok(), test_matrix.todok(), n_users, n_items, neg_user_item_matrix, test_user_item_matrix, unique_users, rate_max


if __name__ == '__main__':
    # check dataloader
    train_data, test_data, \
    num_users, num_items, \
    neg_user_item_matrix, \
    train_user_item_matrix, \
    unqiue, rate_max = load_rating_data(path="../data/ratings.dat", test_size=0.1, sep="::")

    train_datasets = BPRData(train_data, num_users, num_items, rate_max)
    train_loader = data.DataLoader(train_datasets, batch_size=256, shuffle=True, num_workers=4)

    test_datasets = BPRData(test_data,   num_users, num_items, rate_max)
    test_loader = data.DataLoader(test_datasets, batch_size=256, shuffle=False, num_workers=4)

    for user, item, rate, rate_max in train_loader:
        print(user, item, rate, rate_max)