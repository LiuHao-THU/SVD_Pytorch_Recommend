# -*- coding: utf-8 -*-
# @Author: hliu.Luke
# @Date:   2020-07-31 10:00:34
# @Last Modified by:   hliu.Luke
# @Last Modified time: 2020-08-06 10:35:59

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import evaluation
from dataLoader import BPRData, load_rating_data

parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
    type=float, 
    default=5e-4, 
    help="learning rate")
parser.add_argument("--lamda", 
    type=float, 
    default=0.001, 
    help="model regularization rate")
parser.add_argument("--batch_size", 
    type=int, 
    default=256, 
    help="batch size for training")
parser.add_argument("--epochs", 
    type=int,
    default=200,  
    help="training epoches")
parser.add_argument("--top_k", 
    type=int, 
    default=10, 
    help="compute metrics@top_k")
parser.add_argument("--factor_num", 
    type=int,
    default=100, 
    help="predictive factors numbers in the model")
parser.add_argument("--num_ng", 
    type=int,
    default=4, 
    help="sample negative items for training")
parser.add_argument("--test_num_ng", 
    type=int,
    default=99, 
    help="sample part of negative items for testing")
parser.add_argument("--out", 
    default=True,
    help="save model or not")
parser.add_argument("--gpu", 
    type=str,
    default="0",  
    help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


############################## PREPARE DATASET ##########################

train_data, test_data, \
num_users, num_items, \
neg_user_item_matrix, \
train_user_item_matrix, \
unqiue, rate_max = load_rating_data(path="../data/ratings.dat", test_size=0.1, sep="::")


train_datasets = BPRData(train_data, num_users, num_items, rate_max)
train_loader = data.DataLoader(train_datasets, batch_size=1024, shuffle=True, num_workers=12)

test_datasets = BPRData(test_data,   num_users, num_items, rate_max)
test_loader = data.DataLoader(test_datasets, batch_size=256, shuffle=False,  num_workers=12)

print('len train loader = {} len test loader = {}'.format(len(train_loader), len(test_loader)))

########################### CREATE MODEL #################################
model = model.BPR(num_users, num_items, np.mean(list(train_data.tocoo().data)), args.factor_num)
model.cuda()

# optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, lr_decay=0, weight_decay=0)
optimizer = optim.RMSprop(
                model.parameters(), lr=args.lr, weight_decay=1e-5)
# optimizer = torch.optim.SGD(
#             model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

writer = SummaryWriter() # for visualization


# def loss_fn(pred, rate, rate_max):
#     c_ui = 1+  0.2 * torch.abs(rate - (rate_max )/2)
#     mse = torch.pow((rate_max - rate)  - pred, 2)
#     return torch.sum(c_ui * mse)

def loss_fn(pred, rate, rate_max):
    # c_ui = 1+  0.2 * torch.abs(rate - (rate_max )/2)
    mse = torch.pow(rate  - pred, 2)
    return torch.sum(mse)

########################### TRAIN  MODEL #################################
count = 0
min_val_metric = np.Inf
n_epochs_stop = 5
epochs_no_improve = 0

min_val_rmse = np.Inf
min_val_mse = np.Inf
min_val_mae = np.Inf
for epoch in range(args.epochs):
    model.train()
    start_time = time.time()
    loss_total = 0
    # train started
    for i, (user, item, rate, rate_max) in enumerate(train_loader):
        user = user.cuda().long()
        item = item.cuda().long()
        rate = rate.cuda()
        rate_max = rate_max.cuda()
        model.zero_grad()
        pred = model.forward(user, item)
        loss = loss_fn(pred, rate, rate_max)
        # calculate loss function
        loss.backward()
        optimizer.step()
        writer.add_scalar('data/loss', loss.item(), count)
        count += 1
        loss_total += loss.item()
    if i % 1000 == 0:
        print("Epoch: %04d; loss = %.4f cost time  %.4f" % (epoch, np.mean(loss.item()), time.time() - start_time))


    if (epoch) % 1 == 0:
        model.eval()
        pred_list = None
        rate_list = None
        for user, item, rate, rate_max in test_loader:
            user = user.cuda().long()
            item = item.cuda().long()
            rate = rate.cuda()
            rate_max  =rate_max.cuda()
            pred = model.forward(user, item)
            # calculate loss
            loss_eval = loss_fn(pred, rate, rate_max)

            if pred_list is not None:
                pred_list = torch.cat((pred, pred_list), dim = 0)
            else:
                pred_list = pred

            if rate_list is not None:
                rate_list = torch.cat((rate, rate_list), dim = 0)
            else:
                rate_list = rate


        mse, rmse, mae = evaluation.mse_rmse_mae(pred_list, rate_list)


        # early stop 
        if rmse.cpu().detach().numpy() < min_val_rmse:
            epochs_no_improve = 0
            min_val_rmse = rmse.cpu().detach().numpy()
            min_val_mse = mse.cpu().detach().numpy()
            min_val_mae = mae.cpu().detach().numpy()
            print("MSE:%.4f ; RMSE:%.4f; MAE:%.4f" %(mse.cpu().detach().numpy(), rmse.cpu().detach().numpy(), mae.cpu().detach().numpy()) )
        else:
            epochs_no_improve += 1
            print("MSE:%.4f ; RMSE:%.4f; MAE:%.4f" %(mse.cpu().detach().numpy(), rmse.cpu().detach().numpy(), mae.cpu().detach().numpy()) )

        # if epochs_no_improve == n_epochs_stop:
        #     print('Early stopping!' )
        #     print("MSE:%.4f ; RMSE:%.4f; MAE:%.4f" %(min_val_mse, min_val_rmse, min_val_mae))
        #     break