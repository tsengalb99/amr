from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn import Parameter
from torch.autograd import Variable
from torch.sparse import FloatTensor as STensor
from torch.cuda.sparse import FloatTensor as CudaSTensor
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from collections import OrderedDict, defaultdict
from model import Seq2seq
import time
import random
from datareader import LorenzDataset
import os
import numpy as np
import matplotlib as mpl
mpl.use('agg')
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def train_epoch(args, model, opt):
    model = model.train()
    data_loader = torch.utils.data.DataLoader(
        LorenzDataset(args, which='train'),
        batch_size=args.batch_size, shuffle=True, drop_last=True, 
        num_workers=1, pin_memory=True)

    totalLoss = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.type(torch.FloatTensor)
        data = torch.transpose(data, 0, 1) # num_step x batch x dim
        target = target.type(torch.FloatTensor)
        target = torch.transpose(target, 0, 1)
        if args.cuda: data, target = data.cuda(args.device), target.cuda(args.device)
        data, target = Variable(data), Variable(target)


        opt.zero_grad()
        output = model(data)
        loss = torch.sqrt(F.mse_loss(output, target))
        loss.backward()
        opt.step()

        #actual coordinate loss
        coordloss = torch.sqrt(F.mse_loss(output[:,:,0:3], target[:,:,0:3]))
        coordloss = coordloss.data.cpu().numpy()[0]
        totalLoss += coordloss
    return float(totalLoss)/(batch_idx + 1)

def eval_epoch(args, model):
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        LorenzDataset(args, which='valid'),
        batch_size=args.batch_size, shuffle=True, drop_last=True, 
        num_workers=1, pin_memory=True)
    
    totalLoss = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.type(torch.FloatTensor)
        data = torch.transpose(data, 0, 1) # num_step x batch x dim
        target = target.type(torch.FloatTensor)
        target = torch.transpose(target, 0, 1)
        if args.cuda: data, target = data.cuda(args.device), target.cuda(args.device)
        data, target = Variable(data), Variable(target)

        output = model(data)

        #actual coordinate loss
        coordloss = torch.sqrt(F.mse_loss(output[:,:,0:3], target[:,:,0:3]))
        coordloss = coordloss.data.cpu().numpy()[0]
        totalLoss += coordloss
    return float(totalLoss)/(batch_idx + 1)

def test_image(args, model):
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        LorenzDataset(args, which='valid'),
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=1, pin_memory=True)
    
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.type(torch.FloatTensor)
        data = torch.transpose(data, 0, 1)
        target = target.type(torch.FloatTensor)
        target = torch.transpose(target, 0, 1)
        if args.cuda: data, target = data.cuda(args.device), target.cuda(args.device)
        data, target = Variable(data), Variable(target)

        output = model(data)

        outputArr = output.data.cpu().numpy()
        targetArr = target.data.cpu().numpy()
        dataArr = data.data.cpu().numpy()

        coordloss = torch.sqrt(F.mse_loss(output[:,:,0:3], target[:,:,0:3]))
        coordloss = coordloss.data.cpu().numpy()[0]

        for i in range(min(args.batch_size, 5)):

            xd = dataArr[:, i, 0]
            yd = dataArr[:, i, 1]
            zd = dataArr[:, i, 2]

            xo = outputArr[:, i, 0]
            yo = outputArr[:, i, 1]
            zo = outputArr[:, i, 2]
            
            xt = targetArr[:, i, 0]
            yt = targetArr[:, i, 1]
            zt = targetArr[:, i, 2]
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot3D(xd, yd, zd, 'ro-')
            ax.plot3D(xo, yo, zo, 'g^-')
            ax.plot3D(xt, yt, zt, 'b+-')
            fig.savefig(args.prefix + "img" + str(args.output_len) + "/plot" + str(i) + ".png")
            plt.close('all')
        return coordloss
