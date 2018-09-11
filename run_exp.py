from __future__ import print_function
import torch._utils
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
import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')
from torch.multiprocessing import Manager
from collections import OrderedDict, defaultdict
import numpy as np
import time
import pickle
import json
import os
import glob
from model import Seq2seq
from train import train_epoch, eval_epoch, test_image
from custom_arg import arg
import torch.multiprocessing as mp

def run_exp(args, verbose=True):
    model = Seq2seq(args)
    print("DEVICE", args.device)
    model.train()
    model.share_memory()
    if args.cuda:
        model.cuda(args.device)


    from_ckpt = args.prefix + "ckpts"+str(args.output_len) + "/*"
    print(from_ckpt)
    try:
        list_of_files = glob.glob(from_ckpt)
        if list_of_files:
            ckpt_fn = list_of_files[-1]
            print(ckpt_fn)
            latest_ckpt = torch.load(ckpt_fn)
            model.load_state_dict(latest_ckpt)
            if(verbose):
                print("Loading weights from", ckpt_fn)
    except:
        if(verbose):
            print("No checkpoints found in", from_ckpt)

    print("Trainable Model Param Sizes", [p.size() for p in model.parameters() if p.requires_grad])

    val = open("logs/" + args.prefix + "vl" + str(args.output_len), "w")
    tr = open("logs/" + args.prefix + "tl" + str(args.output_len), "w")

    init_lr = args.lr

    if args.opt == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=init_lr,
            momentum=args.momentum,
            weight_decay=args.l2)
        print("using sgd", init_lr)
    if args.opt == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=init_lr,
            betas=(0.9, 0.999), # (0.,0.), #                                                         
            weight_decay=args.l2)
        print("using adam")

    for epoch in range(0, args.epochs):
        if(epoch%100 == 0):
            vl = eval_epoch(args, model)
            val.write(str(vl) + "\n")
            print("valid loss", epoch, vl)
            print(genimg(args, model=model))
        if(epoch%100 == 0):
            torch.save(model.state_dict(), "/tmp/" + args.prefix+"ckpts"+str(args.output_len)+"/" + str(epoch) + ".pth.tar")
        tl = train_epoch(args, model, optimizer)
        print("train loss", epoch, tl)
        tr.write(str(tl) + "\n")
    val.close()
    tl.close()

def genimg(args, model=None, verbose=False):
    if(model == None):

        model = Seq2seq(args)
        if(verbose):
            print("DEVICE", args.device)
        if args.cuda:
            model.cuda(args.device)

        from_ckpt = args.prefix + "ckpts"+str(args.output_len) + "/*"
        print(from_ckpt)
        try:
            list_of_files = glob.glob(from_ckpt)
            if list_of_files:
                list_of_files.sort(key=os.path.getctime)

                ckpt_fn = list_of_files[-1]
                print(ckpt_fn)
                latest_ckpt = torch.load(ckpt_fn)
                model.load_state_dict(latest_ckpt)
                if(verbose):
                    print("Loading weights from", ckpt_fn)
        except:
            if(verbose):
                print("No checkpoints found in", from_ckpt)
    else:
        if(verbose):
            print("using existent model")
    return test_image(args, model)
