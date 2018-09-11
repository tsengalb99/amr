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
import torch.multiprocessing as mp


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, args=None):
        print(n_layers, "encoder layers")
        super(EncoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.RNN(hidden_size, hidden_size, n_layers)
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = torch.unsqueeze(embedded, 0)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return result.cuda(self.args.device)
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, args=None):
        super(DecoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(args.state_dim, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = F.relu(embedded)
        embedded = torch.unsqueeze(embedded, 0)
        output, hidden = self.gru(embedded, hidden)
        self.gru.flatten_parameters()
        output = self.out(output.squeeze(0))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return result.cuda(self.args.device)
        else:
            return result

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, args=None):
        super(AttnDecoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(args.state_dim, hidden_size)#output_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

        self.attn = nn.Linear(self.hidden_size * 2, self.args.output_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.out = nn.Linear(self.hidden_size * 2, output_size)#output_size

        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, encoder_outputs):
        embedded = F.relu(self.embedding(input))
        attn_weights = F.relu(self.attn(torch.cat((embedded, hidden.squeeze(0)), 1)))
        context = torch.bmm(attn_weights.unsqueeze(1),
                            encoder_outputs.transpose(0, 1)).squeeze(1)
        rnn_input = self.attn_combine(torch.cat((embedded, context), 1))
        rnn_input = rnn_input.unsqueeze(0)

        output, hidden = self.gru(rnn_input, hidden)

        output = self.out( torch.cat((output.squeeze(0), context), 1) ) #self.softmax

        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return result.cuda(self.args.device)
        else:
            return result
        

class Seq2seq(nn.Module):
    def __init__(self, args):
        print(args.state_dim, args.state_dim_out)
        super(Seq2seq, self).__init__()
        self.args = args

        T = torch.cuda if self.args.cuda else torch

        self.enc = EncoderRNN(self.args.state_dim, self.args.hidden_size, args=args)

        self.use_attn = bool(args.attn)
        print("USING ATTN" if self.use_attn else "NOT USING ATTN")
        
        if self.use_attn:
            self.dec = AttnDecoderRNN(self.args.hidden_size, self.args.state_dim_out, args=args)
        else:
            self.dec = DecoderRNN(self.args.hidden_size, self.args.state_dim_out, args=args)

    def parameters(self):
        return list(self.enc.parameters()) + list(self.dec.parameters())

    def forward(self, x):
        encoder_hidden = self.enc.initHidden()
        hs = []
        #front padding? maybe this works who knows
        sztmp = None
        for t in range(self.args.input_len):
            encoder_output, encoder_hidden = self.enc(x[t], encoder_hidden)
            sztmp = encoder_output.size()
            hs += [encoder_output]

        decoder_hidden = hs[-1]
        
        hs = torch.cat(hs, 0).cuda(self.args.device)
        hs = hs.permute(1,2,0) #get the last one to be embedded properly
        embedding = nn.Linear(self.args.input_len, self.args.output_len).cuda(self.args.device)
        hs = F.relu(embedding(hs))
        hs = hs.permute(2,0,1)


        inp = Variable(torch.zeros(self.args.batch_size, self.args.state_dim))
        if self.args.cuda: inp = inp.cuda(self.args.device)
        ys = []

        if self.use_attn:
            for t in range(self.args.output_len):
                decoder_output, decoder_hidden = self.dec(inp, decoder_hidden, hs)
                inp = decoder_output[:, 0:self.args.state_dim]
                ys += [decoder_output]
        else:
            for t in range(self.args.output_len):
                decoder_output, decoder_hidden = self.dec(inp, decoder_hidden)
                inp = decoder_output[:, 0:self.args.state_dim]
                ys += [decoder_output]

        return torch.cat([torch.unsqueeze(y, dim=0) for y in ys])
