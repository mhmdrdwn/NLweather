#!/usr/bin/env python3

__author__ = 'Mohamed Radwan'


import torch
from torch import nn
from torch.autograd import Variable
    
class BiLinearPoolingLSTM(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super(BiLinearPoolingLSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = 10
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(((hidden_size*2)+1)**2, output_size)
    
        
    def TFN(self, lstm_out1, lstm_out2): 
        """This idea is taken from paper: Multimodal Tensorfusion network
        
        steps:
        - add vectors of ones at the end of each lstm outputs to account
        for each output alone before the cross multiplication
        - matrix multiplications for the two vectors will end up with 
        33x33 matrix"""
        
        out1 = lstm_out1[:, -1, :]
        out2 = lstm_out2[:, -1, :]
        out2_ones = out1_ones = torch.ones((out2.shape[0], 1))
        out2 = torch.cat((out2, out2_ones), 1)
        out1 = torch.cat((out1, out1_ones), 1)
        
        """We move the batch size to the last dimenstion to perform 
        matrix multiplications and add extra dimension...This is only a hack"""
        out1 = out1.view(out1.shape[-1], out1.shape[0])
        out2 = out2.view(out2.shape[-1], out2.shape[0])
        out2 = out2.unsqueeze(1)
        main_output = out1*out2
        
        return main_output
        
        
    def forward(self, x1, x2):
        h1 = Variable(torch.zeros(self.num_layers*2, x1.size(0), self.hidden_size))
        c1 = Variable(torch.zeros(self.num_layers*2, x1.size(0), self.hidden_size))
        h2 = Variable(torch.zeros(self.num_layers*2, x2.size(0), self.hidden_size))
        c2 = Variable(torch.zeros(self.num_layers*2, x2.size(0), self.hidden_size))
        
        """Get the LSTM outputs, Hidden states (hn1, hn2) and 
        cell states(cn1, cn2) can be used also"""
        lstm_out1, (hn1, cn1) = self.lstm1(x1, (h1, c1))
        lstm_out2, (hn2, cn2) = self.lstm2(x2, (h2, c2))
        
        main_output = self.TFN(lstm_out1, lstm_out2)
        
        """Now we reshape into batchsize vs 33*33 where each 32 is 
        the output of LSTM and 1 for the added ones vector"""
        main_output = torch.flatten(main_output)
        main_output = main_output.view(int(main_output.shape[0]/(33*33)), 33*33)
        
        out = self.linear(main_output)
    
        return out
