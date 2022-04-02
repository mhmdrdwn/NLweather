#!/usr/bin/env python3

__author__ = 'Mohamed Radwan'


import torch
from torch import nn
from torch.autograd import Variable
    

class LSTM(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = 10
        self.lstm = nn.LSTM(input_size=input_size[0], hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, 
                            bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), 
            self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), 
            self.hidden_size))

        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        #hn = hn.view(-1, self.hidden_size*2)
        out = lstm_out[:, -1, :]
        out = self.linear(out)
    
        return out
    

class BiLinearPoolingLSTM(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super(BiLinearPoolingLSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = 10
        self.lstm1 = nn.LSTM(input_size=input_size[0], hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, 
                            bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=input_size[1], hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, 
                            bidirectional=True)
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
        h1 = Variable(torch.zeros(self.num_layers*2, x1.size(0), 
            self.hidden_size))
        c1 = Variable(torch.zeros(self.num_layers*2, x1.size(0), 
            self.hidden_size))
        h2 = Variable(torch.zeros(self.num_layers*2, x2.size(0), 
            self.hidden_size))
        c2 = Variable(torch.zeros(self.num_layers*2, x2.size(0), 
            self.hidden_size))
        
        """Get the LSTM outputs, Hidden states (hn1, hn2) and 
        cell states(cn1, cn2) can be used also"""
        lstm_out1, (hn1, cn1) = self.lstm1(x1, (h1, c1))
        lstm_out2, (hn2, cn2) = self.lstm2(x2, (h2, c2))
        
        main_output = self.TFN(lstm_out1, lstm_out2)
        
        """Now we reshape into batchsize vs 33*33 where each 32 is 
        the output of LSTM and 1 for the added ones vector"""
        main_output = torch.flatten(main_output)
        main_output = main_output.view(int(main_output.shape[0]/(33*33)), 
                33*33)
        
        out = self.linear(main_output)
    
        return out
    
    
class BiLinearPoolingAutoEncoderLSTM(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super(BiLinearPoolingAutoEncoderLSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = 10
        self.lstm1 = nn.LSTM(input_size=input_size[0], 
                hidden_size=self.hidden_size, num_layers=num_layers, 
                batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=input_size[1], 
                hidden_size=self.hidden_size, num_layers=num_layers, 
                batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(input_size=32, hidden_size=self.hidden_size,
                            num_layers=num_layers, batch_first=True, 
                            bidirectional=True)
        self.lstm4 = nn.LSTM(input_size=32, hidden_size=self.hidden_size,
                            num_layers=num_layers, batch_first=True, 
                            bidirectional=True)
        
        self.linear1 = nn.Linear((hidden_size*2), 
                input_size[0]*self.seq_length) 
        self.linear2 = nn.Linear((hidden_size*2), 
                input_size[1]*self.seq_length)
        self.linear3 = nn.Linear(output_size*self.seq_length, output_size)
        self.linear4 = nn.Linear(((hidden_size*2)+1)**2, 
                output_size*self.seq_length)
    
    
    def TFN(self, enc1, enc2):         
        enc2_ones = enc1_ones = torch.ones((enc1.shape[0], 1))
        enc2 = torch.cat((enc2, enc2_ones), 1)
        enc1 = torch.cat((enc1, enc1_ones), 1)
        enc1 = enc1.view(enc1.shape[-1], enc1.shape[0])
        enc2 = enc2.view(enc2.shape[-1], enc2.shape[0])
        enc2 = enc2.unsqueeze(1)
        pooling_enc = enc1*enc2
        return pooling_enc
    
    def encode(self, x1, x2):
        h1 = Variable(torch.zeros(self.num_layers*2, x1.size(0), 
            self.hidden_size))
        c1 = Variable(torch.zeros(self.num_layers*2, x1.size(0), 
            self.hidden_size))
        h2 = Variable(torch.zeros(self.num_layers*2, x2.size(0), 
            self.hidden_size))
        c2 = Variable(torch.zeros(self.num_layers*2, x2.size(0), 
            self.hidden_size))
        lstm_out1, (hn1, cn1) = self.lstm1(x1, (h1, c1))
        lstm_out2, (hn2, cn2) = self.lstm2(x2, (h2, c2))
        enc1 = lstm_out1[:, -1, :]
        enc2 = lstm_out2[:, -1, :]
        return enc1, enc2
        
    def decode(self, encoded_x1, encoded_x2, pooled_x):
        h1 = Variable(torch.zeros(self.num_layers*2, encoded_x1.size(0), 
            self.hidden_size))
        c1 = Variable(torch.zeros(self.num_layers*2, encoded_x1.size(0), 
            self.hidden_size))
        h2 = Variable(torch.zeros(self.num_layers*2, encoded_x2.size(0), 
            self.hidden_size))
        c2 = Variable(torch.zeros(self.num_layers*2, encoded_x2.size(0), 
            self.hidden_size))
        lstm_out1, (hn1, cn1) = self.lstm3(encoded_x2, (h1, c1))
        lstm_out2, (hn2, cn2) = self.lstm4(encoded_x2, (h2, c2))
        dec1 = lstm_out1[:, -1, :]
        dec2 = lstm_out2[:, -1, :]
        dec1 = self.linear1(dec1)
        dec2 = self.linear2(dec2)
        dec_y = self.linear3(pooled_x)
        return dec1, dec2, dec_y
        
    def forward(self, x1, x2):
        enc1, enc2 = self.encode(x1, x2)
        pooling_enc = self.TFN(enc1, enc2)
        pooling_enc = torch.flatten(pooling_enc)
        pooling_enc = pooling_enc.view(int(pooling_enc.shape[0]/(33*33)), 33*33)
        pooling_enc = self.linear4(pooling_enc)
        enc1 = enc1.unsqueeze(1).repeat(1, self.seq_length, 1)
        enc2 = enc2.unsqueeze(1).repeat(1, self.seq_length, 1)
        dec_x1, dec_x2, dec_y = self.decode(enc1, enc2, pooling_enc)
        dec_x1 = dec_x1.view(dec_x1.shape[0], self.seq_length, 
                int(dec_x1.shape[-1]/self.seq_length))
        dec_x2 = dec_x2.view(dec_x2.shape[0], self.seq_length, 
                int(dec_x2.shape[-1]/self.seq_length))
        return dec_x1, dec_x2, dec_y

