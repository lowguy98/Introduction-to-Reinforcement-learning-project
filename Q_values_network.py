import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self,n_input_layer,n_hidden_layer,n_output_layer):
        super(Network,self).__init__()
        self.hidden1 = nn.Linear(n_input_layer,n_hidden_layer)
        self.predict = nn.Linear(n_hidden_layer,n_output_layer)

    def initialize_weights(self):
        for m in self.modules():
            nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def forward(self,x):
        out = self.hidden1(x)
        out = F.relu(out)
        out =self.predict(out)
        out = F.relu(out)       
        return out

         
