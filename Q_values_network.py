import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self,n_input_layer,n_hidden_layer,n_output_layer):
        super(Network,self).__init__()
        self.hidden1 = nn.Linear(n_input_layer,n_hidden_layer)
        self.predict = nn.Linear(n_hidden_layer,n_output_layer)


    def forward(self,x):
        a1 = self.hidden1(x)
        x1 = F.relu(a1)
        a2 =self.predict(x1)
        out = F.relu(a2)
        
        return out

         
