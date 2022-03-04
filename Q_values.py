import torch
import torch.nn as nn
import numpy as np

class Network():
    
    def __init__(self,n_input_layer,n_hidden_layer,n_output_layer,W1,b1,W2,b2):
        self.n_input_layer = n_input_layer
        self.n_hidden_layer = n_hidden_layer
        self.n_output_layerr = n_output_layer
        self.w1 = W1
        self.b1 = b1
        self.w2 = W2
        self.b2 = b2
        self.x1 = 0
        self.y = 0


    def relu(self,x):
        """relu function"""
        return np.where(x<0,0,x)

    def forward(self,x):
        print('w1',self.w1.shape)
        print('b1',self.b1.shape)
        print('x',x.shape)
        self.x1=self.relu(self.w1.T@x+self.b1)
        self.y=self.relu(self.w2.T@self.x1+self.b2)
        return self.y,self.x1

    def back_prop(self,delta,eta,a_agent,x):
                
        delta_W2=eta*delta*self.x1
        delta_W1=eta*np.outer(x,delta*self.w2[:,a_agent]*(self.x1>0))                
               
        self.w2[:,a_agent]=self.w2[:,a_agent]+eta*delta_W2
        self.b2[a_agent]=self.b2[a_agent]+eta*delta
        
        self.w1[:,a_agent]=self.w1[:,a_agent]+eta*delta_W1
        b1=b1+eta*delta*self.w2[:,a_agent]*(self.x1>0)
        
        
