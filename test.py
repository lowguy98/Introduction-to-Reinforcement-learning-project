import torch

class Network(torch.nn.Module):
    
    def __init__(self):
        super(Network,self).__init__()
        self.predict = torch.nn.Linear(2,2)
    
    def initialize_weights(self):
        for m in self.modules():
            torch.nn.init.normal_(m.weight.data, 0, 0.1)
            m.bias.data.zero_()

    def forward(self,x):
        out=self.predict(x)       
        return out

net = Network() 
optimizer = torch.optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

x=torch.tensor([2.,7.],requires_grad=True)
print(x)
out = net.forward(x)
for parameter in net.parameters():
    print(parameter)
loss_func = torch.nn.MSELoss(reduction='none')
y = torch.tensor([3.,6.])
loss = loss_func(out,y)
print('loss',loss)
loss_weight = torch.tensor([0.,1.])
loss.backward(loss_weight)
optimizer.step()
for parameter in net.parameters():
    print(parameter.grad)
print('x.grad',x.grad)

