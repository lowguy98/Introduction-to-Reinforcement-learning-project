import numpy as np
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from generate_game import *
from Q_values_network import *
from Chess_env import *
import torch

flag = 1 # Q-learning if flag==1,else SARSA

size_board = 4

def EMA(list, alpha=0.99):
    new_list = np.zeros(len(list))
    new_list[0] = list[0]
    for i in range(1,len(new_list)):
        new_list[i] = alpha * new_list[i-1] + (1 - alpha) * list[i]
    return new_list

def plot(R_save,N_moves_save):
    print(R_save.shape)
    R_save = EMA(R_save)
    N_moves_save = EMA(N_moves_save)
    x=range(R_save.shape[0])
    plt.plot(x,R_save)
    plt.ylabel('Reward')
    plt.xlabel('Episode Number')
    plt.title('Reward per game')
    plt.show()
    plt.plot(x,N_moves_save)
    plt.ylabel('Moves')
    plt.xlabel('Episode Number')
    plt.title('Number of moves per game')
    plt.show()
    # plt.subplot(1,2,1)
    # plt.plot(x,R_save)
    # plt.ylabel('Reward')
    # plt.xlabel('Episode Number')
    # plt.title('Reward per game')
    # plt.subplot(1,2,2)
    # plt.plot(x,N_moves_save)
    # plt.ylabel('Reward')
    # plt.xlabel('Episode Number')
    # plt.title('Number of moves per game')
    # plt.show()

def epsilon_greedy(epsilon_f,Qvalues,allowed_index,i):
    # np.random.seed(i)
    rand_a=np.random.uniform(0,1)<epsilon_f
            
    if rand_a==1:
        
        a_agent=np.random.permutation(allowed_index)[0]
        
    else:
        
        a=np.argmax(Qvalues)
        a_agent=np.copy(allowed_index[a])
    
    return a_agent

def main():

    env=Chess_Env(size_board)   

    # Parameters
    epsilon_0 = 0.1   #epsilon for the e-greedy policy
    beta = 0.00005    #epsilon discount factor
    gamma = 0.99      #Learning discount factor
    eta = 0.001      #learning rate
    N_episodes = 4000  #Number of games, each game ends when we have a checkmate or a draw

    network = Network(n_input_layer=58,n_hidden_layer=200,n_output_layer=32) #Initialise network
    optimizer = torch.optim.Adam(network.parameters(),lr = eta)
    loss_func = torch.nn.MSELoss(reduction='none')

    R_save = np.zeros(N_episodes)
    N_moves_save = np.zeros(N_episodes)
    

    for n in range(N_episodes):
        epsilon_f = epsilon_0 / (1 + beta * n)
        S,X,allowed_a=env.Initialise_game()
        # print('Initial board\n',S)
        allowed_index = np.where(allowed_a==1)[0]
        Done=0                       
        i=1                                     
        
        if (n%100==0) and (n!=0):    
            print(np.mean(R_save[n-100:n]))
            print(np.mean(N_moves_save[n-100:n]))
        # if n%10==0:    
        #     print("R",R_save[n-10:n])
        #     print("N",N_moves_save[n-10:n])
        
        while Done==0:

            X = env.Features()
            input = torch.tensor(X,requires_grad=True).float()
            output = network(input)
            Q = np.copy(output.data.numpy())
            Qvalues=np.copy(Q[allowed_index])
            
            
            a_agent = epsilon_greedy(epsilon_f,Qvalues,allowed_index,i)
            S_next,X_next,allowed_a_next,R,Done=env.OneStep(a_agent)
            # print('updated board\n',S)
            allowed_index_next = np.where(allowed_a_next==1)[0] 


            if Done==1:
                # if game has ended
                target = torch.zeros_like(output)
                target[a_agent] = R
                loss_weight = torch.zeros_like(output)
                loss_weight[a_agent] = 1
                loss = loss_func(output,target)   # (R-output)²

                optimizer.zero_grad()
                loss.backward(loss_weight)
                optimizer.step()
                
                R_save[n]=np.copy(R)
                N_moves_save[n]=np.copy(i)
                break


            else:
                # if game has not ended
                if flag: 
                    # flag==1,Q-learning
                    input_next = torch.tensor(X_next,requires_grad=True).float()
                    output_next = network(input_next)
                    Q_next =  np.copy(output_next.data.numpy())
                    Qvalues_next=np.copy(Q_next[allowed_index_next])   # Q values of allowed actions               
                    Qmax_index_next = allowed_index_next[np.argmax(Qvalues_next)] # index of the maximum Q values in allowed actions

                    target = torch.zeros_like(output)
                    target[a_agent] = R + gamma*Q_next[Qmax_index_next]
                    loss_weight = torch.zeros_like(output_next)
                    loss_weight[a_agent] = 1.
                    loss_weight = loss_weight.float()
                    loss = loss_func(output,target) # (R+gamma*Q_next[Qmax_index_next]-output)²
               
                    optimizer.zero_grad()
                    loss.backward(loss_weight)
                    optimizer.step()

                    allowed_index = np.copy(allowed_index_next)
                    

                else:
                    # flag==0,SARSA
                    input_next = torch.tensor(X_next,requires_grad=True).float()
                    output_next = network(input_next)
                    Q_next =  network.forward(input_next).data.numpy()
                    Qvalues_next=np.copy(Q_next[allowed_index_next])
                    a_agent_next = epsilon_greedy(epsilon_f,Qvalues_next,allowed_index_next,i)  # choose an action by epsilon greedy
                    
                    target = torch.zeros_like(output)
                    target[a_agent] = R + gamma*Q_next[a_agent_next]
                    loss_weight = torch.zeros_like(output_next)
                    loss_weight[a_agent] = 1.
                    loss_weight = loss_weight.float()
                    loss = loss_func(output,target)  # (R+gamma*Q_next[a_agent_next]-output)²

                    optimizer.zero_grad()
                    loss.backward(loss_weight)
                    optimizer.step()

                    allowed_index = np.copy(allowed_index_next)
            
            i += 1
            
    plot(R_save,N_moves_save)

if __name__ == '__main__':
    main()
