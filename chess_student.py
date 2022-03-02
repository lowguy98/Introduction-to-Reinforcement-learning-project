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

def EMA(list, alpha=0.8):
    new_list = np.zeros(len(list))
    new_list[0] = list[0]
    for i in range(1, len(list)):
        new_list[i] = alpha * new_list[i-1] + (1 - alpha) * list[i]
    return new_list

def plot(R_save,N_moves_save):
    R_save = EMA(R_save)
    N_moves_save = EMA(N_moves_save)
    x=range(R_save.shape[0])
    plt.subplot(1,2,1)
    plt.plot(x,R_save)
    plt.ylabel('Reward')
    plt.xlabel('Episode Number')
    plt.title('Reward per game')
    plt.subplot(1,2,2)
    plt.plot(x,N_moves_save)
    plt.ylabel('Reward')
    plt.xlabel('Episode Number')
    plt.title('Number of moves per game')
    plt.show()

def epsilon_greedy(epsilon_f,Qvalues,allowed_index):

    rand_a=np.random.uniform(0,1)<epsilon_f
            
    if rand_a==1:
        
        a_agent=np.random.permutation(allowed_index)[0]
        
    else:
        
        a=np.argmax(Qvalues)
        a_agent=np.copy(allowed_index[a])
    
    return a_agent

def main():
    """
    Generate a new game

    """

    env=Chess_Env(size_board)   
   


    # Network Parameters
    epsilon_0 = 0.2   #epsilon for the e-greedy policy
    beta = 0.00005    #epsilon discount factor
    gamma = 0.85      #SARSA Learning discount factor
    eta = 0.0035      #learning rate
    N_episodes = 1000 #Number of games, each game ends when we have a checkmate or a draw

    ###  Training Loop  ###
    
    # THE FOLLOWING VARIABLES COULD CONTAIN THE REWARDS PER EPISODE AND THE
    # NUMBER OF MOVES PER EPISODE, FILL THEM IN THE CODE ABOVE FOR THE
    # LEARNING. OTHER WAYS TO DO THIS ARE POSSIBLE, THIS IS A SUGGESTION ONLY.    

    R_save = np.zeros([N_episodes, 1])
    N_moves_save = np.zeros([N_episodes, 1])

    # END OF SUGGESTIONS
    

    for n in range(N_episodes):
        epsilon_f = epsilon_0 / (1 + beta * n)
        S,X,allowed_a=env.Initialise_game()
        allowed_index = np.where(allowed_a==1)[0]     # INITIALISE GAME
        Done=0                                  # SET Done=0 AT THE BEGINNING
        i=1                                     # COUNTER FOR THE NUMBER OF ACTIONS (MOVES) IN AN EPISODE

        n_input_layer = 3*S.shape[0]*S.shape[0]+10  # Number of neurons of the input layer. 
        n_hidden_layer = 200  # Number of neurons of the hidden layer
        n_output_layer = 8*(S.shape[0]-1)+8 # Number of neurons of the output layer.

        network = Network(n_input_layer,n_hidden_layer,n_output_layer) #Initialise network
        optimizer = torch.optim.Adam(network.parameters(),lr = eta)
        loss_func = torch.nn.MSELoss(reduction='none')

        if n%50==0:
            
            print(np.mean(R_save[n-50:n]))
            print(np.mean(N_moves_save[n-50:n]))
        
        
        while Done==0:

            # Computing Features
            X = env.Features()

            # FILL THE CODE 
            # Enter inside the Q_values function and fill it with your code.
            # You need to compute the Q values as output of your neural
            # network. You can change the input of the function by adding other
            # data, but the input of the function is suggested. 
            input = torch.autograd.Variable(torch.from_numpy(X)).float()
            Q = network(input).detach().numpy()
            Qvalues=np.copy(Q[allowed_index])
            
            np.random.seed(i)
            a_agent = epsilon_greedy(epsilon_f,Qvalues,allowed_index)
            #print('S',S)
            #print('a_agent',a_agent)
            S_next,X_next,allowed_a_next,R,Done=env.OneStep(a_agent)
            allowed_index_next = np.where(allowed_a_next==1)[0] 


            # Check for draw or checkmate
            if Done==1:
                # King 2 has no freedom and it is checked
                # Checkmate and collect reward
                
                """
                FILL THE CODE
                Update the parameters of your network by applying backpropagation and Q-learning. You need to use the 
                rectified linear function as activation function (see supplementary materials). Exploit the Q value for 
                the action made. You computed previously Q values in the Q_values function. Be careful: this is the last 
                iteration of the episode, the agent gave checkmate.
                """
                target = Q[allowed_index[np.argmax(Qvalues)]]
                prediction = np.zeros(allowed_a.shape[0])
                prediction[a_agent] = target
                y = np.zeros(allowed_a.shape[0])
                y[a_agent] = R
                print('prediction',prediction)
                print('y',y)
                loss = loss_func(torch.autograd.Variable(torch.from_numpy(prediction)),torch.autograd.Variable(torch.from_numpy(y))).requires_grad_()
                print('loss',loss)
                optimizer.zero_grad()
                loss.backward(torch.ones_like(loss))
                optimizer.step()
                #delta=R-Q[a_agent]
                #network.back_prop(delta,eta,a_agent,X)
                R_save[n]=np.copy(R)
                N_moves_save[n]=np.copy(i)

                break


            else:
                if flag: 
                    # flag==1,Q-learning
                    input_next = torch.autograd.Variable(torch.from_numpy(X_next)).float()
                    Q_next =  network(input_next).detach().numpy()
                    Qvalues_next=np.copy(Q_next[allowed_index_next])
                    
                    target = Q[a_agent]-gamma*Q_next[allowed_index_next[np.argmax(Qvalues_next)]]
                    prediction = np.zeros(allowed_a_next.shape[0])
                    prediction[a_agent] = target
                    y = np.zeros(allowed_a.shape[0])
                    y[a_agent] = R
                    #print('prediction',prediction)
                    #print('y',y)
                    loss = loss_func(torch.autograd.Variable(torch.from_numpy(prediction)),torch.autograd.Variable(torch.from_numpy(y))).requires_grad_()
                    #print('loss',loss)
                    optimizer.zero_grad()
                    loss.backward(torch.ones_like(loss))
                    optimizer.step()
                    #delta=R+gamma*np.max(Qvalues_next)-Q[a_agent]
                    #network.back_prop(delta,eta,a_agent,X)

                    allowed_index = allowed_index_next
                    R_save[n]=np.copy(R)
                    N_moves_save[n]=np.copy(i)

                else:
                    # flag==0,SARSA
                    input_next = torch.autograd.Variable(torch.from_numpy(X_next)).float()
                    Q_next =  network.forward(input_next).detach().numpy()
                    Qvalues_next=np.copy(Q_next[allowed_index_next])
                    a_agent_next = epsilon_greedy(epsilon_f,Qvalues_next,allowed_index_next)
                    
                    target = Q[a_agent]-gamma*Q_next[a_agent_next]
                    prediction = np.zeros(allowed_a_next.shape[0])
                    prediction[a_agent] = target
                    y = np.zeros(allowed_a.shape[0])
                    y[a_agent] = R
                    loss = loss_func(torch.autograd.Variable(torch.from_numpy(prediction)),torch.autograd.Variable(torch.from_numpy(y))).requires_grad_()
                    print('loss',loss)
                    optimizer.zero_grad()
                    loss.backward(torch.ones_like(loss))
                    optimizer.step()
                    #delta=R+gamma*Qvalues_next[a_agent]-Q[a_agent]
                    #network.back_prop(delta,eta,a_agent,X)

                    allowed_index = allowed_index_next
                    R_save[n]=np.copy(R)
                    N_moves_save[n]=np.copy(i)
            

            # YOUR CODE ENDS HERE
            i += 1
            
    plot(R_save,N_moves_save)

if __name__ == '__main__':
    main()
