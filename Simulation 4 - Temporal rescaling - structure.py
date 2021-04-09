#Code written by Cristian Buc Calderon
#This code implements a RNN connected to a motor node
#The weights from RNN to motor read out the reservor dynamics
#and learn to produce an action at a specific point in times
#using reward-modulated hebbian learning

##to do list
#maybe in how strong the feedback weights!

import numpy as np, matplotlib.pyplot as plt, scipy as sc, os
from scipy.sparse import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy
import random as rd


os.chdir('C:/Cris/VoT Project/Feedback-Driven-Self-Organization/Cris way/noisy space')

###Intialization parameters
show_pic         = 1
save_pic         = 0
N_E_rnn          = 200
N_motor          = 6
time             = 1000
tau_E            = 1
tau_I            = 1/1
tau_G            = 1/1000
tau_A            = 1/10
tau_N            = 1/10
tau_w            = 1/2
N_trials         = 1
k_a              = 1
k_g              = 1
k_n              = 0
k_rnn_e          = 1
k_rnn_i          = 1
bias             = 0.4
gain_A_I         = 21
gain_A_RNN       = 21.4

###opening storing arrays
RNN_units      = np.zeros((N_trials,time,N_E_rnn))
sd_units       = np.zeros((N_trials,time,N_E_rnn))
G_units        = np.zeros((N_trials,time+1,N_motor))
A_units        = np.zeros((N_trials,time+1,N_motor))
N_units        = np.zeros((N_trials,time+1,N_motor))
I_units        = np.ones((N_trials,time+1))
test           = np.ones((N_trials,time+1))


###Loading weight matrices and desired times
W_rnn_E_E  = np.loadtxt('RNN_weights.txt', delimiter=',')
W_shut     = np.loadtxt('W_shut.txt', delimiter=',')
W_feedback = np.loadtxt('Feedback_weights.txt', delimiter=',')
W_go       = np.loadtxt('RNN_Go_weights.txt', delimiter=',')
W_a        = np.loadtxt('Go_A_weights.txt', delimiter=',')
W_n        = np.loadtxt('A_N_weights.txt', delimiter=',')
W_g_inh    = np.loadtxt('G_inh_weights.txt', delimiter=',')
W_inh      = np.loadtxt('N_G_inh_weights.txt', delimiter=',')
inputs     = np.loadtxt('Inputs_weights.txt', delimiter=',')
jEI        = np.loadtxt('I_E_weights.txt', delimiter=',')
jIE        = np.loadtxt('E_I_weights.txt', delimiter=',')
times      = np.loadtxt('Times.txt', delimiter=',')


lamb = 10
def tanh_f(x,lamb):
    z = ((2/(1 + np.exp(-lamb*x)))-1)
    z[z<0] = 0
    return z

def tanh_f_s(x,lamb):
    z = ((2/(1 + np.exp(-lamb*x)))-1)
    if z < 0:
        z = 0
    return z

def tanh_eli(x):
    lambd = 10
    d= 2.5
    z = d/(1 + np.exp((-lambd*x) + 5))
    z[z<0.02] = 0
    return z


#noise sd
sd_i   = 0
sd_rnn = 0
sd_g   = 0
sd_a   = 0
sd_n   = 0

#global input to g
gain_to_G = np.linspace(0.9,1.2,100)    #set to any positive number to accelerate the sequence

#record ratio
record_ratio = np.zeros(len(gain_to_G))

for u in range(len(gain_to_G)):
    print(u)
    for i in range(N_trials):

        ### initial state
        rnn_E          = np.zeros(N_E_rnn)
        rnn_I          = 0. #1.
        G              = np.zeros(N_motor)
        A              = np.zeros(N_motor)
        N              = np.zeros(N_motor)


        ### dynamics loop
        for j in range(time):

            ### RNN layer
            lamb = 10
            rnn_i_exc = np.dot(jIE,rnn_E) + np.dot(W_shut,A*gain_A_I) + np.random.normal(0,sd_i,1)
            rnn_I    += (-k_rnn_i*rnn_I + (rnn_i_exc)) * tau_I

            rnn_e_exc = np.dot(W_rnn_E_E,rnn_E) - (jEI*rnn_I) + (inputs[:,j]) + (np.dot(W_feedback,A*gain_A_RNN))
            rnn_E    += (-k_rnn_e*rnn_E + tanh_f(rnn_e_exc,lamb) + np.random.normal(0,sd_rnn,N_E_rnn)) * tau_E
            rnn_E[rnn_E>1] = 1
            rnn_E[rnn_E<0] = 0

            ### G layer
            g_exc = (np.dot(W_go,rnn_E) - np.dot(W_inh,N) - np.dot(W_g_inh,G))*gain_to_G[u] #if single gain then gain_to_G
            #if j <100:
             #g_exc[0] += -1 #for window shift
            G    += (-k_g*G + g_exc + (np.random.normal(0,sd_g,N_motor))) * tau_G
            G[G<0]=0

            ### A layer
            lamb = 10000
            a_exc = np.dot(W_a,G) - bias
            A    += (-k_a*A + tanh_f(a_exc,lamb) + np.random.normal(0,sd_a,N_motor)) * tau_A
            A[A<0]=0

            ### N Layer
            n_exc = np.dot(W_n,A) #* gain_to_N
            N    += (-k_n*N + n_exc + (np.random.normal(0,sd_n,N_motor))) * tau_N
            N[N<0]=0

            ### storing dynamics
            A_units[i,j+1,:] = A


    #quantifying the structure of the sequence (see paper)
    tot_ratio = 0
    for yy in range(N_motor-2):
        first_dif  = np.abs(np.argwhere(A_units[0,:,yy]>0.5)[0][0] - np.argwhere(A_units[0,:,yy+1]>0.5)[0][0])
        second_dif = np.abs(np.argwhere(A_units[0,:,yy+1]>0.5)[0][0] - np.argwhere(A_units[0,:,yy+2]>0.5)[0][0])

        ratio = second_dif / first_dif
        tot_ratio = tot_ratio + ratio

    record_ratio[u]=tot_ratio



plt.figure(figsize=(10,6))
plt.scatter(gain_to_G,record_ratio,s=350,color='k',marker='o',facecolors="none")
plt.ylim(0,15)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlabel('Mulitplicative input value (ρ)', fontsize=22)
plt.ylabel('Sum of ratios', fontsize=22)
plt.title('Temporal rescaling maintains structure',fontsize=30,y=1.05)
plt.savefig('relative_rescaling.png', dpi=500, bbox_inches='tight')
plt.show()
