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
import wavio
import copy
import random as rd


os.chdir('C:/Cris/VoT Project/Feedback-Driven-Self-Organization/Cris way/thunderstruck - rock - Continued')

###Intialization parameters
show_pic         = 1
save_pic         = 0
apply_new_tempo  = 0
N_E_rnn          = 68
N_motor          = 16
time             = 3800
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
threshold        = 0.5
gain_A_I         = 21
gain_A_RNN       = 21.4
position         = 0
riff             = [0,0]
riff_speed       = 15 #or 20 higher this value the slowest is the spead
riff_stack       = np.zeros((riff_speed,2))
riff_stack       = riff_stack.astype('int32')

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

rhythm      = np.ones(time)


flag         = 0
flag1        = 0
window_start = 0
window_end   = 120
window_size  = 120
position     = 0

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
        g_exc = (np.dot(W_go,rnn_E) - np.dot(W_inh,N) - np.dot(W_g_inh,G))*rhythm[j] #if single gain then gain_to_G
        G    += (-k_g*G + g_exc + (np.random.normal(0,sd_g,N_motor))) * tau_G
        G[G<0]=0

        ### A layer
        lamb = 10000
        a_exc = np.dot(W_a,G) - bias
        A    += (-k_a*A + tanh_f(a_exc,lamb) + np.random.normal(0,sd_a,N_motor)) * tau_A
        A[A<0]=0

        ### N Layer
        n_exc = np.dot(W_n,A) #* gain_to_N[j]
        N    += (-k_n*N + n_exc + (np.random.normal(0,sd_n,N_motor))) * tau_N
        N[N<0]=0

        ### storing dynamics
        RNN_units[i,j,:] = rnn_E
        G_units[i,j+1,:] = G
        A_units[i,j+1,:] = A
        N_units[i,j+1,:] = N
        I_units[i,j+1]   = rnn_I
        test[i,j+1]      = rnn_i_exc

        if flag == 0:
            string_note = 'note'+str(position+1)+'.wav'
            note        = wavio.read(string_note)
            note_data   = note.data
            trim_note   = note_data[0:np.where(note_data!=0)[0][-1]+1]
            flag = 1

        if A[position] > 0.5:
            add_note = trim_note[window_start:window_end,:]
            riff = sc.vstack((riff,add_note))
            window_start += window_size
            window_end   += window_size
            flag1 = 1
        elif A[position] < 0.5:
            riff = sc.vstack((riff,riff_stack))


        if A[position] < 0.5 and flag1 == 1:
            flag      = 0
            flag1     = 0
            if position < 15:
                position += 1



### figures stuff
x_axis = np.arange(1,time+2,1)


fig = plt.figure(figsize=(9,5))
plt.title('Thunderstruck - Rock Tempo',fontsize=25, y=1.05)
plt.plot(x_axis,A_units[N_trials-1,:,0],'b',label="B5",linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,1],'r',label="A5",linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,2],'m',label="G#5",linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,3],'r',linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,4],'m',linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,5],'y',label="F#5",linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,6],'m',linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,7],'c',label="E5",linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,8],'y',linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,9],'g',label="D#5",linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,10],'c',linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,11],'g',linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,12],'c',linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,13],'g',linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,14],'c',linewidth=3)
plt.plot(x_axis,A_units[N_trials-1,:,15],'g',linewidth=3)
plt.xlabel('Time (ms)', fontsize=23)
plt.ylabel('A nodes activation', fontsize=23)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.savefig('Thunderstruck rock tempo.png', dpi=500, bbox_inches='tight')
plt.show()

#record the sound played by the model
rate = 45000
wavio.write("thunderstruck_rock.wav",riff,rate)
