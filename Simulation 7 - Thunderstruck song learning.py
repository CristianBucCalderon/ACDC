#Code written by Cristian Buc Calderon
#This code implements a RNN connected to a motor node
#The weights from RNN to motor read out the reservor dynamics
#and learn to produce an action at a specific point in times
#using reward-modulated hebbian learning

##to do list
#maybe in how strong the feedback weights!

import numpy as np, matplotlib.pyplot as plt, scipy as sc, os, shutil
from scipy.sparse import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy
import random as rd


os.chdir('C:/Cris/VoT Project/Feedback-Driven-Self-Organization/Cris way')

###Intialization parameters
show_pic         = 1
save_weights     = 1
N_E_rnn          = 68
N_motor          = 16
dt               = 1
time             = 3800
tau_E            = 1
tau_sd           = 0.05
tau_I            = 1/1
tau_G            = 1/1000
tau_A            = 1/10
tau_N            = 1/10
tau_w            = 1/2
beta_sd          = 0.1
N_trials         = 1200
times            = [210,420,640,850,1070,1280,1500,1710,1920,2140,2350,2570,2780,3000,3210,3420]
k_eli            = 0
k_w              = 0.01
synaptic_dep     = 0
k_a              = 1
k_g              = 1
k_n              = 0
k_rnn_e          = 1
k_rnn_i          = 1
bias             = 0.4
w_max            = 1
w_rnn_max        = 1
w_go_max         = 1/2
confidence       = 0.01
alpha_rnn_1      = 0.01
alpha_rnn_2      = 0.1
alpha_go_1       = 0.00002 #0.00002
alpha_go_2       = 0.4
threshold        = 0.5
eta_timing       = 0.4 #0.4
gain_A_I         = 21
gain_A_RNN       = 21.4
teta             = 0
teta1            = 0

###opening storing arrays
RNN_units      = np.zeros((N_trials,time,N_E_rnn))
rbar_units     = np.zeros((N_trials,time,N_E_rnn))
sd_units       = np.zeros((N_trials,time,N_E_rnn))
G_units        = np.zeros((N_trials,time+1,N_motor))
A_units        = np.zeros((N_trials,time+1,N_motor))
N_units        = np.zeros((N_trials,time+1,N_motor))
I_units        = np.zeros((N_trials,time+1))
W_readout      = np.zeros((N_trials+1,N_motor,N_motor))
W_state_action = np.zeros((N_trials+1,N_motor,N_E_rnn))
W_rnn          = np.zeros((N_trials+1,N_E_rnn,N_E_rnn))
store_error    = np.ones((N_trials,N_motor))

### Functions used in simulation below
def initialize_weights(N_E_rnn,N_motor):

    ###RNN weights
    W_rnn_E_E = np.zeros((N_E_rnn,N_E_rnn))

    ###RNN to Go-BG neurons
    W_go         = np.random.normal((0.5/N_E_rnn),(0.1/N_E_rnn),(N_motor,N_E_rnn))
    W_go[W_go<0] = 0

    ###Go to Action
    W_a = np.zeros((N_motor,N_motor))
    for i in range(N_motor):
        W_a[i,i] = np.random.normal(0.5,0.0)
    W_a[W_a<0] = 0.01

    ###Action to N
    W_n = np.zeros((N_motor,N_motor))
    n = 1
    np.fill_diagonal(W_n, n)

    ###A to RNN
    W_feedback = np.zeros((N_E_rnn,N_motor))
    vec = np.arange(0,N_E_rnn,1)
    rd.shuffle(vec)
    start = 4
    cutoff = 8
    feed_weight_val = 1
    for u in range(N_motor):
        temp = vec[start:cutoff]
        start += 4
        cutoff += 4
        for h in range(len(temp)):
            W_feedback[temp[h],u] = feed_weight_val

    ### input function
    Amp            = 1
    temp           = vec[0:3]
    time           = 3800
    inputs         = np.zeros((N_E_rnn,time))
    inputs[temp,0:20] = Amp

    ###N to G
    W_inh = np.zeros((N_motor,N_motor))
    inh = 1 #make sure this maintains off!
    np.fill_diagonal(W_inh, inh)

    ### A to I
    force_A = 1
    W_shut = np.ones((1,N_motor))*force_A

    ### G mutual inhibition
    force = 0 #or 3, or 0
    W_g_inh = np.ones((N_motor,N_motor))*force
    np.fill_diagonal(W_g_inh, 0)

    #inhibition pool weights
    force_inh = 1
    force_act = 0.1
    jEI = np.ones(N_E_rnn)*force_inh
    jIE = np.ones(N_E_rnn)*force_act

    return W_rnn_E_E,jEI,jIE,W_go,W_a,W_n,W_inh,W_g_inh,W_feedback,W_shut,inputs


lamb = 10
def tanh_f(d1,lamb):
    z1 = ((2/(1 + np.exp(-lamb*d1)))-1)
    z1[z1<0] = 0
    return z1

def tanh_f_s(d2,lamb):
    z2 = ((2/(1 + np.exp(-lamb*d2)))-1)
    if z2 < 0:
        z2 = 0
    return z2

def tanh_eli(d3):
    lambd = 10
    d= 2.5
    z3 = d/(1 + np.exp((-lambd*d3) + 5))
    z3[z3<0.02] = 0
    return z3

def fi_trans(z4,threshold1):
    z4[z4<threshold1] = 0
    return z4

### intializing all weight matrices
W_rnn_E_E,jEI,jIE,W_go,W_a,W_n,W_inh,W_g_inh,W_feedback,W_shut,inputs = initialize_weights(N_E_rnn,N_motor)

W_readout[0,:,:]      = W_a
W_state_action[0,:,:] = W_go
W_rnn[0,:,:]          = W_rnn_E_E

test1 = copy.deepcopy(W_rnn_E_E)
test2 = copy.deepcopy(W_go)

### (in)activation masks
act_filter_RNN      = np.zeros((N_motor,N_E_rnn))
act_filter_RNN[0,:] = 1
acti_filter_motor   = np.zeros((N_E_rnn,N_motor))
noise_filter        = np.zeros(N_motor)
noise_filter[0]     = 1

### dumy codes
flag1  = 0
flag2  = 0
flag3  = 0
flag4  = 0
flag5  = 0
flag6  = 0
flag7  = 0
flag8  = 0
flag9  = 0
flag10 = 0
flag11 = 0
flag12 = 0
flag13 = 0
flag14 = 0
flag15 = 0
flag16 = 0
temp_error = np.ones(N_motor)
cached_error = np.ones(N_motor)
sd_i   = 0
sd_rnn   = 0
sd_g   = 0
sd_a   = 0
sd_n   = 0

for i in range(N_trials):

    ### initial state
    sd             = np.ones(N_E_rnn)
    rnn_E          = np.zeros(N_E_rnn)
    rnn_I          = 0. #1.
    G              = np.zeros(N_motor)
    A              = np.zeros(N_motor)
    N              = np.zeros(N_motor)
    rbar           = np.zeros(N_E_rnn)

    ### reset dumy codes
    error          = 1
    dum1           = 0
    dum2           = 0
    dum3           = 0
    dum4           = 0
    dum5           = 0
    dum6           = 0
    dum7           = 0
    dum8           = 0
    dum9           = 0
    dum10          = 0
    dum11          = 0
    dum12          = 0
    dum13          = 0
    dum14          = 0
    dum15          = 0
    dum16          = 0

    ### dynamics loop
    for j in range(time):

        ### activate connections depending on the error to induce sequential learning
        if (np.abs(temp_error[0]) < confidence) and (flag1 == 0):
            flag1 = 1
            act_filter_RNN[1,:]    = 1
            acti_filter_motor[:,0] = 1
            noise_filter[1]        = 1
        if (np.abs(temp_error[1]) < confidence) and (flag2 == 0):
            flag2 = 1
            act_filter_RNN[2,:]    = 1
            acti_filter_motor[:,1] = 1
            noise_filter[2]        = 1
        if (np.abs(temp_error[2]) < confidence) and (flag3 == 0):
            flag3 = 1
            act_filter_RNN[3,:]    = 1
            acti_filter_motor[:,2] = 1
            noise_filter[3]        = 1
        if (np.abs(temp_error[3]) < confidence) and (flag4 == 0):
            flag4 = 1
            act_filter_RNN[4,:]    = 1
            acti_filter_motor[:,3] = 1
            noise_filter[4]        = 1
        if (np.abs(temp_error[4]) < confidence) and (flag5 == 0):
            flag5 = 1
            act_filter_RNN[5,:]    = 1
            acti_filter_motor[:,4] = 1
            noise_filter[5]        = 1
        if (np.abs(temp_error[5]) < confidence) and (flag6 == 0):
            flag6 = 1
            act_filter_RNN[6,:]    = 1
            acti_filter_motor[:,5] = 1
            noise_filter[6]        = 1
        if (np.abs(temp_error[6]) < confidence) and (flag7 == 0):
            flag7 = 1
            act_filter_RNN[7,:]    = 1
            acti_filter_motor[:,6] = 1
            noise_filter[7]        = 1
        if (np.abs(temp_error[7]) < confidence) and (flag8 == 0):
            flag8 = 1
            act_filter_RNN[8,:]    = 1
            acti_filter_motor[:,7] = 1
            noise_filter[8]        = 1
        if (np.abs(temp_error[8]) < confidence) and (flag9 == 0):
            flag9 = 1
            act_filter_RNN[9,:]    = 1
            acti_filter_motor[:,8] = 1
            noise_filter[9]        = 1
        if (np.abs(temp_error[9]) < confidence) and (flag10 == 0):
            flag10 = 1
            act_filter_RNN[10,:]    = 1
            acti_filter_motor[:,9] = 1
            noise_filter[10]        = 1
        if (np.abs(temp_error[10]) < confidence) and (flag11 == 0):
            flag11 = 1
            act_filter_RNN[11,:]    = 1
            acti_filter_motor[:,10] = 1
            noise_filter[11]        = 1
        if (np.abs(temp_error[11]) < confidence) and (flag12 == 0):
            flag12 = 1
            act_filter_RNN[12,:]    = 1
            acti_filter_motor[:,11] = 1
            noise_filter[12]        = 1
        if (np.abs(temp_error[12]) < confidence) and (flag13 == 0):
            flag13 = 1
            act_filter_RNN[13,:]    = 1
            acti_filter_motor[:,12] = 1
            noise_filter[13]        = 1
        if (np.abs(temp_error[13]) < confidence) and (flag14 == 0):
            flag14 = 1
            act_filter_RNN[14,:]    = 1
            acti_filter_motor[:,13] = 1
            noise_filter[14]        = 1
        if (np.abs(temp_error[14]) < confidence) and (flag15 == 0):
            flag15 = 1
            act_filter_RNN[15,:]    = 1
            acti_filter_motor[:,14] = 1
            noise_filter[15]        = 1
        if (np.abs(temp_error[15]) < confidence) and (flag16 == 0):
            flag16 = 1
            acti_filter_motor[:,15] = 1


        ### rnn activity at time t-1
        rprev = copy.deepcopy(rnn_E)

        ### RNN layer
        lamb = 10
        rnn_i_exc = np.dot(jIE,rnn_E) + np.dot(W_shut,A*gain_A_I) + np.random.normal(0,sd_i,1)
        rnn_I    += (-k_rnn_i*rnn_I + (rnn_i_exc)) * tau_I

        rnn_e_exc = np.dot(W_rnn_E_E,sd*rnn_E) - (jEI*rnn_I) + (inputs[:,j]) + (np.dot(W_feedback*acti_filter_motor,A*gain_A_RNN))
        rnn_E    += (-k_rnn_e*rnn_E + tanh_f(rnn_e_exc,lamb) + np.random.normal(0,sd_rnn,N_E_rnn)) * tau_E
        rnn_E[rnn_E>1] = 1
        rnn_E[rnn_E<0] = 0

        ### G layer
        g_exc = np.dot(W_go*act_filter_RNN,rnn_E) - np.dot(W_inh,N) - np.dot(W_g_inh,G)
        G    += (-k_g*G + g_exc + (np.random.normal(0,sd_g,N_motor))*noise_filter) * tau_G
        G[G<0]=0

        ### A layer
        lamb = 10000
        a_exc = np.dot(W_a,G) - bias
        A    += (-k_a*A + tanh_f(a_exc,lamb) + np.random.normal(0,sd_a,N_motor)) * tau_A
        A[A<0]=0

        ### N Layer
        n_exc = np.dot(W_n,A)
        N    += (-k_n*N + n_exc + (np.random.normal(0,sd_n,N_motor))*noise_filter) * tau_N
        N[N<0]=0

        ### checking if each action crossed the threshold
        if (A_units[i,j,0] > threshold)  and (dum1 == 0):
            temp_error[0]  = (j - times[0])/time
            dum1 = 1
        if (A_units[i,j,1] > threshold)  and (dum2 == 0) and (dum1 == 1):
            temp_error[1]  = (j - times[1])/time
            dum2 = 1
        if (A_units[i,j,2] > threshold)  and (dum3 == 0) and (dum2 == 1):
            temp_error[2]  = (j - times[2])/time
            dum3 = 1
        if (A_units[i,j,3] > threshold)  and (dum4 == 0) and (dum3 == 1):
            temp_error[3]  = (j - times[3])/time
            dum4 = 1
        if (A_units[i,j,4] > threshold)  and (dum5 == 0) and (dum4 == 1):
            temp_error[4]  = (j - times[4])/time
            dum5 = 1
        if (A_units[i,j,5] > threshold)  and (dum6 == 0) and (dum5 == 1):
            temp_error[5]  = (j - times[5])/time
            dum6 = 1
        if (A_units[i,j,6] > threshold)  and (dum7 == 0) and (dum6 == 1):
            temp_error[6]  = (j - times[6])/time
            dum7 = 1
        if (A_units[i,j,7] > threshold)  and (dum8 == 0) and (dum7 == 1):
            temp_error[7]  = (j - times[7])/time
            dum8 = 1
        if (A_units[i,j,8] > threshold)  and (dum9 == 0) and (dum8 == 1):
            temp_error[8]  = (j - times[8])/time
            dum9 = 1
        if (A_units[i,j,9] > threshold)  and (dum10 == 0) and (dum9 == 1):
            temp_error[9]  = (j - times[9])/time
            dum10 = 1
        if (A_units[i,j,10] > threshold)  and (dum11 == 0) and (dum10 == 1):
            temp_error[10]  = (j - times[10])/time
            dum11 = 1
        if (A_units[i,j,11] > threshold)  and (dum12 == 0) and (dum11 == 1):
            temp_error[11]  = (j - times[11])/time
            dum12 = 1
        if (A_units[i,j,12] > threshold)  and (dum13 == 0) and (dum12 == 1):
            temp_error[12]  = (j - times[12])/time
            dum13 = 1
        if (A_units[i,j,13] > threshold)  and (dum14 == 0) and (dum13 == 1):
            temp_error[13]  = (j - times[13])/time
            dum14 = 1
        if (A_units[i,j,14] > threshold)  and (dum15 == 0) and (dum14 == 1):
            temp_error[14]  = (j - times[14])/time
            dum15 = 1
        if (A_units[i,j,15] > threshold)  and (dum16 == 0) and (dum15 == 1):
            temp_error[15]  = (j - times[15])/time
            dum16 = 1


        ### weight update in RNN and between RNN and G units
        rnn_act11   = copy.deepcopy(rnn_E)
        rnn_act12   = copy.deepcopy(rnn_E)
        rnn_act21   = copy.deepcopy(rnn_E)
        rnn_act22   = copy.deepcopy(rnn_E)
        rbar       += (-rbar + rprev)*tau_w
        rbar1       = copy.deepcopy(rbar)
        rbar2       = copy.deepcopy(rbar)

        W_rnn_E_E += (-alpha_rnn_1*np.outer(1-(fi_trans(rnn_act11,teta)), fi_trans(rbar1,teta1))) + (alpha_rnn_2*np.outer((fi_trans(rnn_act12,teta)), fi_trans(rbar2,teta1))*(w_rnn_max-W_rnn_E_E))
        W_go      += (-alpha_go_1*np.outer(1-G, fi_trans(rnn_act21,teta))) + (alpha_go_2*np.outer(G,fi_trans(rnn_act22,teta))*(w_go_max-W_go))
        W_rnn_E_E[W_rnn_E_E<0] = 0
        W_go[W_go<0]           = 0


        ### storing dynamics
        RNN_units[i,j,:]        = rnn_E
        rbar_units[i,j,:]       = rbar
        sd_units[i,j,:]         = sd
        G_units[i,j+1,:]        = G
        A_units[i,j+1,:]        = A
        N_units[i,j+1,:]        = N
        I_units[i,j+1]          = rnn_I

    cached_error[0]  = (np.argmax(A_units[i,:,0])  - times[0])/time
    cached_error[1]  = (np.argmax(A_units[i,:,1])  - times[1])/time
    cached_error[2]  = (np.argmax(A_units[i,:,2])  - times[2])/time
    cached_error[3]  = (np.argmax(A_units[i,:,3])  - times[3])/time
    cached_error[4]  = (np.argmax(A_units[i,:,4])  - times[4])/time
    cached_error[5]  = (np.argmax(A_units[i,:,5])  - times[5])/time
    cached_error[6]  = (np.argmax(A_units[i,:,6])  - times[6])/time
    cached_error[7]  = (np.argmax(A_units[i,:,7])  - times[7])/time
    cached_error[8]  = (np.argmax(A_units[i,:,8])  - times[8])/time
    cached_error[9]  = (np.argmax(A_units[i,:,9])  - times[9])/time
    cached_error[10] = (np.argmax(A_units[i,:,10]) - times[10])/time
    cached_error[11] = (np.argmax(A_units[i,:,11]) - times[11])/time
    cached_error[12] = (np.argmax(A_units[i,:,12]) - times[12])/time
    cached_error[13] = (np.argmax(A_units[i,:,13]) - times[13])/time
    cached_error[14] = (np.argmax(A_units[i,:,14]) - times[14])/time
    cached_error[15] = (np.argmax(A_units[i,:,15]) - times[15])/time


    if np.max(A_units[i,:,0]) < 0.01:
        cached_error[0] = (time - times[0])/time
    if np.max(A_units[i,:,1]) < 0.01:
        cached_error[1] = (time - times[1])/time
    if np.max(A_units[i,:,2]) < 0.01:
        cached_error[2] = (time - times[2])/time
    if np.max(A_units[i,:,3]) < 0.01:
        cached_error[3] = (time - times[3])/time
    if np.max(A_units[i,:,4]) < 0.01:
        cached_error[4] = (time - times[4])/time
    if np.max(A_units[i,:,5]) < 0.01:
        cached_error[5] = (time - times[5])/time
    if np.max(A_units[i,:,6]) < 0.01:
        cached_error[6] = (time - times[6])/time
    if np.max(A_units[i,:,7]) < 0.01:
        cached_error[7] = (time - times[7])/time
    if np.max(A_units[i,:,8]) < 0.01:
        cached_error[8] = (time - times[8])/time
    if np.max(A_units[i,:,9]) < 0.01:
        cached_error[9] = (time - times[9])/time
    if np.max(A_units[i,:,10]) < 0.01:
        cached_error[10] = (time - times[10])/time
    if np.max(A_units[i,:,11]) < 0.01:
        cached_error[11] = (time - times[11])/time
    if np.max(A_units[i,:,12]) < 0.01:
        cached_error[12] = (time - times[12])/time
    if np.max(A_units[i,:,13]) < 0.01:
        cached_error[13] = (time - times[13])/time
    if np.max(A_units[i,:,14]) < 0.01:
        cached_error[14] = (time - times[14])/time
    if np.max(A_units[i,:,15]) < 0.01:
        cached_error[15] = (time - times[15])/time


    store_error[i,:] = cached_error
    print(temp_error)

    ## Weight update action timing

    if (dum1 == 1) or (dum1 == 0):
        loc = 0
        if dum1 == 1:
            error = temp_error[loc]
        elif dum1 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum2 == 1) or (dum2 == 0) and (flag1 == 1):
        loc = 1
        if dum2 == 1:
            error = temp_error[loc]
        elif dum2 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum3 == 1) or (dum3 == 0) and (flag2 == 1):
        loc = 2
        if dum3 == 1:
            error = temp_error[loc]
        elif dum3 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum4 == 1) or (dum4 == 0) and (flag3 == 1):
        loc = 3
        if dum4 == 1:
            error = temp_error[loc]
        elif dum4 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum5 == 1) or (dum5 == 0) and (flag4 == 1):
        loc = 4
        if dum5 == 1:
            error = temp_error[loc]
        elif dum5 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum6 == 1) or (dum6 == 0) and (flag5 == 1):
        loc = 5
        if dum6 == 1:
            error = temp_error[loc]
        elif dum6 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum7 == 1) or (dum7 == 0) and (flag6 == 1):
        loc = 6
        if dum7 == 1:
            error = temp_error[loc]
        elif dum7 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum8 == 1) or (dum8 == 0) and (flag7 == 1):
        loc = 7
        if dum8 == 1:
            error = temp_error[loc]
        elif dum8 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum9 == 1) or (dum9 == 0) and (flag8 == 1):
        loc = 8
        if dum9 == 1:
            error = temp_error[loc]
        elif dum9 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum10 == 1) or (dum10 == 0) and (flag9 == 1):
        loc = 9
        if dum10 == 1:
            error = temp_error[loc]
        elif dum10 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum11 == 1) or (dum11 == 0) and (flag10 == 1):
        loc = 10
        if dum11 == 1:
            error = temp_error[loc]
        elif dum11 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum12 == 1) or (dum12 == 0) and (flag11 == 1):
        loc = 11
        if dum12 == 1:
            error = temp_error[loc]
        elif dum12 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum13 == 1) or (dum13 == 0) and (flag12 == 1):
        loc = 12
        if dum13 == 1:
            error = temp_error[loc]
        elif dum13 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum14 == 1) or (dum14 == 0) and (flag13 == 1):
        loc = 13
        if dum14 == 1:
            error = temp_error[loc]
        elif dum14 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum15 == 1) or (dum15 == 0) and (flag14 == 1):
        loc = 14
        if dum15 == 1:
            error = temp_error[loc]
        elif dum15 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum16 == 1) or (dum16 == 0) and (flag15 == 1):
        loc = 15
        if dum16 == 1:
            error = temp_error[loc]
        elif dum16 == 0:
            error = cached_error[loc]
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0


    ### storing the weights
    W_readout[i+1,:,:]      = W_a
    W_state_action[i+1,:,:] = W_go
    W_rnn[i+1,:,:]          = W_rnn_E_E

### figures stuff
if show_pic == 1:

    x_axis = np.arange(1,time+2,1)

    fig = plt.figure(figsize=(10,10))
    plt.title('A units',fontsize=30)
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
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Activation', fontsize=20)
    plt.legend()
    plt.savefig('Thunderstruck - A units.png', dpi=500, bbox_inches='tight')
    plt.show()

    x_error = np.arange(1,N_trials+1,1)
    fig = plt.figure(figsize=(10,10))
    plt.title('Learning evolution',fontsize=20)
    plt.plot(x_error,store_error[:,0],'b',label="action 1",linewidth=4)
    plt.plot(x_error,store_error[:,1],'r',label="action 2",linewidth=4)
    plt.plot(x_error,store_error[:,2],'m',label="action 3",linewidth=4)
    plt.plot(x_error,store_error[:,3],'g',label="action 4",linewidth=4)
    plt.plot(x_error,store_error[:,4],'c',label="action 5",linewidth=4)
    plt.plot(x_error,store_error[:,5],'y',label="action 6",linewidth=4)
    plt.plot(x_error,store_error[:,6],'k',label="action 7",linewidth=4)
    plt.plot(x_error,store_error[:,7],'b--',label="action 8",linewidth=4)
    plt.plot(x_error,store_error[:,8],'r--',label="action 9",linewidth=4)
    plt.plot(x_error,store_error[:,9],'m--',label="action 10",linewidth=4)
    plt.plot(x_error,store_error[:,10],'g--',label="action 11",linewidth=4)
    plt.plot(x_error,store_error[:,11],'c--',label="action 12",linewidth=4)
    plt.plot(x_error,store_error[:,12],'y--',label="action 13",linewidth=4)
    plt.plot(x_error,store_error[:,13],'k--',label="action 14",linewidth=4)
    plt.plot(x_error,store_error[:,14],'b--',label="action 15",linewidth=4)
    plt.plot(x_error,store_error[:,15],'r--',label="action 16",linewidth=4)
    plt.xlabel('Trials', fontsize=20)
    plt.ylabel('Error', fontsize=20)
    plt.legend()
    plt.savefig('Thunderstruck - error.png', dpi=500, bbox_inches='tight')
    plt.show()

#saving weights and moving them to the noisy space folder
if save_weights == 1:
    #save the weights
    np.savetxt('RNN_weights.txt', W_rnn_E_E, delimiter=',')
    np.savetxt('W_shut.txt', W_shut, delimiter=',')
    np.savetxt('Feedback_weights.txt', W_feedback, delimiter=',')
    np.savetxt('RNN_Go_weights.txt', W_go, delimiter=',')
    np.savetxt('Go_A_weights.txt', W_a, delimiter=',')
    np.savetxt('A_N_weights.txt', W_n, delimiter=',')
    np.savetxt('G_inh_weights.txt', W_g_inh, delimiter=',')
    np.savetxt('N_G_inh_weights.txt', W_inh, delimiter=',')
    np.savetxt('Inputs_weights.txt', inputs, delimiter=',')
    np.savetxt('I_E_weights.txt', jEI, delimiter=',')
    np.savetxt('E_I_weights.txt', jIE, delimiter=',')
    np.savetxt('Times.txt', times, delimiter=',')
