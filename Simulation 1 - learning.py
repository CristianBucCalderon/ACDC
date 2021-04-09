#Written by Cristian Buc Calderon
#Code for Simulation 1 of paper:
#Thunderstruck: the ACDC model of flexible sequences and rhythms in recurrent neural circuits
#more information available here: https://www.biorxiv.org/content/10.1101/2021.04.07.438842v1


import numpy as np, matplotlib.pyplot as plt, scipy as sc, os, shutil
from scipy.sparse import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy
import random as rd


os.chdir('C:/Cris/VoT Project/Feedback-Driven-Self-Organization/Cris way') #set this to the directory containung Simulation 1.py

###Intialization parameters
show_pic         = 1 #set this to 0 to not show pictures
save_pic         = 0
save_weights     = 1 #this has to be set to one in order to save the weights
N_E_rnn          = 200
N_motor          = 6
dt               = 1
time             = 1000
tau_E            = 1
tau_sd           = 0.05
tau_I            = 1/1
tau_G            = 1/1000
tau_A            = 1/10
tau_N            = 1/10
tau_w            = 1/2
beta_sd          = 0.1
N_trials         = 1700
times            = [200,250,400,700,750,900]
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
w_go_max         = 1/20
confidence       = 0.01
alpha_rnn_1      = 0.01
alpha_rnn_2      = 0.1
alpha_go_1       = 0.00002
alpha_go_2       = 0.4
threshold        = 0.5
eta_timing       = 0.4
gain_A_I         = 21
gain_A_RNN       = 21.4
teta             = 0 #you can set these tetas to values > 0 if you want a threshold on the Hebbian learning rule (see below)
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
        W_a[i,i] = np.random.normal(2,0.2)
    W_a[W_a<0] = 0.01

    ###Action to N
    W_n = np.zeros((N_motor,N_motor))
    n = 1
    np.fill_diagonal(W_n, n)

    ###A to RNN
    W_feedback = np.zeros((N_E_rnn,N_motor))
    vec = np.arange(0,N_E_rnn,1)
    rd.shuffle(vec)
    start = 20
    cutoff = 40
    feed_weight_val = 1
    for u in range(N_motor):
        temp = vec[start:cutoff]
        start += 20
        cutoff += 20
        for h in range(len(temp)):
            W_feedback[temp[h],u] = feed_weight_val

    ### input function
    Amp            = 1
    temp           = vec[0:20]
    time           = 1000
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
flag1 = 0
flag2 = 0
flag3 = 0
flag4 = 0
flag5 = 0
flag6 = 0
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
            acti_filter_motor[:,5] = 1

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

    #storing temp error
    cached_error[0] = (np.argmax(A_units[i,:,0]) - times[0])/time
    cached_error[1] = (np.argmax(A_units[i,:,1]) - times[1])/time
    cached_error[2] = (np.argmax(A_units[i,:,2]) - times[2])/time
    cached_error[3] = (np.argmax(A_units[i,:,3]) - times[3])/time
    cached_error[4] = (np.argmax(A_units[i,:,4]) - times[4])/time
    cached_error[5] = (np.argmax(A_units[i,:,5]) - times[5])/time

    if np.max(A_units[i,:,0]) < 0.01:
        cached_error[0] = (1000 - times[0])/time
    if np.max(A_units[i,:,1]) < 0.01:
        cached_error[1] = (1000 - times[1])/time
    if np.max(A_units[i,:,2]) < 0.01:
        cached_error[2] = (1000 - times[2])/time
    if np.max(A_units[i,:,3]) < 0.01:
        cached_error[3] = (1000 - times[3])/time
    if np.max(A_units[i,:,4]) < 0.01:
        cached_error[4] = (1000 - times[4])/time
    if np.max(A_units[i,:,5]) < 0.01:
        cached_error[5] = (1000 - times[5])/time

    store_error[i,:] = cached_error
    print(temp_error)

    ## Updating G-A weights to learn action timing

    if (dum1 == 1) or (dum1 == 0):
        loc = 0
        if dum1 == 1:
            error = temp_error[loc]
        elif dum1 == 0:
            error = 1
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum2 == 1) or (dum2 == 0) and (flag1 == 1):
        loc = 1
        if dum2 == 1:
            error = temp_error[loc]
        elif dum2 == 0:
            error = 1
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum3 == 1) or (dum3 == 0) and (flag2 == 1):
        loc = 2
        if dum3 == 1:
            error = temp_error[loc]
        elif dum3 == 0:
            error = 1
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum4 == 1) or (dum4 == 0) and (flag3 == 1):
        loc = 3
        if dum4 == 1:
            error = temp_error[loc]
        elif dum4 == 0:
            error = 1
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum5 == 1) or (dum5 == 0) and (flag4 == 1):
        loc = 4
        if dum5 == 1:
            error = temp_error[loc]
        elif dum5 == 0:
            error = 1
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0

    if (dum6 == 1) or (dum6 == 0) and (flag5 == 1):
        loc = 5
        if dum6 == 1:
            error = temp_error[loc]
        elif dum6 == 0:
            error = 1
        W_a[loc,loc] += (eta_timing*error)
        W_a[W_a<0] = 0


    ### storing the weights
    W_readout[i+1,:,:]      = W_a
    W_state_action[i+1,:,:] = W_go
    W_rnn[i+1,:,:]          = W_rnn_E_E

### figures stuff
if show_pic == 1:

    ###plot RNN dynamics
    asp = 2
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    axins = inset_axes(ax,
                       width="5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.01, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    ax.set_xlabel('time (a.u.)', fontsize=30)
    ax.set_ylabel('Units', fontsize=30)
    ax.tick_params(labelsize=25)
    ax.set_title('RNN dynamics', fontsize=35, pad=20)
    im = ax.imshow(RNN_units[i-jj,:,:].T, cmap='plasma',vmin=0, vmax=np.max(RNN_units), interpolation='nearest',aspect=asp)
    cbar = fig.colorbar(im, cax=axins,)
    cbar.ax.tick_params(labelsize=20)
    if save_pic == 1:
        plt.savefig('RNN_dynamics.png', dpi=500)
    plt.show()

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    axins = inset_axes(ax,
                       width="5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.01, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    ax.set_xlabel('Sending units', fontsize=20)
    ax.set_ylabel('Receiving units', fontsize=20)
    ax.tick_params(labelsize=25)
    ax.set_title('RNN Connectivity Matrix', fontsize=25,y=1.05)
    im = ax.imshow(W_rnn_E_E, cmap=plt.cm.viridis,vmin=np.min(W_rnn_E_E), vmax=np.max(W_rnn_E_E))
    cbar = fig.colorbar(im, cax=axins,)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(labelsize=15, bottom=False)
    ax.xaxis.set_ticks_position('bottom')
    if save_pic == 1:
        plt.savefig('learned_connectivity.png', dpi=500, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    axins = inset_axes(ax,
                       width="5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.01, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    ax.set_xlabel('Sending units', fontsize=20)
    ax.set_ylabel('Receiving units', fontsize=20)
    ax.tick_params(labelsize=25)
    ax.set_title('RNN Connectivity Matrix | Pre-learning', fontsize=25,y=1.05)
    im = ax.imshow(W_rnn[0,:,:], cmap=plt.cm.viridis,vmin=np.min(W_rnn[0,:,:]), vmax=np.max(W_rnn[1,:,:]))
    cbar = fig.colorbar(im, cax=axins,)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(labelsize=15, bottom=False)
    ax.xaxis.set_ticks_position('bottom')
    if save_pic == 1:
        plt.savefig('learned_connectivity.png', dpi=500, bbox_inches='tight')
    plt.show()


    fig = plt.figure()
    plt.title('G-A Matrix | Pre-learning', fontsize=20)
    ax  = fig.add_subplot(111)
    cax = ax.matshow(W_readout[0,:,:], cmap=plt.cm.viridis,vmin=0, vmax=12)
    cbar = fig.colorbar(cax, pad=0.02)
    ax.set_xlabel('G Nodes', fontsize=20)
    ax.set_ylabel('A Nodes', fontsize=20)
    ax.tick_params(labelsize=15, bottom=False)
    ax.xaxis.set_ticks_position('bottom')
    if save_pic == 1:
        plt.savefig('learned_connectivity.png', dpi=500, bbox_inches='tight')
    plt.show()

    fig = plt.figure()
    plt.title('G-A Connectivity Matrix | Post-learning', fontsize=20)
    ax  = fig.add_subplot(111)
    cax = ax.matshow(W_readout[N_trials-1,:,:], cmap=plt.cm.viridis,vmin=np.min(W_readout[N_trials-1,:,:]), vmax=np.max(W_readout[N_trials-1,:,:]))
    cbar = fig.colorbar(cax, pad=0.02)
    ax.set_xlabel('Sending units', fontsize=20)
    ax.set_ylabel('Receiving units', fontsize=20)
    ax.tick_params(labelsize=15, bottom=False)
    ax.xaxis.set_ticks_position('bottom')
    if save_pic == 1:
        plt.savefig('learned_connectivity.png', dpi=500, bbox_inches='tight')
    plt.show()

    asp=10
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    axins = inset_axes(ax,
                       width="5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.01, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    ax.set_xlabel('Sending units', fontsize=20)
    ax.set_ylabel('Receiving units', fontsize=20)
    ax.tick_params(labelsize=25)
    ax.set_title('RNN-Go Connectivity Matrix | Pre-learning', fontsize=25,y=1.05)
    im = ax.imshow(W_state_action[0,:,:], cmap=plt.cm.viridis,vmin=0, vmax=np.max(W_state_action[0,:,:]),aspect=asp)
    cbar = fig.colorbar(im, cax=axins,)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(labelsize=15, bottom=False)
    ax.xaxis.set_ticks_position('bottom')
    if save_pic == 1:
        plt.savefig('learned_connectivity.png', dpi=500, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(1,1,figsize=(10,10))
    axins = inset_axes(ax,
                       width="5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.01, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    ax.set_xlabel('Sending units', fontsize=20)
    ax.set_ylabel('Receiving units', fontsize=20)
    ax.tick_params(labelsize=25)
    ax.set_title('RNN-Go Connectivity Matrix | Post-learning', fontsize=25,y=1.05)
    im = ax.imshow(W_state_action[N_trials-1,:,:], cmap=plt.cm.viridis,vmin=0, vmax=np.max(W_state_action[N_trials-1,:,:]),aspect=asp)
    cbar = fig.colorbar(im, cax=axins,)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(labelsize=15, bottom=False)
    ax.xaxis.set_ticks_position('bottom')
    if save_pic == 1:
        plt.savefig('learned_connectivity.png', dpi=500, bbox_inches='tight')
    plt.show()


    fig, ax = plt.subplots(1,1,figsize=(5,5))
    axins = inset_axes(ax,
                       width="5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.01, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    ax.set_xlabel('Sending units', fontsize=20)
    ax.set_ylabel('Receiving units', fontsize=20)
    ax.tick_params(labelsize=25)
    ax.set_title('RNN Connectivity Matrix | Post-learning', fontsize=25,y=1.05)
    im = ax.imshow(W_rnn[N_trials-1,:,:], cmap=plt.cm.viridis,vmin=np.min(W_rnn[N_trials-1,:,:]), vmax=np.max(W_rnn[N_trials-1,:,:]))
    cbar = fig.colorbar(im, cax=axins,)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(labelsize=15, bottom=False)
    ax.xaxis.set_ticks_position('bottom')
    if save_pic == 1:
        plt.savefig('learned_connectivity.png', dpi=500, bbox_inches='tight')
    plt.show()

    asp=0.01
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    axins = inset_axes(ax,
                       width="5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.01, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    ax.set_xlabel('Sending units', fontsize=20)
    ax.set_ylabel('Receiving units', fontsize=20)
    ax.tick_params(labelsize=25)
    ax.set_title('A-RNN Connectivity Matrix', fontsize=25,y=1.05)
    im = ax.imshow(W_feedback, cmap=plt.cm.viridis,vmin=np.min(W_feedback), vmax=np.max(W_feedback), aspect=asp)
    cbar = fig.colorbar(im, cax=axins,)
    cbar.ax.tick_params(labelsize=20)
    ax.tick_params(labelsize=15, bottom=False)
    ax.xaxis.set_ticks_position('bottom')
    if save_pic == 1:
        plt.savefig('learned_connectivity.png', dpi=500, bbox_inches='tight')
    plt.show()

    x_axis = np.arange(1,time+2,1)

    fig = plt.figure()
    plt.title('G units',fontsize=30)
    plt.plot(x_axis,G_units[N_trials-1,:,0],'b',label="action 1",linewidth=3)
    plt.plot(x_axis,G_units[N_trials-1,:,1],'r',label="action 2",linewidth=3)
    plt.plot(x_axis,G_units[N_trials-1,:,2],'m',label="action 3",linewidth=3)
    plt.plot(x_axis,G_units[N_trials-1,:,3],'g',label="action 4",linewidth=3)
    plt.plot(x_axis,G_units[N_trials-1,:,4],'c',label="action 5",linewidth=3)
    plt.plot(x_axis,G_units[N_trials-1,:,5],'y',label="action 6",linewidth=3)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Activation', fontsize=20)
    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.title('A units',fontsize=30)
    plt.plot(x_axis,A_units[N_trials-1,:,0],'b',label="action 1",linewidth=3)
    plt.plot(x_axis,A_units[N_trials-1,:,1],'r',label="action 2",linewidth=3)
    plt.plot(x_axis,A_units[N_trials-1,:,2],'m',label="action 3",linewidth=3)
    plt.plot(x_axis,A_units[N_trials-1,:,3],'g',label="action 4",linewidth=3)
    plt.plot(x_axis,A_units[N_trials-1,:,4],'c',label="action 5",linewidth=3)
    plt.plot(x_axis,A_units[N_trials-1,:,5],'y',label="action 6",linewidth=3)
    plt.axvline(x=times[0]+1, linestyle='dashed', color='b',linewidth=2)
    plt.axvline(x=times[1]+1, linestyle='dashed', color='r',linewidth=2)
    plt.axvline(x=times[2]+1, linestyle='dashed', color='m',linewidth=2)
    plt.axvline(x=times[3]+1, linestyle='dashed', color='g',linewidth=2)
    plt.axvline(x=times[4]+1, linestyle='dashed', color='c',linewidth=2)
    plt.axvline(x=times[5]+1, linestyle='dashed', color='y',linewidth=2)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Activation', fontsize=20)
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(18,5))
    fig.tight_layout()
    ax1 = fig.add_subplot(2,3,1)
    ax2 = fig.add_subplot(2,3,2)
    ax3 = fig.add_subplot(2,3,3)
    ax4 = fig.add_subplot(2,3,4)
    ax5 = fig.add_subplot(2,3,5)
    ax6 = fig.add_subplot(2,3,6)

    for ii in range(N_motor):
        if store_error[0,ii] > 0:
            vec=np.where(store_error[:,ii]>0.005)
        elif store_error[0,ii] < 0:
            vec=np.where(store_error[:,ii]<-0.005)

        pos=np.max(vec)
        colors = np.linspace(0.1,1,pos)
        if ii == 0:
            for j in range(len(colors)):
                vec1 = A_units[j,:,ii]
                ax1.plot(vec1,color=(0,0,colors[j]),linewidth=2)
        elif ii == 1:
            for j in range(len(colors)):
                vec1 = A_units[j,:,ii]
                ax2.plot(vec1,color=(colors[j],0,0),linewidth=2)
        elif ii == 2:
            for j in range(len(colors)):
                vec1 = A_units[j,:,ii]
                ax3.plot(vec1,color=(colors[j],0,colors[j]),linewidth=2)
        elif ii == 3:
            for j in range(len(colors)):
                vec1 = A_units[j,:,ii]
                ax4.plot(vec1,color=(0,colors[j],0),linewidth=2)
        elif ii == 4:
            for j in range(len(colors)):
                vec1 = A_units[j,:,ii]
                ax5.plot(vec1,color=(0,colors[j],colors[j]),linewidth=2)
        elif ii == 5:
            for j in range(len(colors)):
                vec1 = A_units[j,:,ii]
                ax6.plot(vec1,color=(colors[j],colors[j],0),linewidth=2)

    #ax1.set_ylabel('A units activation', fontsize=15)
    #ax4.set_ylabel('A units activation', fontsize=15)
    fig.text(-0.01, 0.5, 'A nodes activation',fontsize=20, va='center', rotation='vertical')
    ax2.set_title('Action timing learning',fontsize=25,y=1.05)
    ax5.set_xlabel('Time', fontsize=20)
    ax1.axvline(x=times[0], linestyle='dashed', color='b')
    ax1.axvline(x=times[1], linestyle='dashed', color='r')
    ax1.axvline(x=times[2], linestyle='dashed', color='m')
    ax1.axvline(x=times[3], linestyle='dashed', color='g')
    ax1.axvline(x=times[4], linestyle='dashed', color='c')
    ax1.axvline(x=times[5], linestyle='dashed', color='y')
    ax2.axvline(x=times[0], linestyle='dashed', color='b')
    ax2.axvline(x=times[1], linestyle='dashed', color='r')
    ax2.axvline(x=times[2], linestyle='dashed', color='m')
    ax2.axvline(x=times[3], linestyle='dashed', color='g')
    ax2.axvline(x=times[4], linestyle='dashed', color='c')
    ax2.axvline(x=times[5], linestyle='dashed', color='y')
    ax3.axvline(x=times[0], linestyle='dashed', color='b')
    ax3.axvline(x=times[1], linestyle='dashed', color='r')
    ax3.axvline(x=times[2], linestyle='dashed', color='m')
    ax3.axvline(x=times[3], linestyle='dashed', color='g')
    ax3.axvline(x=times[4], linestyle='dashed', color='c')
    ax3.axvline(x=times[5], linestyle='dashed', color='y')
    ax4.axvline(x=times[0], linestyle='dashed', color='b')
    ax4.axvline(x=times[1], linestyle='dashed', color='r')
    ax4.axvline(x=times[2], linestyle='dashed', color='m')
    ax4.axvline(x=times[3], linestyle='dashed', color='g')
    ax4.axvline(x=times[4], linestyle='dashed', color='c')
    ax4.axvline(x=times[5], linestyle='dashed', color='y')
    ax5.axvline(x=times[0], linestyle='dashed', color='b')
    ax5.axvline(x=times[1], linestyle='dashed', color='r')
    ax5.axvline(x=times[2], linestyle='dashed', color='m')
    ax5.axvline(x=times[3], linestyle='dashed', color='g')
    ax5.axvline(x=times[4], linestyle='dashed', color='c')
    ax5.axvline(x=times[5], linestyle='dashed', color='y')
    ax6.axvline(x=times[0], linestyle='dashed', color='b')
    ax6.axvline(x=times[1], linestyle='dashed', color='r')
    ax6.axvline(x=times[2], linestyle='dashed', color='m')
    ax6.axvline(x=times[3], linestyle='dashed', color='g')
    ax6.axvline(x=times[4], linestyle='dashed', color='c')
    ax6.axvline(x=times[5], linestyle='dashed', color='y')
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    ax3.tick_params(labelsize=15)
    ax4.tick_params(labelsize=15)
    ax5.tick_params(labelsize=15)
    ax6.tick_params(labelsize=15)
    plt.tight_layout()
    if save_pic == 1:
        plt.savefig('motor_accumulation.png', dpi=500, bbox_inches='tight')
    plt.show()

    fig = plt.figure()
    plt.title('N units',fontsize=30)
    plt.plot(x_axis,N_units[N_trials-1,:,0],'b',label="action 1")
    plt.plot(x_axis,N_units[N_trials-1,:,1],'r',label="action 2")
    plt.plot(x_axis,N_units[N_trials-1,:,2],'m',label="action 3")
    plt.plot(x_axis,N_units[N_trials-1,:,3],'g',label="action 4")
    plt.plot(x_axis,N_units[N_trials-1,:,4],'c',label="action 5")
    plt.plot(x_axis,N_units[N_trials-1,:,5],'y',label="action 6")
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Activation', fontsize=20)
    plt.legend()
    plt.show()

    fig = plt.figure()
    plt.title('I units',fontsize=30)
    plt.plot(x_axis,I_units[N_trials-1,:],'b',label="I dynamics")
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Activation', fontsize=20)
    plt.legend()
    plt.show()

    x_error = np.arange(1,N_trials+1,1)
    fig = plt.figure()
    plt.title('Learning evolution',fontsize=20)
    plt.plot(x_error,store_error[:,0],'b',label="action 1",linewidth=4)
    plt.plot(x_error,store_error[:,1],'r',label="action 2",linewidth=4)
    plt.plot(x_error,store_error[:,2],'m',label="action 3",linewidth=4)
    plt.plot(x_error,store_error[:,3],'g',label="action 4",linewidth=4)
    plt.plot(x_error,store_error[:,4],'c',label="action 5",linewidth=4)
    plt.plot(x_error,store_error[:,5],'y',label="action 6",linewidth=4)
    plt.xlabel('Trials', fontsize=20)
    plt.ylabel('Error', fontsize=20)
    plt.legend()
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
