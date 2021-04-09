#Code written by Cristian Buc Calderon
#This code implements a RNN connected to a motor node
#The weights from RNN to motor read out the reservor dynamics
#and learn to produce an action at a specific point in times
#using reward-modulated hebbian learning

##to do list
#maybe in how strong the feedback weights!

import numpy as np, matplotlib.pyplot as plt, scipy as sc, os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy


os.chdir('C:/Cris/VoT Project/Feedback-Driven-Self-Organization/Synaptic Depression')

###Intialization parameters
show_pic         = 1
save_pic         = 0
N_E_rnn          = 10
N_motor          = 1
dt               = 1
time             = 1000
tau_E            = 1
tau_sd           = 0.05
tau_I            = 1/20
tau_G            = 1/1000
tau_A            = 1/10
tau_N            = 1/10
tau_w            = 1/3
beta_sd          = 0.1
N_trials         = 500
threshold        = 0.35
k_w              = 0.01
synaptic_dep     = 0
k_a              = 1
k_g              = 1
k_n              = 0
k_rnn_e          = 1
k_rnn_i          = 1
k_eli_rnn        = 0
bias             = 0.4
w_max            = 1
w_rnn_max        = 1
w_go_max         = 1
confidence       = 0.01
alpha_rnn_1      = 0.01 #0.01
alpha_rnn_2      = 0.1
alpha_go_1       = 0.0002
alpha_go_2       = 0.4
threshold        = 0.5
eta_timing       = 0.4 #0.4


### input function
Amp         = 0.5
inputs      = np.ones(time)*Amp


###opening storing arrays
store_error    = np.zeros(N_trials)
store_times    = np.zeros(N_trials)


### Functions used in simulation below
def initialize_weights():

    ###RNN weights
    W_rnn_E_E = 0

    ###RNN to Go-BG neurons
    W_go = np.random.normal(0.5,0.1)

    ###Go to Action
    W_a = np.random.normal(1,0.1)

    ### Action to RNN
    W_feedback = 1

    ###Action to N
    W_n = 1

    ###N to G
    W_inh = 1

    ###N to RNN
    W_rnn_N_E = 2

    #inhibition pool weights
    jEI = 0.5
    jIE = 1

    return W_rnn_E_E,jEI,jIE,W_go,W_a,W_n,W_inh,W_feedback,W_rnn_N_E


lamb = 10
def tanh_f_s(x,lamb):
    z = ((2/(1 + np.exp(-lamb*x)))-1)
    if z < 0:
        z = 0
    return z

### intializing all weight matrices
W_rnn_E_E,jEI,jIE,W_go,W_a,W_n,W_inh,W_feedback,W_rnn_N_E = initialize_weights()


desired_time  = [200,400,600,800]
nb_sim        = 100
store_RT_sd   = np.zeros((nb_sim,len(desired_time)))
store_RT_mean = np.zeros((nb_sim,len(desired_time)))

for ti in range(len(desired_time)):

    for sim in range(nb_sim):
        print(ti+sim+1)

        store_times = []
        temp_times  = []
        count       = 0
        ssd         = 0
        learning    = 1
        dum2        = 0

        for i in range(N_trials):

            ### initial state
            rnn_E          = 0
            rnn_I          = 1. #1.
            G              = 0
            A              = 0
            N              = 0
            dum1           = 0
            rbar           = 0
            temp_error     = 1
            RT             = 1


            ### dynamics loop
            for j in range(time):

                ### rnn activity at time t-1
                rprev = copy.deepcopy(rnn_E)

                ### RNN layer
                #rnn_i_exc = np.sign((np.dot(sd,rnn_E))*jIE)*np.tanh((np.dot(sd,rnn_E))*jIE)
                lamb = 10
                rnn_i_exc = (jIE*rnn_E) + np.random.normal(0,ssd)
                rnn_I    += (-k_rnn_i*rnn_I + (tanh_f_s(rnn_i_exc,lamb))) * tau_I

                rnn_e_exc = (W_rnn_E_E*rnn_E) - (jEI*rnn_I) + (inputs[j]) + (W_feedback*A) - (W_rnn_N_E*N) + np.random.normal(0,ssd)
                rnn_E    += (-k_rnn_e*rnn_E + tanh_f_s(rnn_e_exc,lamb)) * tau_E

                ### G layer
                g_exc = (W_go*rnn_E) - (W_inh*N) + (np.random.normal(0,ssd))
                G    += (-k_g*G + g_exc) * tau_G
                if G < 0:
                    G = 0


                ### A layer
                lamb = 10000
                a_exc = (W_a*G) - bias
                A    += (-k_a*A + tanh_f_s(a_exc,lamb)) * tau_A
                if A < 0:
                    A = 0

                ### N Layer
                n_exc = (W_n*A) + (np.random.normal(0,ssd))
                N    += (-k_n*N + n_exc) * tau_N
                if N < 0:
                    N = 0


                ### checking if each action crossed the threshold
                if (A > threshold)  and (dum1 == 0):
                    temp_error = (j - desired_time[ti])/time
                    RT         = j
                    dum1       = 1


                ### weight update in RNN and between RNN and G units
                rbar      += (-rbar + rprev)*tau_w
                W_rnn_E_E += (-alpha_rnn_1*((1-rnn_E)*rbar)) + (alpha_rnn_2*(rnn_E*rbar)*(w_rnn_max-W_rnn_E_E))
                W_go      += (-alpha_go_1*(1-G*rnn_E)) + (alpha_go_2*(G*rnn_E)*(w_go_max-W_go))


            W_a += (eta_timing*temp_error)

            if np.abs(temp_error) == 0 and dum2 == 0:
                dum2 = 1
                ssd = 0.05 #change this to 0.01 for panel left in figure 6 of the paper

            if dum2 == 1:
                if RT > 1:
                    store_times.append(RT)

        store_RT_sd[sim,ti]   = np.std(store_times)
        store_RT_mean[sim,ti] = np.mean(store_times)

#### figures stuff
# Generate data...
plt.figure(figsize=(7,4))
for jj in range(4):
    colors = (0.5,0,(jj/5)+0.2)
    for ii in range(nb_sim):
        plt.scatter(jj+np.random.uniform(-0.1,0.1),store_RT_sd[ii,jj],color=colors)
x=[0,1,2,3]
plt.xticks(x,('200', '400', '600', '800'))
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.xlabel('Desired action time (ms)', fontsize=20)
plt.ylabel('Standard deviation (ms)', fontsize=20)
if ssd == 0.05:
    plt.title('Scalar variability (σ = 0.05)',fontsize=25,y=1.05)
elif ssd == 0.01:
    plt.title('Scalar variability (σ = 0.01)',fontsize=25,y=1.05)    
plt.show()
