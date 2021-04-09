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


os.chdir('C:\Cris\VoT Project\Feedback-Driven-Self-Organization\Cris way\Video - thunderstruck') #change to state space

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
gain_to_N = 1

#increase last weight; this is a technical reason because their is no feedback
#from the last action to the RNN; this is to overcome the weight max bound from
#rnn to BG and allows the last action to have higher activity


if apply_new_tempo == 1:
    new_tempo   = [275,300,450,550,600,850]
    low_bound   = 0
    high_bound  = 50
    rhythm      = np.ones(time)*low_bound
    half_window = 5
    for yy in range(len(new_tempo)):
        rhythm[new_tempo[yy]-half_window:new_tempo[yy]+half_window] = high_bound
elif apply_new_tempo == 0:
    rhythm      = np.ones(time)



for i in range(N_trials):

    ### initial state
    rnn_E          = np.zeros(N_E_rnn)
    rnn_I          = 0. #1.
    G              = np.zeros(N_motor)
    A              = np.zeros(N_motor)
    N              = np.zeros(N_motor)

#    if i > 0:
#        gain_to_G      += 0.7

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


N_simulations   = 200
store_noisy_rnn = np.zeros ((time,N_E_rnn,N_simulations))
sd = 0.05

for sim in range(N_simulations):
    noise                    = np.random.normal(0,sd,(time,N_E_rnn))
    noisy_rnn                = RNN_units + noise
    noisy_rnn[noisy_rnn<0]   = 0
    noisy_rnn[noisy_rnn>1]   = 1
    store_noisy_rnn[:,:,sim] = noisy_rnn


from sklearn.decomposition import PCA
N_pca = 3
pca = PCA(n_components=N_pca)

store_pca = np.zeros((time,N_pca,N_simulations))
for u in range(N_simulations):
    principalComponents = pca.fit_transform(store_noisy_rnn[:,:,0])
    store_pca[:,:,u] = principalComponents

pca_reshaped = store_pca.reshape(store_pca.shape[0], -1)
np.savetxt("pca.txt", pca_reshaped)
A_units_reshaped = A_units[0,:,:]
np.savetxt("A_units.txt", A_units_reshaped)



asp = 15
import matplotlib.animation as animation
fig, ax = plt.subplots()
ax.set_xlabel('time (a.u.)', fontsize=20)
ax.set_ylabel('Units', fontsize=20)
ax.tick_params(labelsize=20)
ax.set_title('RNN dynamics - Thunderstruck', fontsize=25, pad=20)

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
mask = np.zeros((N_E_rnn,time))
ims = []
for i in range(time):
    matrix=RNN_units[0,:,:].T*mask
    im = ax.imshow(matrix,cmap='plasma',vmin=0, vmax=np.max(RNN_units), interpolation='nearest',aspect=asp ,animated=True)
    ims.append([im])
    mask[:,i] = 1

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
writer = animation.FFMpegWriter(fps=100, metadata=dict(artist='Me'), bitrate=1800)
ani.save("full_rnn.mp4", writer=writer)
