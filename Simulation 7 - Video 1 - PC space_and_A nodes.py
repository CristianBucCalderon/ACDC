"""
============
3D animation
============

An animated plot in 3D.
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import os

os.chdir('C:\Cris\VoT Project\Feedback-Driven-Self-Organization\Cris way\Video - thunderstruck') #change to state space

# Fixing random state for reproducibility
np.random.seed(19680801)

#loading and reshaping the PCA data
time          = 3800
N_pca         = 3
N_simulations = 200
data_reshaped = np.loadtxt("pca.txt")
Actions       = np.loadtxt("A_units.txt")
times         = np.loadtxt('Times.txt', delimiter=',')
times         = times + 15 
dum           = np.random.rand(time, N_pca, N_simulations)
PCAs          = data_reshaped.reshape(data_reshaped.shape[0], data_reshaped.shape[1] // dum.shape[2], dum.shape[2]) 


#Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800) #1800

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# transpose the PCA data to fit the set_data format
data = PCAs[:,:,0].T
x    = data[0,:]
y    = data[1,:]
z    = data[2,:]

#create the artist that will be updated in the function below
line, = ax.plot(x, y, z, color='b', linewidth = 3)

#color list
color_list = ('b','r','m','y','c','g','k')
def update_line(num,x,y,z,line): 
    line.set_data(x[:num], y[:num])
    line.set_3d_properties(z[:num])
    if num<times[0]:
        line.set_color(color_list[0])
    elif num>times[0] and num<times[1]:
        line.set_color(color_list[1])
    elif num>times[1] and num<times[2]:
        line.set_color(color_list[2])    
    elif num>times[2] and num<times[3]:
        line.set_color(color_list[1])
    elif num>times[3] and num<times[4]:
        line.set_color(color_list[2])
    elif num>times[4] and num<times[5]:
        line.set_color(color_list[3])
    elif num>times[5] and num<times[6]:
        line.set_color(color_list[2])
    elif num>times[6] and num<times[7]:
        line.set_color(color_list[4])
    elif num>times[7] and num<times[8]:
        line.set_color(color_list[3])
    elif num>times[8] and num<times[9]:
        line.set_color(color_list[5])
    elif num>times[9] and num<times[10]:
        line.set_color(color_list[4])
    elif num>times[10] and num<times[11]:
        line.set_color(color_list[5])
    elif num>times[11] and num<times[12]:
        line.set_color(color_list[4])
    elif num>times[12] and num<times[13]:
        line.set_color(color_list[5])
    elif num>times[13] and num<times[14]:
        line.set_color(color_list[4])
    elif num>times[14] and num<times[15]:
        line.set_color(color_list[5])
    elif num>times[15]:
        line.set_color(color_list[6])
    return line
    
# Setting the axes properties
ax.set_xlabel('PC 1',fontsize=15, rotation=-15)
ax.set_ylabel('PC 2',fontsize=15,rotation=50)
ax.set_zlabel('PC 3',fontsize=15,rotation=90)
ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)
ax.set_zlim(-1.5,1.5)
ax.set_title('Neural Trajectory - Thunderstruck',fontsize=18,y=1)
ax.grid(False)

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_line, time, fargs=[x,y,z,line],
                                   interval=1, blit=False)

line_ani.save('RNN_Neural_trajectory.mp4', writer=writer)

### now we save the action animation
x  = np.arange(0,time+1)
y1 = Actions[:,0]
y2 = Actions[:,1]
y3 = Actions[:,2]
y4 = Actions[:,3]
y5 = Actions[:,4]
y6 = Actions[:,5]
y7 = Actions[:,6]
y8 = Actions[:,7]
y9 = Actions[:,8]
y10 = Actions[:,9]
y11 = Actions[:,10]
y12 = Actions[:,11]
y13 = Actions[:,12]
y14 = Actions[:,13]
y15 = Actions[:,14]
y16 = Actions[:,15]

fig, ax = plt.subplots()
line1, = ax.plot(x, y1,color='b', linewidth = 3)
line2, = ax.plot(x, y2,color='r', linewidth = 3)
line3, = ax.plot(x, y3,color='m', linewidth = 3)
line4, = ax.plot(x, y4,color='r', linewidth = 3)
line5, = ax.plot(x, y5,color='m', linewidth = 3)
line6, = ax.plot(x, y6,color='y', linewidth = 3)
line7, = ax.plot(x, y7,color='m', linewidth = 3)
line8, = ax.plot(x, y8,color='c', linewidth = 3)
line9, = ax.plot(x, y9,color='y', linewidth = 3)
line10, = ax.plot(x, y10,color='g', linewidth = 3)
line11, = ax.plot(x, y11,color='c', linewidth = 3)
line12, = ax.plot(x, y12,color='g', linewidth = 3)
line13, = ax.plot(x, y13,color='c', linewidth = 3)
line14, = ax.plot(x, y14,color='g', linewidth = 3)
line15, = ax.plot(x, y15,color='c', linewidth = 3)
line16, = ax.plot(x, y16,color='g', linewidth = 3)

def update_actions(num,x,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12,line13,line14,line15,line16): 
    line1.set_data(x[:num], y1[:num])
    line2.set_data(x[:num], y2[:num])
    line3.set_data(x[:num], y3[:num])
    line4.set_data(x[:num], y4[:num])
    line5.set_data(x[:num], y5[:num])
    line6.set_data(x[:num], y6[:num])
    line7.set_data(x[:num], y7[:num])
    line8.set_data(x[:num], y8[:num])
    line9.set_data(x[:num], y9[:num])
    line10.set_data(x[:num], y10[:num])
    line11.set_data(x[:num], y11[:num])
    line12.set_data(x[:num], y12[:num])
    line13.set_data(x[:num], y13[:num])
    line14.set_data(x[:num], y14[:num])
    line15.set_data(x[:num], y15[:num])
    line16.set_data(x[:num], y16[:num])

ax.set_xlabel('Time',fontsize=15)
ax.set_ylabel('A units activity',fontsize=15)
ax.set_title('Thunderstruck Rock tempo',fontsize=20,y=1.05)

line_actions = animation.FuncAnimation(fig, update_actions, time+1, fargs=[x,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12,line13,line14,line15,line16],
                                   interval=1, blit=False)

line_actions.save('action_units.mp4', writer=writer)
