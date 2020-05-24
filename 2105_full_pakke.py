# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:03:54 2020

@author: iver_
"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from control import *
from casadi import *
from fp_lib import *
import scipy.io as io

class sim_model(SimulationModel):
    
    def __init__(self, rhs, x0):
        
        self.rhs = rhs
        
        self.x = x0
        
        
    def model_output(self, u):
        y = f(self.x, u)
        self.x = self.x + y*T
        if self.x[2] < 0:
            self.x[2] = 1e-9
        #print(f)
        return self.x
        
g = 9.81
Cc = 2*10**(-5)
A1 = 0.008107
A2 = 0.008107
D1 = 0.1016
D2 = 0.1016
h1 = 200
h2 = 800
hw = 1000
L1 = 500
L2 = 1200
V1 = 4.054
V2 = 9.729
#ESP DATA
B1 = 1.5*10**9
B2 = 1.5*10**9
M = 1.992*10**8
rho = 950
pr = 1.26 * 10 ** 7
f0 = 60 # Hz
#f = [57, 57, 54, 64, 64, 49, 49, 51, 53, 54]
#f = 53 # Hz
#UNKNOWN
PI = 2.32*10**(-9)
#KANSKJE
pm = 20e5
my = 0.025 # Pa*s
CQ = 1 - 2.6266*my + 6.0032*my**2 - 6.8104*my**3 + 2.7944*my**4
CH = 1 - 0.03*my



x = MX.sym('x',3)
p = MX.sym('p',2)
n_states = x.size1()
n_controls = p.size1()


rhs = vertcat((B1/V1)*(PI*(pr-x[0])-x[2]),(B2/V2)*(x[2]-Cc*(sqrt(fmax(10**-5, x[1]-pm))*p[0])),(1/M)*(x[0]-x[1]-rho*g*hw - 0.158*((rho*L1*x[2]**2)/(D1*A1**2))*(my/(rho*D1*x[2]))**(1/4) - 0.158*((rho*L2*x[2]**2)/(D2*A2**2))*(my/(rho*D2*x[2]))**(1/4) + rho*g* (CH*(9.5970e2 + 7.4959e3*((x[2]/CQ)*(f0/p[1])) - 1.2454e6*((x[2]/CQ)*(f0/p[1]))**2)*(p[1]/f0)**2)))

f = Function('f', [x,p], [rhs])

X = np.empty([1, 3])
# Init values
X[0,0] = 70e5
X[0,1] = 30e5
X[0,2] = 0.01

x0 = vertcat(X[0,0], X[0,1], X[0,2])

esp_system = sim_model(rhs, x0)


#these lines of code create the identificaton signal:
simtime = 300
T = 0.02 # Samples pr minute
sim_length = int(simtime/T)


# Creating random inputs
u_min_valve = np.array([0.0])
u_max_valve = np.array([1.0])
some_input_valve = RFRAS(u_min_valve,u_max_valve,sim_length,30)

u_min_freq = np.array([35])
u_max_freq = np.array([65])
some_input_freq = RFRAS(u_min_freq,u_max_freq,sim_length,30)


#THe data MUST be 2dimensional, and data must organized in row form, so in this case:
some_input = np.vstack((some_input_valve, some_input_freq)).T


#the resulting dimension should always be (number of data x number of features)

#now we need to run the system with that signal to obtain the system response, which is recorded in y_data
y_data = np.empty([sim_length,3])

#the simulation and plotting for data gathering is as follows:
for i in range(sim_length):
    y = esp_system.model_output(some_input[i]).T
    #print(y)
    y_data[[i],] = y  

np.save('some_data.npy', some_input) # for testing purposes, comparing it with other framworks.


time = np.linspace(0,simtime,len(y_data)-1)

# Plots the system with the random inputs. Just too see what dataset we are working with

# plt.figure(4)
# plt.subplot(511)
# plt.plot(time, y_data[1:,0]/10**5,color='blue', label='$p_{bh}$')
# plt.grid()
# plt.legend()
# plt.ylabel('$p_{bh} \ [bar]$')


# plt.subplot(512)
# plt.plot(time, y_data[1:,1]/10**5,color='blue',label='$p_{wh}$')
# plt.grid()
# plt.legend()
# plt.ylabel('$p_{wh} \ [bar]$')

# plt.subplot(513)
# plt.plot(time, y_data[1:,2]*3600,color='blue',label='$q$')
# plt.grid()
# plt.legend()
# plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')

# plt.subplot(514)
# plt.step(time, some_input[1:,0],color='red',label='$z$')
# plt.grid()
# plt.legend()
# plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')

# plt.subplot(515)
# plt.step(time, some_input[1:,1] ,color='red',label='$f$')
# plt.grid()
# plt.legend()
# plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')

########################################################
# Normalize
u_max = np.vstack((u_max_valve, u_max_freq)).T
u_min = np.vstack((u_min_valve, u_min_freq)).T
u_data_normal = feature_scaling(some_input,u_max,u_min)

y_max = np.max(y_data,axis=0)
y_min = np.min(y_data,axis=0)
y_data_normal = feature_scaling(y_data,y_max,y_min)

# gama= 0.27
esp_esn = EchoStateNetwork(neu = 200, n_in = 2, n_out = 3, gama=0.14,
                 ro=0.99, psi=0.0,in_scale=0.03, bias_scale=0.1,
                 output_feedback = False, noise_amplitude = 0,
                 load_initial = False, save_initial=False)




regularization = 1e-8
warmupdrop = 100
# regularization = 1e-8
# esp_esn.offline_training1(u_data_normal,y_data_normal,regularization,warmupdrop)
# esp_esn.reset()
# Train network with training data
#esp_esn.offline_training1(u_data_normal,y_data_normal,regularization,warmupdrop)

#esp_esn.reset() # RESET WILL DESTROY THE START, DON'T USE IT!!

esp_esn.add_data(u_data_normal,y_data_normal,warmupdrop)


# #now, we train. The training function also returns the best regularization parameter while insternally changing the weights.
min_reg = 1e-15
max_reg = 1e-8
error, reg = esp_esn.cum_train_cv(min_reg,max_reg)

print("best cv error:",error,"reg value:",reg)




#adter training the ESN, it is recommended you generate another input signal to test the ESN:

test_points = 5000

# Creating random inputs
test_u_min_valve = np.array([0.0])
test_u_max_valve = np.array([1.0])
test_input_valve = RFRAS(test_u_min_valve,test_u_max_valve,test_points,30)

test_u_min_freq = np.array([35])
test_u_max_freq = np.array([65])
test_input_freq = RFRAS(test_u_min_freq,test_u_max_freq,test_points,30)


#THe data MUST be 2dimensional, and data must organized in row form, so in this case:
test_input = np.vstack((test_input_valve, test_input_freq)).T

y_pred_test = np.empty([test_points,3])
y_test = np.empty_like(y_pred_test)
#this is where the function feature_descaling comes in handing, as it descales the output back into system scale.



####################################################
# These lines below could work as a warmup for the prediction. (DO not use!)
# modwarmup = 200
# warmup_input = np.full((modwarmup,  2), (0.5, 40))

# for i in range(0,modwarmup):
#     k = esp_system.model_output(warmup_input[i]).T
#     u_normal = feature_scaling(warmup_input[i],u_max,u_min)
#     esp_esn.update(u_normal)

for i in range(test_points):
    #get system output
    y = esp_system.model_output(test_input[i]).T
    #print(y)
    y_test[[i],] = y  
    # y_test[i,:] = esp_system.model_output(test_input[i]).T
    
    #normalize to apply in ESN
    u_normal = feature_scaling(test_input[i],u_max,u_min)
    y_normal = esp_esn.update(u_normal)
    y_pred_test[i] = feature_descaling(y_normal.flatten(),y_max,y_min)
    
    
time = np.linspace(0,test_points*T,len(y_test)-1)

plt.figure(1)
plt.subplot(311)
# plt.plot(y_test[200:,0]/10**5,color='blue', label='$p_{bh}$ real')
plt.plot(y_test[0:,0]/10**5,color='blue', label='$p_{bh}$ real')
plt.plot(y_pred_test[0:,0]/10**5,color='red', label='$p_{bh}$ pred')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')


plt.subplot(312)
# plt.plot(y_test[200:,1]/10**5,color='blue',label='$p_{wh}$ real')
plt.plot(y_test[0:,1]/10**5,color='blue',label='$p_{wh}$ real')
plt.plot(y_pred_test[0:,1]/10**5,color='red',label='$p_{wh}$ pred')
plt.grid()
plt.legend()
plt.ylabel('$p_{wh} \ [bar]$')

plt.subplot(313)
# plt.plot(y_test[200:,2]*3600,color='blue',label='$q$ real')
plt.plot(y_test[0:,2]*3600,color='blue',label='$q$ real')
plt.plot(y_pred_test[0:,2]*3600,color='red',label='$q$ pred')
plt.grid()
plt.legend()
plt.ylabel(r'$q \ [\dfrac{m^3}{h}]$')

#########################################################################
# Saving the network to a pickle file. For exporting the trained network
# import pickle
# pickle_file = open('esnFullpakkecompressed.pickle','wb')

# pickle.dump(esp_esn,pickle_file)

# pickle_file.close()


# # Things are good for now.
# ############## MPC ############################################################
# import time
# saved_weights = esp_esn.save_reservoir("weightsTest")
# data = {}
# io.loadmat('weightsTest', data)

# Wir = data['Wir']
# Wbr = data['Wbr']
# Wro = data['Wro']
# Wrr = data['Wrr']
# leakrate = 0.27



# def esn_pred(con, N, b):
#     #con[0,:] is valve, con[1,:] is freq
#     y_esn_pred = SX.sym('y_esn_pred',n_states,(N))
#     #a = network.get_current_state()
#     a = b
#     for k in range(0,N):
#         Input = con[:,k]
#         z = Wrr @ a + Wir @ Input + Wbr
#         next_network_state = (1-leakrate)*a + leakrate*tanh(z) # Dette er neste a, så den skal brukast
#         a_wbias = vertcat(1.0, next_network_state)
#         output = Wro @ a_wbias
        
#         a = next_network_state
        
#         y_esn_pred[:,k] = output
        
#     return y_esn_pred#output, next_network_state


# def shift(T, t0, x0, u, f):
#     # Gir dette egentlig meining?
#     x0 = DM(x0)
#     st = x0
#     #st = feature_descaling(x0, y_max, y_min)
#     con = u[0,:].T
#     print(u)
#     #con = feature_descaling(cont, u_max, u_min)
#     # f_value = f(st,con)
#     # st = st + (f_value*T) 
#     st = esp_system.model_output(some_input[i]).T
#     x0 = st.full()
#     t0 = t0 + T
#     a1 = u[1:u.size1(),:]
#     a2 = u[u.size1()-1,:]
#     u0 = np.vstack((a1, a2))
#     print(u0)
#     return t0, x0, u0

# simtime = 3 #Minutes
# N = 4# Prediction horizon
# length = int(simtime/T) # Number of datapoints, T is initaliaized in the start.

# # Min and max on inputs.
# z_max = 1
# z_min = 0
# f_max = 65
# f_min = 35

# U = SX.sym('U',n_controls,N) # Declaration variables (control). Prediction.
# P = SX.sym('P',n_states + n_states) # parameters (which include the initial and the reference state of the ESP)
# X = SX.sym('X',n_states,(N+1)) # A matrix that represents the states over the optimization problem. Prediction


# #obj = 0 # Objective function
# g = [] # Constraints vector
# ## Tuning parameters
# Q = np.zeros((3,3))
# Q[0,0] = 1
# Q[1,1] = 1
# #Q[2,2] = 7e6 # Doesn't work to increase to inf.
# Q[2,2] = 1
# R = np.zeros((2,2)) # Feilen til SOndre, R va 3x3
# R[0,0] = 1
# R[1,1] = 1


# # Constraints kjem her.
# # g = ...
# # Make the decision variables one column vector

# opts = {'ipopt':{'max_iter':100, 'print_level':0, 'acceptable_tol':1e-8, 'acceptable_obj_change_tol':1e-6 }, 'print_time':0}



# # ##-------------------------------------------------------------------------
# ## THE SIMULATION LOOP STARTS FROM HERE
# t0 = 0
# # x0 = vertcat(88e5,30e5,0.01)
# x0 = vertcat(70e5,30e5,0.01)
# xs = vertcat(70e5,23e5,0.0115)
# #xs = vertcat(88e5,24e5,0.01)
# #xs = vertcat(76e5,24e5,0.0115)

# xx = np.zeros([3,length+1])
# xx[0,0] = x0[0]# xx contains the history of the states
# xx[1,0] = x0[1]
# xx[2,0] = x0[2]
# t = np.zeros([length])
# t[0] = 0

# u0 = np.zeros([N,2])

# u0[:,0] = 0.4
# u0[:,1] = 40
# #u0 = zeros(N,1) # One control input.
# sim_tim = simtime # Maximum simulation time.xxl

# #  Creating a dict where all constraints is stored
# lbx = np.zeros(N*n_controls)
# lbx[::2] = z_min
# lbx[1::2] = f_min

# ubx = np.zeros(N*n_controls)
# ubx[::2] = z_max
# ubx[1::2] = f_max

# lbg = np.zeros(N*n_states)
# lbg[::3] = 0
# lbg[1::3] = 0
# lbg[2::3] = 0

# p = vertcat(x0,xs)

# # Trying to make some kind of warm-up for the network. But not working as I want.
# # for k in range(0,200):
# #     u_data = feature_scaling([0.4, 40], u_max, u_min)
# #     network.update(u_data)




# args = {'lbx':lbx, 'ubx':ubx, 'lbg':lbg, 'p':p}

# # Start MPC
# mpciter = 0 # Counter for the loop
# #xxl = np.zeros([N+1,1,length])
# xxl = np.zeros([N+1,3,length]) # Matrix that shows prediction horizons.
# u_cl = np.zeros([length,2]) # Matrix with control history
# #main_loop = tic; # Find something later to measure the time spent.
# f_val = np.zeros([length,1]) # Tried to make an array with cost function values (currently strange)

# stateshistory = np.zeros([length, 200, 2]) # Just to check if NN states are changing.

# y_pred = np.zeros([length,3])
# while(mpciter < sim_tim / T):
    
#     tic = time.clock()
    
#     obj = 0
#     a = esp_esn.get_current_state() # Get ESN state to use in the esn_pred.
#     stateshistory[mpciter, :, :] = a
#     X[:,0] = P[0:3] # P[0:3] is the "real" first value
#     #############################################################################
#     # Filling up the prediction matrix (Casadi syntaxt with symbolic expression)
#     #st = X[:,0:N]
#     con = U[:,0:N]
#     cont = feature_scaling(con, u_max, u_min)
#     # WOULD IT BE POSSIBLE TO MAKE A COPY OF THE NETWORK TO GET THE PREDICTIONS FROM?
#     X_NN = esn_pred(cont, N, a) # Getting a state prediction matrix from the ESN.
#     X_NN_scaled = feature_descaling(X_NN, y_max, y_min)
#     X[:,1:N+1] = X_NN_scaled
#     #############################################################################
#    # X[:,k+1] = X_NN_scaled[:,k]
#     # for k in range(0,N): 
#     #     st = X[:,k]
#     #     con = U[:,k]
#     # for k in range(0,N):   
#     #     obj = obj + (X[:,k]-P[3:6]).T @ Q @ (X[:,k]-P[3:6])
        
#     #### Cost function
#     # P[3:6] = xs, so the cost functions will minimize the deviation between y_pred (NN) and xs.
#     for k in range(0,N):
#          obj = obj + (X_NN_scaled[:,k]-P[3:6]).T @ Q @ (X_NN_scaled[:,k]-P[3:6])
        
        
        
#     ff = Function('ff',[U,P], [X]) # Gives prediction of X.jj
    
#     OPT_variables = reshape(U,2*N,1)
#     nlp_prob = {'f':obj,'x':OPT_variables, 'g': g, 'p': P} # Defining NLP problem

#     solver = nlpsol('solver','ipopt', nlp_prob, opts)  
    
#     ax0 = reshape(u0.T,2*N,1) # initial value of the optimization variables (x0)
#     #sol = solver(x0=ax0, lbx=args['lbx'], ubx=args['ubx'], p=args['p']) # Finding optimal control in the nlp_prob.
#     sol = solver(x0=ax0, p=vertcat(x0,xs), lbx=args['lbx'], ubx=args['ubx'])
#     u = reshape(sol['x'].full().T, (2,N)).T
 
    
#    # print(u)
#     ff_value = ff(u.T,vertcat(x0,xs))
#     #print(ff_value)
#     esp_esn[mpciter,:] = ff_value[:,1].full().T
#     xxl[:,0:3,mpciter] = ff_value.full().T
    
    
#     u_cl[mpciter] = u[0,:]
    

#     f_val[mpciter] = sol['f'].full()
#     t[mpciter] = t0
    
#     [t0, x0, u0] = shift(T, t0, x0, u, f)


    
#     # Update network with the control input
#     network.update(u[0,:])
#     print(u[0,:])
    
#     # Filling up matrix with the states history (the one that is plotted)
#     xx[0,mpciter+1] = x0[0]
#     xx[1,mpciter+1] = x0[1]
#     xx[2,mpciter+1] = x0[2]
#     mpciter = mpciter +1
#     toc = time.clock()
#     print(toc-tic)
# # toc = time.clock()
# #print(toc-tic)
# # plt.figure()
# # plt.subplot(311)
# # plt.step(t,u_cl[:,0],color='red', label='$u$')
# # plt.grid()
# # plt.legend()
# # plt.ylabel('$p_{bh} \ [bar]$')

# plt.figure(45)
# plt.subplot(311)
# plt.plot(xx[0,:]/10**5,color='blue')
# plt.grid()
# plt.legend()
# plt.ylabel('$p_{bh} \ [bar]$')

# plt.subplot(312)
# plt.plot(xx[1,:]/10**5,color='blue')
# plt.grid()
# plt.legend()
# plt.ylabel('$p_{wh} \ [bar]$')

# plt.subplot(313)
# plt.plot(xx[2,:],color='blue')
# plt.grid()
# plt.legend()
# plt.ylabel('$q \ [bar]$')

    
# plt.figure(46)
# plt.subplot(211)
# plt.step(t,u_cl[:,0],color='red', label='$z$')
# plt.grid()
# plt.legend()
# plt.ylabel('$p_{bh} \ [bar]$')

    
# plt.subplot(212)
# plt.step(t,u_cl[:,1],color='red', label='$f$')
# plt.grid()
# plt.legend()
# plt.ylabel('$p_{bh} \ [bar]$')

