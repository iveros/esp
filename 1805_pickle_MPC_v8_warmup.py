# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:32:44 2020

@author: iver_
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
#import Project_esnmpc.RNN as RNN
import pickle
import os
from casadi import *
import time

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

def feature_scaling(x,xmax,xmin,normalmode=True):
    if normalmode:
        y = (x - xmin)/(xmax - xmin)
    else:
        y = x 	 
    return y
 

def feature_descaling(x,xmax,xmin,normalmode=True):
    if normalmode:
        y = x*(xmax-xmin) + xmin 
    else:
        y = x
    return y


def nn_prediction(con):
    #y = network.update(u_data_normal[k]).flatten()
    #y = network.update(con.flatten())
    control = con[0]
    y = network.update(control)
    return y

def esn_pred(con, N, b):
    #con[0,:] is valve, con[1,:] is freq
    y_esn_pred = SX.sym('y_esn_pred',n_states,(N))
    #a = network.get_current_state()
    a = b
    for k in range(0,N):
        Input = con[:,k]
        z = Wrr @ a + Wir @ Input + Wbr
        next_network_state = (1-leakrate)*a + leakrate*tanh(z) # Dette er neste a, så den skal brukast
        a_wbias = vertcat(1.0, next_network_state)
        output = Wro @ a_wbias
        
        a = next_network_state
        
        y_esn_pred[:,k] = output
        
    return y_esn_pred#output, next_network_state


def shift(T, t0, x0, u, f):
    # Gir dette egentlig meining?
    x0 = DM(x0)
    st = x0
    #st = feature_descaling(x0, y_max, y_min)
    con = u[0,:].T
    print(u)
    #con = feature_descaling(cont, u_max, u_min)
    f_value = f(st,con)
    st = st + (f_value*T) 
    x0 = st.full()
    t0 = t0 + T
    a1 = u[1:u.size1(),:]
    a2 = u[u.size1()-1,:]
    u0 = np.vstack((a1, a2))
    print(u0)
    return t0, x0, u0

# Echo state has 12 samples pr minute
simtime = 3 #Minutes

T = 1/12 # Sampling time
N = 3# Prediction horizon
length = int(simtime/T) # Number of datapoints

# Min and max on inputs.
z_max = 1
z_min = 0
f_max = 65
f_min = 35

# Importing network and some of the network parameters.
pickle_file = open('esnespJean.pickle','rb') # Open Echo State Network

network = pickle.load(pickle_file)
saved_weights = network.save_reservoir("weightsTest")
data = {}
io.loadmat('weightsTest', data)

a = network.get_current_state()
Wir = data['Wir']
Wbr = data['Wbr']
Wro = data['Wro']
Wrr = data['Wrr']
leakrate = 0.27


x = MX.sym('x',3); # states
p = MX.sym('p',2); # Controls
n_states = x.size1()
n_controls = p.size1()

# rhs is the true system, used in the shift function in the while loop.
rhs = vertcat((B1/V1)*(PI*(pr-x[0])-x[2]),(B2/V2)*(x[2]-Cc*(sqrt(fmax(10**-5, x[1]-pm))*p[0])),(1/M)*(x[0]-x[1]-rho*g*hw - 0.158*((rho*L1*x[2]**2)/(D1*A1**2))*(my/(rho*D1*x[2]))**(1/4) - 0.158*((rho*L2*x[2]**2)/(D2*A2**2))*(my/(rho*D2*x[2]))**(1/4) + rho*g* (CH*(9.5970e2 + 7.4959e3*((x[2]/CQ)*(f0/p[1])) - 1.2454e6*((x[2]/CQ)*(f0/p[1]))**2)*(p[1]/f0)**2)))
f = Function("f", [x,p],[rhs]) # Nonlinear mapping function f(x,u) (used in shift function)
U = SX.sym('U',n_controls,N) # Declaration variables (control). Prediction.
P = SX.sym('P',n_states + n_states) # parameters (which include the initial and the reference state of the ESP)
X = SX.sym('X',n_states,(N+1)) # A matrix that represents the states over the optimization problem. Prediction

#data = io.loadmat('ESPV12new0605.mat')

# Import input and output data
#u_data = data['u']
#y_data = data['y']

# Min og max for later normalization.
y_max = np.array([1.24606e07,9.40229e06, 0.0175620])
y_min = np.array([6.18823e06,2.03568e06,6.79091e-08])
u_max = np.array([1,65])
u_min = np.array([0,35])



#obj = 0 # Objective function
g = [] # Constraints vector
## Tuning parameters
Q = np.zeros((3,3))
Q[0,0] = 1
Q[1,1] = 1
#Q[2,2] = 7e6 # Doesn't work to increase to inf.
Q[2,2] = 10
R = np.zeros((2,2)) # Feilen til SOndre, R va 3x3
R[0,0] = 1
R[1,1] = 1


# Constraints kjem her.
# g = ...
# Make the decision variables one column vector

opts = {'ipopt':{'max_iter':100, 'print_level':0, 'acceptable_tol':1e-8, 'acceptable_obj_change_tol':1e-6 }, 'print_time':0}



# ##-------------------------------------------------------------------------
## THE SIMULATION LOOP STARTS FROM HERE
t0 = 0
# x0 = vertcat(88e5,30e5,0.01)
x0 = vertcat(70e5,30e5,0.01)
xs = vertcat(76e5,23e5,0.0115)
#xs = vertcat(88e5,24e5,0.01)
#xs = vertcat(76e5,24e5,0.0115)

xx = np.zeros([3,length+1])
xx[0,0] = x0[0]# xx contains the history of the states
xx[1,0] = x0[1]
xx[2,0] = x0[2]
t = np.zeros([length])
t[0] = 0

u0 = np.zeros([N,2])

u0[:,0] = 0.4
u0[:,1] = 40
#u0 = zeros(N,1) # One control input.
sim_tim = simtime # Maximum simulation time.xxl

#  Creating a dict where all constraints is stored
lbx = np.zeros(N*n_controls)
lbx[::2] = z_min
lbx[1::2] = f_min

ubx = np.zeros(N*n_controls)
ubx[::2] = z_max
ubx[1::2] = f_max

lbg = np.zeros(N*n_states)
lbg[::3] = 0
lbg[1::3] = 0
lbg[2::3] = 0

p = vertcat(x0,xs)

# Trying to make some kind of warm-up for the network. But not working as I want.
for k in range(0,200):
    u_data = feature_scaling([0.4, 40], u_max, u_min)
    network.update(u_data)




args = {'lbx':lbx, 'ubx':ubx, 'lbg':lbg, 'p':p}

# Start MPC
mpciter = 0 # Counter for the loop
#xxl = np.zeros([N+1,1,length])
xxl = np.zeros([N+1,3,length]) # Matrix that shows prediction horizons.
u_cl = np.zeros([length,2]) # Matrix with control history
#main_loop = tic; # Find something later to measure the time spent.
f_val = np.zeros([length,1]) # Tried to make an array with cost function values (currently strange)

stateshistory = np.zeros([length, 200, 2]) # Just to check if NN states are changing.


while(mpciter < sim_tim / T):
    
    tic = time.clock()
    
    obj = 0
    a = network.get_current_state() # Get ESN state to use in the esn_pred.
    stateshistory[mpciter, :, :] = a
    X[:,0] = P[0:3] # P[0:3] is the "real" first value
    #############################################################################
    # Filling up the prediction matrix (Casadi syntaxt with symbolic expression)
    #st = X[:,0:N]
    con = U[:,0:N]
    cont = feature_scaling(con, u_max, u_min)
    
    X_NN = esn_pred(cont, N, a) # Getting a state prediction matrix from the ESN.
    X_NN_scaled = feature_descaling(X_NN, y_max, y_min)
    X[:,1:N+1] = X_NN_scaled
    #############################################################################
   # X[:,k+1] = X_NN_scaled[:,k]
    # for k in range(0,N): 
    #     st = X[:,k]
    #     con = U[:,k]
    # for k in range(0,N):   
    #     obj = obj + (X[:,k]-P[3:6]).T @ Q @ (X[:,k]-P[3:6])
        
    #### Cost function
    # P[3:6] = xs, so the cost functions will minimize the deviation between y_pred (NN) and xs.
    for k in range(0,N):
         obj = obj + (X_NN_scaled[:,k]-P[3:6]).T @ Q @ (X_NN_scaled[:,k]-P[3:6])
        
        
        
    ff = Function('ff',[U,P], [X]) # Gives prediction of X.jj
    
    OPT_variables = reshape(U,2*N,1)
    nlp_prob = {'f':obj,'x':OPT_variables, 'g': g, 'p': P} # Defining NLP problem

    solver = nlpsol('solver','ipopt', nlp_prob, opts)  
    
    ax0 = reshape(u0.T,2*N,1) # initial value of the optimization variables (x0)
    #sol = solver(x0=ax0, lbx=args['lbx'], ubx=args['ubx'], p=args['p']) # Finding optimal control in the nlp_prob.
    sol = solver(x0=ax0, p=vertcat(x0,xs), lbx=args['lbx'], ubx=args['ubx'])
    u = reshape(sol['x'].full().T, (2,N)).T
 
    
   # print(u)
    ff_value = ff(u.T,vertcat(x0,xs))
    #print(ff_value)
    xxl[:,0:3,mpciter] = ff_value.full().T
    
    
    u_cl[mpciter] = u[0,:]
    

    f_val[mpciter] = sol['f'].full()
    t[mpciter] = t0
    
    [t0, x0, u0] = shift(T, t0, x0, u, f)


    
    # Update network with the control input
    network.update(u[0,:])
    print(u[0,:])
    
    # Filling up matrix with the states history (the one that is plotted)
    xx[0,mpciter+1] = x0[0]
    xx[1,mpciter+1] = x0[1]
    xx[2,mpciter+1] = x0[2]
    mpciter = mpciter +1
    toc = time.clock()
    print(toc-tic)
# toc = time.clock()
#print(toc-tic)
# plt.figure()
# plt.subplot(311)
# plt.step(t,u_cl[:,0],color='red', label='$u$')
# plt.grid()
# plt.legend()
# plt.ylabel('$p_{bh} \ [bar]$')

plt.figure(45)
plt.subplot(311)
plt.plot(xx[0,:]/10**5,color='blue')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')

plt.subplot(312)
plt.plot(xx[1,:]/10**5,color='blue')
plt.grid()
plt.legend()
plt.ylabel('$p_{wh} \ [bar]$')

plt.subplot(313)
plt.plot(xx[2,:],color='blue')
plt.grid()
plt.legend()
plt.ylabel('$q \ [bar]$')

    
plt.figure(46)
plt.subplot(211)
plt.step(t,u_cl[:,0],color='red', label='$z$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')

    
plt.subplot(212)
plt.step(t,u_cl[:,1],color='red', label='$f$')
plt.grid()
plt.legend()
plt.ylabel('$p_{bh} \ [bar]$')


# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:12:02 2020

@author: iver_
"""


