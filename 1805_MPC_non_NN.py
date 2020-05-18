# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:08:21 2020

@author: iver_
"""

import matplotlib.pyplot as plt
from casadi import *
import numpy as np
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

def shift(T, t0, x0, u, f):
    st = x0
    con = u[0,:].T
    f_value = f(st,con)
    st = st + (T*f_value)
    x0 = st.full()
    t0 = t0 + T
    a1 = u[1:u.size1(),:]
    print(a1)
    a2 = u[u.size1()-1,:]
    u0 = np.vstack((a1, a2)) # HER ER FUCKING PROBLEMZ FFFFS
    print(u0)
    return t0, x0, u0

simtime = 20
T = 1/12 # Samlingtime
N = 3 # Prediction horizon
length = int(simtime/T)

# Må ha egne grenser for z og f. 35 < f < 65, 0 < z < 1
z_max = 1
z_min = 0
f_max = 65
f_min = 35

pbh = SX.sym('pbh')
pwh = SX.sym('pwh')
q = SX.sym('q')
states = vertcat(pbh, pwh, q)
n_states = states.size1()
#f = 50
z = SX.sym('z')
f = SX.sym('f')
controls = vertcat(z,f)
n_controls = controls.size1()
#rhs = vertcat((1-x2**2)*x1 - x2 + u, x1)  #Right hand side of eq
rhs = vertcat((B1/V1)*(PI*(pr-pbh)-q),(B2/V2)*(q-Cc*(sqrt(fmax(10**-5, pwh-pm))*z)),(1/M)*(pbh-pwh-rho*g*hw - 0.158*((rho*L1*q**2)/(D1*A1**2))*(my/(rho*D1*q))**(1/4) - 0.158*((rho*L2*q**2)/(D2*A2**2))*(my/(rho*D2*q))**(1/4) + rho*g* (CH*(9.5970e2 + 7.4959e3*((q/CQ)*(f0/f)) - 1.2454e6*((q/CQ)*(f0/f))**2)*(f/f0)**2)))
#
#
##Algebraic equation
#dae.add_alg('qr', qr-PI*(pr-pbh))
#dae.add_alg('qc', qc-Cc*(sqrt(fmax(10**-5, pwh-pm))*u))
#
#dae.add_alg('F1', F1 - 0.158*((rho*L1*q**2)/(D1*A1**2))*(my/(rho*D1*q))**(1/4))
#dae.add_alg('F2', F2 - 0.158*((rho*L2*q**2)/(D2*A2**2))*(my/(rho*D2*q))**(1/4))
#
## Algebraic ESP equations:
## VCF for ESP flow rate
#CQ = 1 - 2.6266*my + 6.0032*my**2 - 6.8104*my**3 + 2.7944*my**4
#CH = 1 - 0.03*my
#dae.add_alg('DeltaPp', DeltaPp - rho*g* (CH*(9.5970e2 + 7.4959e3*((q/CQ)*(f0/f)) - 1.2454e6*((q/CQ)*(f0/f))**2)*(f/f0)**2))
f = Function("f", [states,controls],[rhs]) # Nonlinear mapping function f(x,u)
U = SX.sym('U',n_controls,N) # Declaration variables (control). Prediction.
P = SX.sym('P',n_states + n_states) # parameters (which include the initial and the reference state of the watertank)
X = SX.sym('X',n_states,(N+1)) # A matrix that represents the states over the optimization problem. Prediction
# Compute solution symbolically
X[:,0] = P[0:3] # Init state 3 nå?
# Filling up stateprediction matrix
for k in range(0,N):
    st = X[:,k]
    con = U[:,k]
    f_value = f(st,con) # This function gives us the right hand side. Should be a NN in the final version
    st_next = st + (T*f_value) # Euler
    X[:,k+1] = st_next
ff = Function('ff',[U,P], [X]) # Gives prediction of X.jj
obj = 0 # Objective function
g = [] # Constraints vector
## Tuning parameters
Q = np.zeros((3,3))
Q[0,0] = 1
Q[1,1] = 1
Q[2,2] = 1
R = np.zeros((2,2)) # Feilen til SOndre, R va 3x3
R[0,0] = 1
R[1,1] = 1


# computing objective
for k in range(0,N):
    st = X[:,k]
    con = U[:,k]
    obj = obj + (st-P[3:6]).T @ Q @ (st-P[3:6]) + con.T @ R @ con
    #obj =  obj + (st-P[2:4])*Q*(st-P[2:4]) #+ con.T*R*con # Calculate obj. Sum, derfor i forloop er referanse
# Constraints kjem her.
# g = ...
# Make the decision variables one column vector
OPT_variables = reshape(U,2*N,1) # Må bli med om vi har meir enn en kontrollvariabel
nlp_prob = {'f':obj,'x':OPT_variables, 'g': g, 'p': P}

opts = {'ipopt':{'max_iter':100, 'print_level':0, 'acceptable_tol':1e-8, 'acceptable_obj_change_tol':1e-6 }, 'print_time':0}

solver = nlpsol('solver','ipopt', nlp_prob, opts)
# Input constraints (Commented out, think I can fix this later on)
#args.lbx(1:1:N,2) = u_min;
#args.ubx(1:1:N,2) = u_max;
#


##-------------------------------------------------------------------------
## THE SIMULATION LOOP STARTS FROM HERE
t0 = 0
x0 = vertcat(70e5,30e5,0.01)
xs = vertcat(76e5,23e5,0.0115)

xx = np.zeros([3,length+1])
xx[0,0] = 70e5
xx[1,0] = 30e5 # xx contains the history of states.
xx[2,0] = 0.025
t = np.zeros([length])
t[0] = 0

u0 = np.zeros([N,2])
u0[:,0] = 0.4
u0[:,1] = 40
#u0 = zeros(N,1) # One control input.
sim_tim = simtime # Maximum simulation time.xxl

#  Creating a dict where all info is stored
lbx = np.zeros(N*n_controls)
lbx[::2] = z_min
lbx[1::2] = f_min

ubx = np.zeros(N*n_controls)
ubx[::2] = z_max
ubx[1::2] = f_max



args = {'lbx':lbx, 'ubx':ubx}

# Start MPC
mpciter = 0 # Counter for the loop
#xxl = np.zeros([N+1,1,length])
xxl = np.zeros([N+1,3,length])
u_cl = np.zeros([length,2])
#main_loop = tic; # Find something later to measure the time spent.

while(mpciter < sim_tim / T):
    # args.p = [x0, xs] # set the values of the parameters vector
    ax0 = reshape(u0.T,2*N,1) # initial value of the optimization variables (x0)
    # args.x0 = u0';
    
    # p skal ligge på rekke og rad, NEDOVER! :)
    # Dette blir på en måte en ny funksjon, sjekk ut docs
    sol = solver(x0=ax0, p=vertcat(x0,xs), lbx=args['lbx'], ubx=args['ubx'])
    # u = np.reshape(sol['x'].full().T,1,N).T#  optimal values for minimize control
    u = reshape(sol['x'].full().T, (2,N)).T 
    #ff_value = ff(u.T,[x0,xs]) #compute OPTIMAL solution TRAJECTORY
    ff_value = ff(u.T,vertcat(x0,xs))
    xxl[:,0:3,mpciter] = ff_value.full().T
    u_cl[mpciter] = u[0,:]
    
    t[mpciter] = t0
    [t0, x0, u0] = shift(T, t0, x0, u, f)
    #xx[:,mpciter+1] = x0
    xx[0,mpciter+1] = x0[0]
    xx[1,mpciter+1] = x0[1]
    xx[2,mpciter+1] = x0[2]
    mpciter = mpciter +1
    
    
# plt.figure()
# plt.subplot(311)
# plt.step(t,u_cl[:,0],color='red', label='$u$')
# plt.grid()
# plt.legend()
# plt.ylabel('$p_{bh} \ [bar]$')

plt.figure()
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

    
plt.figure()
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
