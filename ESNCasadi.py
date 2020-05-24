# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:08:17 2020

@author: iver_
"""
from casadi import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
#import Project_esnmpc.RNN as RNN
import pickle
import os
from casadi import *
import time

def feature_scaling(x,xmax,xmin,normalmode=True):
    if normalmode:
        y = (x - xmin)/(xmax - xmin)
    else:
        y = x 	 
    return y
 
class ESNCasadi(Callback):
  def __init__(self, name, opts={}):
    pickle_file = open('esnFullpakkecompressed.pickle', 'rb')  # Open Echo State Network
    self.esn = pickle.load(pickle_file)

    Callback.__init__(self)
    self.construct(name, opts)

  def get_n_in(self): return 1
  def get_n_out(self): return 1

  def get_sparsity_in(self,i):
    return Sparsity.dense(self.esn.n_in, 1)

  def get_sparsity_out(self,i):
    return Sparsity.dense(self.esn.n_out, 1)

  # Evaluate numerically
  def eval(self, arg):
    # "return ESN prediction with self.esn object"
    # print(np.array(arg[0]))
    pred = self.esn.update(np.array(arg[0]).T)
    return [pred]
  
  # Kjører først save_state, så gjer prediction horizon for så å kjøre reset state for å få tilbake staten som var før prediction horizon.
  # Må på en eller annen måte også greie å få inn en update. i slutten for å gå vidare til neste state.  
  def reset_state(self):
    self.esn.a = self.state

  def save_state(self):
    self.state = self.esn.a


u_max = np.array([1,65])
u_min = np.array([0,35])
u_data = feature_scaling([0.4, 40], u_max, u_min)

f_esn = ESNCasadi('f_esn') # [inp], [out])
f_esn.save_state()
st1 = f_esn.esn.get_current_state()
for k in range(0,10):
    out = f_esn(vertcat(u_data))
    print(out)
## check current ESN state:
#print(f_esn.esn.get_current_state())
f_esn.reset_state()
st2 = f_esn.esn.get_current_state()
print(out)

f_esn.save_state()
### do prediction horizon
f_esn.reset_state()
