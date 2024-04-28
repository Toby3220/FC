
import numpy as np
import pandas as pd
import American_Put_Close_Form as close_form

import Least_Square_Monte_Carlo as ls_monte_carlo
from tqdm import tqdm
from matplotlib import pyplot as plt

#------------------------------#
rf = 0.05
vol = 0.25
n = 10000
s0 = 100
K = 100

horizon = 1
dtime = 0.01
N_steps = horizon/dtime

# dtime_CS = 1/5
dtime_CS = 0.01
times = np.arange(0,horizon+round(dtime_CS,3),dtime_CS).round(4)
#------------------------------#


american_close_form = close_form.AmericanPut(rf, vol)
price_cs = american_close_form.price(s0,K,times)

american_lsm = ls_monte_carlo.American_Put_LSMC(0.01,1)
american_lsm.sample((rf-0.5*vol)*dtime,np.sqrt(vol*dtime),n,s0)
option_value, exercise = american_lsm.price(K,rf,4,0.05)
price_lsm = option_value[0].mean()
std_error = option_value[0].std()/np.sqrt(n)

print("Close form solution price        : {}".format(round(price_cs,5)))
print("Least Square Monte Carlo price   : {} ± {}".format(round(price_lsm,5),round(3*std_error,5)))

print("end")