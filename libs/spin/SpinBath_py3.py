#!/bin/sh

#  SpinBath_py3.sh
#  
#
#  Created by Dale Scerri on 13/04/2018.
#

import sys
from matplotlib import pyplot as plt
import numpy as np
import importlib
sys.path.append ('/Users/dalescerri/Documents/GitHub/simulations')
from libs.spin import diluted_NSpin_bath_ClusterTest_py3 as NSpin 
importlib.reload (NSpin)

exp2 = NSpin.SpinExp_cluster1()
exp2.set_experiment(nr_spins=8, hf_approx=False, clus=True)
for i in range(20):
    exp2.Ramsey(tau=1e-5, phi=0)

exp2.plot_bath_evolution()
