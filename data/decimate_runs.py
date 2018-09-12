import random
from glob import glob
import pickle
import numpy as np
import sys

def decimate_run (pickle_file, frac_remain):
  # returns fraction of events chosen randomly
  #  with livetime corrected so rate is same 
  with open(pickle_file, 'rb') as f: g = pickle.load(f)

  keys = g.keys()
  length = len(g[keys[0]])
  
  n_remain = int(length * frac_remain)
  indices = range(length)
  i_remain = random.sample(indices, n_remain)

  new_livetime = g['livetime'][0] * frac_remain
  g['livetime'] = np.array([new_livetime]*length)  

  for key in g.keys():
    g[key] = g[key][i_remain]

  filename = pickle_file[:-4]+'_decimated.pkl'
  pickle.dump(g, open(filename ,'wb'))
 
files = sorted(glob('/data/user/sfahey/FRB/L2_Analysis/data/%s/offtime/*all.pkl'%sys.argv[1]))

for run in files:
  decimate_run(run, 0.05)


