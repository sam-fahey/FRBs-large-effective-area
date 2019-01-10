from glob import glob
import pickle
from datetime import datetime
from datetime import timedelta
import sys
import numpy as np
import numpy.ma as ma

delta_t = 300
direct = '/data/user/sfahey/FRB/L2_Analysis/data/'
onfiles = glob(direct+ '201*/ontime/*all.pkl')

srcfile = '/data/user/sfahey/FRB/L2_Analysis/bg_srcs_noRepeater_topocentered.pkl'
with open( srcfile, 'rb' ) as opened: srcs = pickle.load( opened )

for onfile in onfiles:
  print onfile
  with open( onfile, 'rb' ) as opened: loaded = pickle.load( opened )

  times = loaded['t_UTC']

  mask = np.array([0]*len(times))
  for i in range(len(srcs['t'])):

    src_time = srcs['t'][i]
    src_bottom = src_time + timedelta(seconds= -1.*delta_t/2.)
    src_top = src_time + timedelta(seconds= delta_t/2.)
    
    mask_bottom = times > src_bottom
    mask_top = times < src_top
    mask_combined = np.array(mask_top) * np.array(mask_bottom)
    mask += mask_combined

  mask = [0 if x==1 else 1 for x in mask]
  for key in loaded.keys():
    loaded[key] = ma.compressed(ma.masked_array(loaded[key], mask=mask))

  for i in range(len(loaded['t_UTC'])):
    if loaded['run'][i] >= 120398 and loaded['run'][i] <= 126377:
      loaded['t_UTC'][i] = loaded['t_UTC'][i] + timedelta(seconds=1.)
      loaded['t_MJD'][i] = loaded['t_MJD'][i] + 1/86400.

  dumpfile = onfile[:-4] + '_final.pkl'
  pickle.dump(loaded, open(dumpfile, 'wb'))


