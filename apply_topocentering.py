from topocentering import *
from datetime import datetime
from datetime import timedelta
import pickle

icecube = observer( np.radians(0.), np.radians(-90.), 2835 )
arecibo = observer( -1.*np.radians(66. + 45./60 + 10./3600), np.radians(18. + 20./60 + 39./3600), 497 )
parkes = observer( np.radians(148.2614), -1.*np.radians(32.9994), 414.8 )
green_bank = observer( -1.*np.radians(79.8399), np.radians(38.4330), 800 )
utmost = observer( np.radians( 149. + 25./60 + 26./3600  ), -1.*np.radians( 35. + 22./60 + 15./3600 ), 732 )
askap = observer( np.radians( 116. + 38./60 + 13./3600 ), -1.*np.radians( 26. + 41./60 + 46./3600 ), 125 )


with open('bg_srcs_noRepeater.pkl', 'rb') as f: srcs = pickle.load(f)
for i in range(len(srcs['FRB'])):

  frb = sources( srcs['RA'][i], srcs['Dec'][i], srcs['t'][i] )
  if srcs['telescope'][i] in ['parkes', 'Parkes']: tele = parkes
  elif srcs['telescope'][i] == 'UTMOST': tele = utmost
  elif srcs['telescope'][i] == 'ASKAP': tele = askap
  elif srcs['telescope'][i] == 'GBT': tele = green_bank

  new_time = srcs['t'][i] + timedelta(seconds= topo_correction( tele, icecube, frb )[0])
  srcs['t'][i] = new_time

pickle.dump(srcs, open('bg_srcs_noRepeater_topocentered.pkl', 'wb'))
