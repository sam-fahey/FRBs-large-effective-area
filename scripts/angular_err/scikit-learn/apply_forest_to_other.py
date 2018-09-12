from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import pickle, time, glob
import numpy as np
t0 = time.time()

MyTree = joblib.load('optimized_forest_E-2.0.pkl')
print "Forest loaded in %i s"%(time.time()-t0)

filename = '/data/user/sfahey/FRB/South_HE_Tracks/arrays/ic86ii/diff86ii.arrays'
outpath = '6yrFRB_southMC_results'

t1 = time.time()
with open(filename, 'rb') as f: g = pickle.load(f)

zenith = g['splinempe_zenith']
rlogl = g['splinempe_rlogl']
logl = rlogl*(g['n_ch']-5)
energy = g['splinempe_muex_energy']
print "Arranging features..."
features = np.array([[zenith[i], energy[i], logl[i], rlogl[i]] for i in range(len(zenith))])
print "\t ...complete: %i s"%(time.time()-t0)

g['sigma_forest'] = np.exp(MyTree.predict(features))/1.177 # convert 50%-containment to Gaussian sigma

with open(outpath, 'wb') as f: pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
print "%i s, File complete: "%(time.time()-t1), filename

print time.time()-t0
