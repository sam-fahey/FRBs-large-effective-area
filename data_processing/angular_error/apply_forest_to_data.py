from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import pickle, time, glob
import numpy as np
t0 = time.time()

# Loading analysis data
datafiles = sorted(glob.glob('/data/user/sfahey/FRB/L2_Analysis/data/20*/ontime/*.pkl'))

# Loading random forest
MyTree = joblib.load('optimized_forest_E-2.0.pkl')
print "Forest loaded in %i s"%(time.time()-t0)

# Load separate files into unified array
# Mask corrupted values as normal
for filename in datafiles:
  t1 = time.time()
  with open(filename, 'rb') as f: g = pickle.load(f)

  zenith = g['zenith']
  logl = g['logl']
  rlogl = g['rlogl']
  energy = g['energy']
  mask1, mask2, mask3, mask4 = zenith > -1, energy > -1, logl > -1, rlogl > -1
  mask = mask1 * mask2 * mask3 * mask4
  zenith, logl, rlogl, energy = zenith[mask], logl[mask], rlogl[mask], energy[mask]
  print "Arranging features..."
  features = np.array([[zenith[i], energy[i], logl[i], rlogl[i]] for i in range(len(zenith))])
  print "\t ...complete: %i s"%(time.time()-t0)

  for key in g.keys():
    g[key] = g[key][mask]

  g['sigma'] = np.exp(MyTree.predict(features))/1.177 # convert 50%-containment to Gaussian sigma

  with open(filename, 'wb') as f: pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
  print "%i s, File complete: "%(time.time()-t1), filename

#print time.time()-t0
