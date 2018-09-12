import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import pickle, time
t0 = time.time()

### Load MC file ###
print "Loading MC..."
filename = '../../../mc/Level2pass2_allSim.pkl'
with open(filename, 'rb') as f: g = pickle.load(f)
print "\t ...complete: %i s"%(time.time()-t0)

### Prepare features ###
zenith = g['zenith']
logl = g['logl']
rlogl = g['rlogl']
energy = g['energy']
sep = g['sep']
mask1, mask2, mask3, mask4 = zenith > -1, energy > -1, logl > -1, rlogl > -1
mask = mask1 * mask2 * mask3 * mask4
zenith, logl, rlogl, energy, sep = zenith[mask], logl[mask], rlogl[mask], energy[mask], sep[mask]

# This is where gamma will affect the fits
ow = g['oneweight_by_ngen'][mask] * np.power(g['true_energy'][mask], -2)
weight = ow / (sep)
target = np.log(sep)

print "Arranging features..."
features = np.array([[zenith[i], energy[i], logl[i], rlogl[i]] for i in range(len(zenith))])
features, test_f, target, test_t, weight, test_w = train_test_split(features, target, weight, train_size=0.5)
print "\t ...complete: %i s"%(time.time()-t0)

MyTree = RandomForestRegressor(n_estimators = 20,
                               max_depth = 42,
                               max_features = 1,
                               n_jobs=10,
                               verbose=3,
                               random_state=2018)
MyTree.fit(features, target, sample_weight=weight)

joblib.dump(MyTree, 'optimized_forest_E-2.0.pkl')

print "Total time = %i s"%(time.time()-t0)

