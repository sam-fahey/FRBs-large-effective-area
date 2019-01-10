#!/usr/bin/env python

import numpy as np
import numpy.ma as ma
from glob import glob
import os, pylab, pickle, time, icecube, datetime, sys
from optparse import OptionParser
from os.path import expandvars
from I3Tray import I3Units
from icecube import grbllh, umdtools
from icecube.umdtools import cache
import pickle

try_one_source = False 
thresh_file = '/data/user/sfahey/FRB/L2_Analysis/tsd/bg/thresholds_maxburst.pkl'
with open(thresh_file, 'rb') as thresh_open: threshes = pickle.load(thresh_open)

# Add options to access a variety of time windows when running jobs on the cluster
usage = "%prog [options] <time>"
time0 = time.time()
parser = OptionParser(usage=usage)

parser.add_option(
    "-g", "--gamma",
    type = "float",
    default = 2.,
    metavar = "<specify signal gamma>",
    help = "spectrum of signal injection (2. gives E^-2)",
    )

parser.add_option(
    "-t", "--centertime",
    type = "float",
    default = 10.0,
    metavar = "<specify time window>",
    help = "size of the time window in seconds.",
    )

parser.add_option(
    "-r", "--randomseed",
    type = "int",
    default = 0,
    metavar = "<Specify unique radom seed>",
    help = "Random seed"
    )

(options, args) = parser.parse_args()

gamma = options.gamma
r = options.randomseed
dT = options.centertime # Search time window
print "Time Window in Seconds: ", dT

#
# LOAD BACKGROUND AND MC
#
print 'Loading files.'

files = sorted(glob('/data/user/sfahey/FRB/L2_Analysis/data/20*/offtime/*decimated.pkl'))

with open('/data/user/sfahey/FRB/L2_Analysis/mc/Level2pass2_allSim.pkl') as f: mc = pickle.load(f)

bg = {}
zenith, azimuth, livetime = np.array([]), np.array([]), np.array([])
energy, t_UTC, run = np.array([]), np.array([]), np.array([])
sigma = np.array([])
for pkl in files:
  sys.stdout.write('Opening %s\r'%pkl)
  sys.stdout.flush()
  with open(pkl, 'rb') as f: bg_file = pickle.load(f)
  zenith = np.append(zenith, bg_file['zenith'])
  azimuth = np.append(azimuth, bg_file['azimuth'])
  energy = np.append(energy, bg_file['energy'])
  livetime = np.append(livetime, bg_file['livetime'])
  run = np.append(run, bg_file['run'])
  t_UTC = np.append(t_UTC, bg_file['t_UTC'])
  sigma = np.append(sigma, bg_file['sigma'])

# some failed recos returned NaN directions/energy; mask them out for now
zmask = zenith > -1
emask = energy > -1
mask = zmask * emask
zenith, azimuth, energy = zenith[mask], azimuth[mask], energy[mask]
livetime, run, t_UTC = livetime[mask], run[mask], t_UTC[mask]

bg['zenith'], bg['azimuth'], bg['energy'] = zenith, azimuth, energy
bg['livetime'], bg['run'], bg['t'] = livetime, run, t_UTC
bg['livetime_per_run'], bg['sigma'] = livetime, sigma

if try_one_source: frbs = {'zenith':[0.], 'azimuth':[0.], 'sigma':[0.001], 'duration':[dT], 't':[datetime.datetime(2016,06,01,00,00,00,000)]}
else: 
  with open('/data/user/sfahey/FRB/L2_Analysis/bg_srcs_noRepeater.pkl', 'rb') as f: frbs = pickle.load(f)
  frbs['duration'] = np.array([dT]*len(frbs['zenith']))

#
# BEGIN ANALYSIS
#
analysis = grbllh.AutoAnalysis ('test', bg, frbs, mc,
                                pssim_frac_of_sphere=0.01,
                                sim_source_smear=True,
                                analysis_zen_range=[0,np.pi],
                                seed=options.randomseed,
                                seasonal_fit = 'per_run')
print "analysis initiated."
analysis.set_pdf_space_bg (bins=20)

analysis.set_flat_pdf_ratio_energy()

analysis.set_config (min_r=1.0e-10, sigma_t_truncate=0, sigma_t_min=1.0e-3, sigma_t_max=1.0e7, max_delang=np.pi)

analysis.set_bg_thrower ()
analysis.add_weighting('Ex', lambda E: E**-2 * (E/(100.*I3Units.TeV))**-(gamma-2.))
analysis.set_sig_throwers('Ex')

t1 = time.time()

thresh = threshes['%.2f'%dT]

mu_bottom = 1. / np.sum ([np.sum (t.prob) for t in analysis.pssig_throwers])
mu_bottom = 1.0e-1
print 'mu_bottom: ', mu_bottom
mu_top = 2. / np.sum ([np.sum (t.prob) for t in analysis.pssig_throwers])
mu_top = 1. 
print 'mu_top: ', mu_top

# Diffuse thrower sensitivity                                                                                                
mu = grbllh.find_sig_thresh (thresh[0], 0.9, 1e3, [analysis],
		t1=[0], t2=[dT],
		mu_bottom = mu_bottom,
		mu_top = mu_top,
		use_diffsig=False,
		log=True,
		llh_type=grbllh.LlhType.max_source,
		n_sig_sources = 1,
		full_output=True)
print ('Point source thrower E^-2 sensitivity: {0:.4f} (GeV cm^-2)'.format (mu['mu'] ))
cache.save(mu, "/data/user/sfahey/FRB/L2_Analysis/tsd/sig/tsd_sens_maxburst_E-%.1f_seed%s_dT%s"%(gamma, options.randomseed, options.centertime))

mu = grbllh.find_sig_thresh (thresh[3], 0.9, 1e3, [analysis],
		mu_bottom = mu_bottom*4.,
		mu_top = mu_top*4.,
		use_diffsig=False,
		log=True,
		llh_type=grbllh.LlhType.max_source,
		n_sig_sources = 1,
		full_output=True)
print ('Point source thrower 3\sigma 90% E^-2 discovery potential: {0:.4f} (GeV cm^-2)'.format (mu['mu'] ))
cache.save(mu, "/data/user/sfahey/FRB/L2_Analysis/tsd/sig/tsd_dp3sig_maxburst_E-%.1f_seed%s_dT%s"%(gamma, options.randomseed, options.centertime))


mu = grbllh.find_sig_thresh (thresh[5], 0.9, 1e3, [analysis],
		mu_bottom = mu_bottom*7.,
		mu_top = mu_top*7.,
		use_diffsig=False,
		log=True,
		llh_type=grbllh.LlhType.max_source,
		n_sig_sources = 1,
		full_output=True)
print ('Point source thrower 5\sigma 90% E^-2 discovery potential: {0:.4f} (GeV cm^-2)'.format (mu['mu'] ))
cache.save(mu, "/data/user/sfahey/FRB/L2_Analysis/tsd/sig/tsd_dp5sig_maxburst_E-%.1f_seed%s_dT%s"%(gamma, options.randomseed, options.centertime))

'''
mu = grbllh.find_sig_thresh (thresh[4], 0.9, 1e4, [analysis],
		mu_bottom = mu_bottom*3.,
		mu_top = mu_top*3.,
		use_diffsig=False,
		log=True,
		llh_type=grbllh.LlhType.max_source,
		n_sig_sources = 1,
		full_output=True)
print ('Point source thrower 4\sigma 90% E^-2 discovery potential: {0:.4f} (GeV cm^-2)'.format (mu['mu'] ))
cache.save(mu, "/data/user/sfahey/FRB/L2_Analysis/tsd/sig/tsd_dp4sig_maxburst_E-%.1f_seed%s_dT%s"%(gamma, options.randomseed, options.centertime))
'''

print "Time: ", time.time() - time0
