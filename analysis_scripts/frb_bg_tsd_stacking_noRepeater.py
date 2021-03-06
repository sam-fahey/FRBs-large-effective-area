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

try_one_source = False 

# Add options to access a variety of time windows when running jobs on the cluster
usage = "%prog [options] <time>"
time0 = time.time()
parser = OptionParser(usage=usage)

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

parser.add_option(
    "-n", "--N_trials",
    type = "float",
    default = 1e6,
    metavar = "<Specify N_trials>",
    help = "Number of trials"
    )

(options, args) = parser.parse_args()

r = options.randomseed
dT = options.centertime # Search time window
N = options.N_trials
print "Time Window in Seconds: ", dT

#
# LOAD BACKGROUND AND MC
#
print 'Loading files.'

files = sorted(glob('/data/user/sfahey/FRB/L2_Analysis/data/20*/offtime/*decimated.pkl'))

with open('/data/user/sfahey/FRB/L2_Analysis/mc/Level2pass2_allSim.pkl') as f:
    mc = pickle.load(f)

bg = {}
zenith, azimuth, livetime = np.array([]), np.array([]), np.array([])
energy, t_UTC, run, sigma = np.array([]), np.array([]), np.array([]), np.array([])

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
livetime, run, t_UTCi, sigma = livetime[mask], run[mask], t_UTC[mask], sigma[mask]

bg['zenith'], bg['azimuth'], bg['energy'] = zenith, azimuth, energy
bg['livetime'], bg['run'], bg['t'] = livetime, run, t_UTC
bg['livetime_per_run'], bg['sigma'] = livetime, sigma

if try_one_source: frbs = {'zenith':[0.], 'azimuth':[0.], 'sigma':[0.001], 'duration':[0.01], 't':[datetime.datetime(2016,06,01,00,00,00,000)]}
else: 
  with open('/data/user/sfahey/FRB/L2_Analysis/bg_srcs_noRepeater.pkl', 'rb') as f: frbs = pickle.load(f)
  frbs['duration'] = np.array([dT]*len(frbs['zenith']))

#
# BEGIN ANALYSIS
#
analysis = grbllh.AutoAnalysis ('test', bg, frbs, mc,
                                pssim_frac_of_sphere=0.01,
                                sim_source_smear=False,
                                analysis_zen_range=[0,np.pi],
                                seed=options.randomseed,
                                seasonal_fit = 'per_run')

analysis.set_pdf_space_bg (bins=20, azimuth=True)

analysis.set_flat_pdf_ratio_energy()

analysis.set_config (min_r=1.0e-10, sigma_t_truncate=0, sigma_t_min=1.0e-3, sigma_t_max=1.0e7, max_delang=np.pi)

analysis.set_bg_thrower ()

t1 = time.time()
tsd_bg_tw = grbllh.do_trials(N, [analysis], llh_type=grbllh.LlhType.overall, mu=0., seed=options.randomseed)

# Store the TestStatDist object in a pickle file
if try_one_source: cache.save(tsd_bg_tw, "/data/user/sfahey/FRB/L2_Analysis/tsd/bg/tsd_oneSrc_seed%s_dT%.2f"%(str(options.randomseed).zfill(2), dT))
else: cache.save([tsd_bg_tw], "/data/user/sfahey/FRB/L2_Analysis/tsd/bg/tsd_stacking_seed%s_dT%.2f"%(str(options.randomseed).zfill(5), dT))

print "Time: ", time.time() - time0
