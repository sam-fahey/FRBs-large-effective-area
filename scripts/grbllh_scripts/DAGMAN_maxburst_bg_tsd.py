import pycondor, argparse, sys, os.path
from glob import glob
import numpy as np

t100 = 1000.00 
N_trials = 1000

error='/home/sfahey/condor/error'
output='/home/sfahey/condor/output'
log='/home/sfahey/condor/log'
submit='/home/sfahey/condor/submit'

job = pycondor.Job('FRB_maxburst_bg','frb_bg_tsd_maxburst_noRepeater.py',
                   error=error,
                   output=output,
                   log=log,
                   submit=submit,
                   verbose=2,
                   request_memory=4000
                   )

for seed in range(100):
    job.add_arg('-t %.2f -r %i -n %i'%(t100, seed, N_trials))

dagman = pycondor.Dagman('FRB_maxburst_bg_dT%.2f'%t100, submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()