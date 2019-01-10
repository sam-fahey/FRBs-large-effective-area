import pycondor, argparse, sys, os.path
from glob import glob
import numpy as np

# submit background tests to cluster for stacking analysis
# specify number of jobs, trials per job, and the time-window of analysis

t100 = 1000.00 
N_trials = 1000
N_jobs = 100

error='/home/sfahey/condor/error'
output='/home/sfahey/condor/output'
log='/home/sfahey/condor/log'
submit='/home/sfahey/condor/submit'

job = pycondor.Job('FRB_stacking_bg','frb_bg_tsd_stacking_noRepeater.py',
                   error=error,
                   output=output,
                   log=log,
                   submit=submit,
                   verbose=2,
                   request_memory=4000
                   )

for seed in range(N_jobs):
    job.add_arg('-t %.2f -r %i -n %i'%(t100, seed, N_trials))

dagman = pycondor.Dagman('FRB_stacking_bg_dT%.2f'%t100, submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()
