import pycondor, argparse, sys, os.path
from glob import glob
import numpy as np

# submit signal tests to cluster for stacking analysis
# scan each time-window of analysis, specify a spectral gamma

times = ['0.01', '0.03', '0.10', '0.32', '1.00', '3.16', '10.00', '31.60', '100.00', '316.00']
gamma = 3.0

error='/home/sfahey/condor/error'
output='/home/sfahey/condor/output'
log='/home/sfahey/condor/log'
submit='/home/sfahey/condor/submit'

job = pycondor.Job('FRB_stacking_sig','frb_sig_tsd_stacking_noRepeater.py',
                   error=error,
                   output=output,
                   log=log,
                   submit=submit,
                   verbose=2,
                   request_memory=7000
                   )

for time in times:
    job.add_arg('-t %.2f -g %.2f'%(float(time), gamma))

dagman = pycondor.Dagman('FRB_stacking_sig', submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()
