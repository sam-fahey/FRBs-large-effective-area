import pycondor, argparse, sys, os.path
from glob import glob
import numpy as np

times = ['0.01', '0.03', '0.10', '0.32', '1.00', '3.16', '10.00', '31.60', '100.00', '316.00']
times = ['100.00', '316.00', '1000.00'] # try large dT with fewer trials, less memory
gamma = 3.0

error='condor/error'
output='condor/output'
log='condor/log'
submit='condor/submit'

job = pycondor.Job('FRB_maxburst_sig','frb_sig_tsd_maxburst_noRepeater.py',
                   error=error,
                   output=output,
                   log=log,
                   submit=submit,
                   verbose=2,
                   request_memory=7000
                   )

for time in times:
    job.add_arg('-t %.2f -g %.2f'%(float(time), gamma))

dagman = pycondor.Dagman('FRB_maxburst_sig', submit=submit, verbose=2)
dagman.add_job(job)
dagman.build_submit()
