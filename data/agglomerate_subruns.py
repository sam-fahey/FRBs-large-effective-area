import tables, sys, pickle
from glob import glob
import numpy as np
from astropy.time import Time

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def int_else_float_of_str(s):
    f = float(s)
    i = int(f)
    return i if i == f else f

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

year = sys.argv[1]
runnum = sys.argv[2] # expects 6 digits, like '$ python <script> 127925'

datapath = '/data/user/sfahey/FRB/L2_Analysis/data/%s/offtime/'%year # location of subrun directories
savepath= datapath+'Level2pass2_Run%s_all'%str(runnum).zfill(8) # saves to location
#runlist = glob(datapath+"*")

#for run in runlist:
#    if runnum in run and 'pkl' not in run: runpath = run; break

subrunlist = sorted(glob(datapath+"*"))
subrunmask1 = ['pkl' not in x for x in subrunlist]
subrunmask2 = [runnum in x for x in subrunlist]
subrunmask = np.array(subrunmask1) * np.array(subrunmask2)
subrunlist = np.array(subrunlist)[subrunmask]

print "Found %i subruns"%(len(subrunlist))
if len(subrunlist) == 0: sys.exit()

info = {}
goodruninfos = sorted(glob('*GoodRunInfo*'))
for grifile in goodruninfos:
    with open(grifile, 'r') as gri:
        next(gri); next(gri)
        for line in gri:
            data = line.split()
            info[data[0]] = [int_else_float_of_str(val) for val in data[:4]]
livetime = info[runnum][3]

fdict = {}
tMJD, e, z, a, r, lt = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
en, logl, rlogl = np.array([]), np.array([]), np.array([])

for i in range(len(subrunlist)):
    sys.stdout.flush()
    sys.stdout.write('Adding subrun %i of %i in Run%s\r'%(i+1, len(subrunlist), str(runnum).zfill(8)))

    f = tables.open_file(subrunlist[i])
    try: i3eh = f.root.I3EventHeader.cols
    except tables.exceptions.NoSuchNodeError:
        with open('agglomerate_subruns_errors.txt','a') as wf: wf.write(subrunlist[i]+'\n'); wf.close()
        print "\nError! Something's up with %s."%subrunlist[i]; f.close(); continue
    smpe = f.root.SplineMPErecommended.cols
    smfp = f.root.SplineMPErecommendedFitParams.cols
    mpef = f.root.MPEFitMuEX.cols

    tMJD = np.append(tMJD, np.array(i3eh.time_start_mjd))
    e = np.append(e, np.array(i3eh.Event))
    z = np.append(z, np.array(smpe.zenith))
    a = np.append(a, np.array(smpe.azimuth))
    en = np.append(en, np.array(mpef.energy))
    r = np.append(r, np.array(i3eh.Run))
    lt = np.append(lt, np.array([livetime]*len(i3eh.Run)))
    logl = np.append(logl, np.array(smfp.logl))
    rlogl = np.append(rlogl, np.array(smfp.rlogl))
    f.close()

tUTC = np.array([Time(t, format='mjd').datetime for t in tMJD])

print "\nComplete."
print "Number of events = %i"%len(z)
print "Livetime = %.2f s"%livetime
print len(z)/livetime, "Hz"

# every object that I think might be useful for analysis
fdict['t_MJD'] = tMJD
fdict['t_UTC'] = tUTC
fdict['event'] = e
fdict['zenith'] = z
fdict['azimuth'] = a
fdict['energy'] = en
fdict['run'] = r
fdict['livetime'] = lt
fdict['logl'] = logl
fdict['rlogl'] = rlogl

print "Saving events to %s.pkl"%savepath
save_obj(fdict, savepath)

