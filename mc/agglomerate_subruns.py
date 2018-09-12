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
    with open(name, 'rb') as f:
        return pickle.load(f)

'''
morepath = sys.argv[1] # expects target dir, like "pass2_lowE/1/"
if 'lowE' in morepath: typeE = 'lowE'
elif 'medE' in morepath: typeE = 'medE'
else: print 'Error: typeE not recognized'; sys.exit()
numdir = morepath[-2]
'''

currentpath = '/data/user/sfahey/FRB/L2_Analysis/mc/'
savepath= currentpath+'Level2pass2_allSim' # saves to location
lowfilelist = glob(currentpath+"pass2_lowE/*/*")
medfilelist = glob(currentpath+"pass2_medE/*/*")

#print "Found %i files in %s"%(len(filelist), datapath)

fdict = {}
lowow, medow, logl, rlogl = np.array([]), np.array([]), np.array([]), np.array([])
en, zen, az, err = np.array([]), np.array([]), np.array([]), np.array([])
pen, pzen, paz, ptype = np.array([]), np.array([]), np.array([]), np.array([])

lown_gen = 0.
for i in range(len(lowfilelist)):
    sys.stdout.flush()
    sys.stdout.write('Adding lowE file %i of %i\r'%(i+1, len(lowfilelist)))

    f = tables.open_file(lowfilelist[i])
    try: i3wd = f.root.I3MCWeightDict.cols
    except tables.exceptions.NoSuchNodeError:
        with open('agglomerate_subruns_errors.txt','a') as wf: wf.write(lowfilelist[i]+'\n'); wf.close()
        print "\nError! Something's up with %s."%lowfilelist[i]; f.close(); continue
    smpe = f.root.SplineMPErecommended.cols
    smfp = f.root.SplineMPErecommendedFitParams.cols
    mpef = f.root.MPEFitMuEX.cols

    lowow = np.append(lowow, np.array(i3wd.OneWeight))
    logl = np.append(logl, np.array(smfp.logl))
    rlogl = np.append(rlogl, np.array(smfp.rlogl))
    en = np.append(en, np.array(mpef.energy))
    zen = np.append(zen, np.array(smpe.zenith))
    az = np.append(az, np.array(smpe.azimuth))
    pen = np.append(pen, np.array(i3wd.PrimaryNeutrinoEnergy))
    pzen = np.append(pzen, np.array(i3wd.PrimaryNeutrinoZenith))
    paz = np.append(paz, np.array(i3wd.PrimaryNeutrinoAzimuth))
    ptype = np.append(ptype, np.array(i3wd.PrimaryNeutrinoType))
    err = np.append(err, np.hypot(f.root.SplineMPEParaboloid2FitParams.cols.err1, f.root.SplineMPEParaboloid2FitParams.cols.err2)/np.sqrt(2))

    lown_gen += i3wd.NEvents[0]

    f.close()

print "\nDone with lowE; Starting medE."
medn_gen = 0.
for i in range(len(medfilelist)):
    sys.stdout.flush()
    sys.stdout.write('Adding lowE file %i of %i\r'%(i+1, len(medfilelist)))

    f = tables.open_file(medfilelist[i])
    try: i3wd = f.root.I3MCWeightDict.cols
    except tables.exceptions.NoSuchNodeError:
        with open('agglomerate_subruns_errors.txt','a') as wf: wf.write(medfilelist[i]+'\n'); wf.close()
        print "\nError! Something's up with %s."%medfilelist[i]; f.close(); continue
    smpe = f.root.SplineMPErecommended.cols
    smfp = f.root.SplineMPErecommendedFitParams.cols
    mpef = f.root.MPEFitMuEX.cols

    medow = np.append(medow, np.array(i3wd.OneWeight))
    logl = np.append(logl, np.array(smfp.logl))
    rlogl = np.append(rlogl, np.array(smfp.rlogl))
    en = np.append(en, np.array(mpef.energy))
    zen = np.append(zen, np.array(smpe.zenith))
    az = np.append(az, np.array(smpe.azimuth))
    pen = np.append(pen, np.array(i3wd.PrimaryNeutrinoEnergy))
    pzen = np.append(pzen, np.array(i3wd.PrimaryNeutrinoZenith))
    paz = np.append(paz, np.array(i3wd.PrimaryNeutrinoAzimuth))
    ptype = np.append(ptype, np.array(i3wd.PrimaryNeutrinoType))
    err = np.append(err, np.hypot(f.root.SplineMPEParaboloid2FitParams.cols.err1, f.root.SplineMPEParaboloid2FitParams.cols.err2)/np.sqrt(2))

    medn_gen += i3wd.NEvents[0]

    f.close()

lowow = lowow / lown_gen
medow = medow / medn_gen
ow = np.r_[lowow, medow]

print "\nComplete."
print "Number of events = %i"%len(ow)

# every object that I think might be useful for analysis
fdict['oneweight_by_ngen'] = ow
fdict['logl'] = logl
fdict['rlogl'] = rlogl
fdict['energy'] = en
fdict['zenith'] = zen
fdict['azimuth'] = az
fdict['true_energy'] = pen
fdict['true_zenith'] = pzen
fdict['true_azimuth'] = paz
fdict['primary_type'] = ptype
fdict['err'] = err

print "Saving events to %s.pkl"%savepath
save_obj(fdict, savepath)

