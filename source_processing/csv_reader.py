import csv, time, pickle
import numpy as np
from datetime import datetime
import icecube
from icecube import astro
from astropy.time import Time

def float_else_neg(instr):
  try: return float(instr)
  except ValueError: return -1. 

def declination(Dec):
  try: # split objects like '-12:34:56.7'
    degrees, minutes, seconds = Dec.split(':')
    dec = float(degrees) + float(minutes)/60. + float(seconds)/3600.
    return dec * (np.pi/180.) # degrees to radians
  except ValueError: # split objects like '-12:34'
    degrees, minutes = Dec.split(':')
    dec = float(degrees) + float(minutes)/60.
    return dec * (np.pi/180.) # degrees to radians

def right_ascension(RA):
  hours, minutes, seconds = RA.split(':')
  ra = float(hours) + float(minutes)/60. + float(seconds)/3600.
  return ra * (2*np.pi / 24.) # hours to radians

def rm_plusminus(value):
  try: return float(value)
  except ValueError:
    value, other = value.split('&')
    return float(value)

# needs time 't', beam semimajor axis 'sigma'
bg_src={'FRB':[], 't':[], 'telescope':[], 
        'beam_semimajor_minutes':[], 'sigma':[], 'RA':[], 'Dec':[], 
        'DM_mwlimit':[], 'DM':[], 'duration_ms':[], 'flux':[],
        'zenith':[], 'azimuth':[]}

with open('frbcat_180413.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    reader.next()
    for row in reader:
        bg_src['FRB'].append(row[0])
        bg_src['t'].append(datetime.strptime(row[1]+'000', '%Y/%m/%d %H:%M:%S.%f'))
        bg_src['telescope'].append(row[2])
        bg_src['beam_semimajor_minutes'].append(float_else_neg(row[3]))
        bg_src['RA'].append(right_ascension(row[4]))
        bg_src['Dec'].append(declination(row[5]))
        bg_src['DM_mwlimit'].append(rm_plusminus(row[6]))
        bg_src['DM'].append(rm_plusminus(row[7]))
        bg_src['duration_ms'].append(float(row[8]))
        bg_src['flux'].append(float(row[9]))

for i in range(len(bg_src['FRB'])):
    zenith, azimuth = astro.equa_to_dir(bg_src['RA'][i], bg_src['Dec'][i], Time(bg_src['t'][i], format='datetime').mjd)
    bg_src['zenith'].append(zenith)
    bg_src['azimuth'].append(azimuth)
    bg_src['sigma'].append(bg_src['beam_semimajor_minutes'][i]*(1./60)*(np.pi/180.))


# Making a list of FRBs after May 30th, 2010 (when we have data) and NOT the repeater (for separate stacking test)
isdata = np.array(bg_src['t']) > datetime(2010, 05, 30, 0, 0, 0)
notrep = np.array(bg_src['FRB']) != 'FRB121102'
mask = isdata * notrep

savedict = {'FRB':np.array(bg_src['FRB'])[mask],
            'telescope':np.array(bg_src['telescope'])[mask],
            't':np.array(bg_src['t'])[mask],
            'zenith':np.array(bg_src['zenith'])[mask],
            'azimuth':np.array(bg_src['azimuth'])[mask],
            'RA':np.array(bg_src['RA'])[mask],
            'Dec':np.array(bg_src['Dec'])[mask],
            'sigma':np.array(bg_src['sigma'])[mask]}
            
savedict['sigma'][5] = 0.1

#save source dictionary for grbllh analysis
with open('bg_srcs_noRepeater.pkl', 'wb') as f: pickle.dump(savedict, f, pickle.HIGHEST_PROTOCOL)

# creates source code for wiki-table
# paste the string output into documentation
string = ''
for i in range(len(savedict['FRB'])):
  string+= '| %s\n'%np.array(bg_src['FRB'])[mask][i].replace('B', 'B ')
  string+= '| %s\n'%np.array(bg_src['t'])[mask][i]
  string+= '| %.2f\n'%np.array(bg_src['duration_ms'])[mask][i]
  string+= '| %.2f\n'%(np.array(bg_src['Dec'])[mask][i] * 180./np.pi)
  string+= '| %.2f\n'%(np.array(bg_src['RA'])[mask][i] * 180./np.pi)
  string+= '| %s\n'%np.array(bg_src['telescope'])[mask][i].replace('parkes', 'Parkes')
  string+= '|   \n'
  string+= '|   \n'
  string+= '|- \n'

print string

