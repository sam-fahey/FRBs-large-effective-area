#!usr/bin/env python
import icecube
from icecube import dataclasses, dataio, icetray, hdfwriter, millipede
from I3Tray import *
from icecube.tableio import I3TableWriter
import os
import sys

# This script reads and filters IceCube's i3 files, saving an output file as an hdf5
# $ python <script_name> <'data' or 'mc'> <desired output name> <input file(s)>
# the 'mc' tag tells the script to save I3MCWeight info for effective area measurements

files=sys.argv[3:]

tray=I3Tray()

tray.AddModule('I3Reader', 'reader', FilenameList = files)

tray.AddModule(lambda frame: frame['I3EventHeader'].sub_event_stream == 'InIceSplit', 'inicesplit')

if sys.argv[1] == 'data':
    print "converting data"
    keys=['MPEFitMuEX', 'MPEFitFitParams'] 
    tray.AddModule(lambda frame: frame['QFilterMask']['MuonFilter_13'].condition_passed==1, 'muon_filter')
elif sys.argv[1] == 'mc':
    print "converting mc"
    keys=['I3MCWeightDict', 'MPEFitMuEX', 'MPEFitFitParams']
    tray.AddModule(lambda frame: frame['FilterMask']['MuonFilter_13'].condition_passed==1, 'muon_filter')
else: sys.exit()

tray.AddSegment(icecube.hdfwriter.I3HDFWriter, 'writer', 
                Output=sys.argv[2]+'.h5',
                Keys=keys,
                SubEventStreams=['InIceSplit']
                )

tray.AddModule('TrashCan', 'YesWeCan')
tray.Execute() # use tray.Execute(100) to test on new files; it will process 100 event frames to inspect output
tray.Finish()
