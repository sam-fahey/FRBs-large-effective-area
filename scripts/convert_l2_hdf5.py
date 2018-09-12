#!usr/bin/env python
import icecube
from icecube import dataclasses, dataio, icetray, hdfwriter, millipede
from I3Tray import *
from icecube.tableio import I3TableWriter

import os
import sys

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
tray.Execute(100)
tray.Finish()
