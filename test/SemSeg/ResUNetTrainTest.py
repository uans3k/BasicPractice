import os
from core.SemSeg.ResUNet import *

ROOTDIR = os.path.dirname(os.path.abspath(__file__))
tfRecordPath=ROOTDIR + "/data/tfRecord/trainData.tfrecords"
logPath=ROOTDIR + "/logs"
savePath=ROOTDIR + "/h5/res34_unet.hdf5"

resUNet=ResUNet()

resUNet.build(512,512,1,2)
resUNet.loadWeight(savePath)
resUNet.train(tfRecordPaths=[tfRecordPath]
           ,savePath=savePath
           ,logPath=logPath
           ,batchSize=3
           ,epochs=20
           ,stepsPerEpoch=1000
           )
