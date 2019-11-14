import os
from core.SemSeg.ResUNet import *

ROOTDIR = os.path.dirname(os.path.abspath(__file__))
tfRecordPath=ROOTDIR + "/data/tfRecord/trainData.tfrecords"
logPath=ROOTDIR + "/logs"
savePath=ROOTDIR + "/h5/res34_unet.hdf5"
testPath=ROOTDIR + "/data/test/"
trainPath=ROOTDIR+ "/data/train/"

resUNet=ResUNet()

resUNet.build(512,512,1,2)
resUNet.loadWeight(savePath)
resUNet.predict(trainPath+"0.tif")


