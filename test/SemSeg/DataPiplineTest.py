import os
import core.SemSeg.DataPipline as DP
#
ROOTDIR = os.path.dirname(os.path.abspath(__file__))
#
sourcePath = ROOTDIR + "/data/train"
labelPath = ROOTDIR + "/data/label"
outAugPath=ROOTDIR+"/data/outAug"
trainAugPath=ROOTDIR + "/data/trainAug"
labelAugPath=ROOTDIR + "/data/labelAug"

tfRecordPath=ROOTDIR + "/data/tfRecord/trainData.tfrecords"
# DP.makeTFRecord(sourcePath,labelPath,outAugPath,trainAugPath,labelAugPath,tfRecordPath,3000,w=512,h=512)
DP.convert2TFRecord(trainPath=trainAugPath,labelPath=labelAugPath,tfRecordPath=tfRecordPath,w=512,h=512)
# DP.checkTFRecord(tfRecordPath,512,512,1)