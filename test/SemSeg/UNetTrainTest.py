import os
from core.SemSeg.UNet import *

ROOTDIR = os.path.dirname(os.path.abspath(__file__))
tfRecordPath=ROOTDIR + "/data/tfRecord/trainData.tfrecords"
summaryPath=ROOTDIR + "/summary"
ckptPath=ROOTDIR + "/ckpt/vgg16_unet.ckpt"

uNet=UNet()
uNet.open()

uNet.build(512,512,1,2)
uNet.train(tfRecordPaths=[tfRecordPath]
           ,summaryPath=summaryPath
           ,ckptPath=ckptPath
           )

uNet.close()