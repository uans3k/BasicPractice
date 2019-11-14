import os
from core.SemSeg.UNet import *

ROOTDIR = os.path.dirname(os.path.abspath(__file__))
testPath=ROOTDIR + "/data/test/"
ckptPath=ROOTDIR + "/ckpt/vgg16_unet.ckpt-7900"

uNet=UNet()
uNet.open()

uNet.build(512,512,1,2)

uNet.restore(ckptPath)
uNet.predict(testPath+"0.tif")

uNet.close()