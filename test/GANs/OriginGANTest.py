from core.GANs.GAN import *
import test.DataSetLoader as DL
import core.GANs.GanConfig as GConfig

trainD,_,_,_=DL.loadMINIST()
# print(NP.shape(trainD))
gan=GAN()
gan.build(ganConfig=GConfig.miniAlexNetConfig)
gan.train(data=trainD)