import test.DataSetLoader as DataLoader
from core.SparseKernel.SVM import *
import numpy as NP
from core.SparseKernel.KernelFunction import *

oData,oTarget=DataLoader.loadCircle()
data=oData
target=NP.array([[1] if t==1 else [-1] for t in oTarget])

svmModel=SVM()
svmModel.open()
svmModel.build(data=data,target=target,kernelFunc=kernelGaussian(-50),slackScale=1/data.shape[0],penaltyFactor=0.2,learningRate=0.005)
svmModel.train(turn=20000,checkTurn=100)
svmModel.close()
