import test.DataSetLoader as DataLoader
import numpy as NP
from core.LinearModel.LinearRegression import *

def polyBaseFunction(x):
    row=x.shape[0]
    powArg=TF.constant([[i] for i in range(row)])
    return TF.math.pow(x,powArg)

oData,oTarget=DataLoader.loadIris()
data=NP.array([[x[3]] for x in oData])
target=NP.array([[x[0]] for x in oData])
modelLR=LinearRegression()
modelLR.open()
modelLR.build(learningRate=0.01,shape=[1,1],reguArg=0.001)
modelLR.train(data=data,target=target,batchSize=25,turn=2000)
modelLR.close()

