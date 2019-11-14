import test.DataSetLoader as DataLoader
import tensorflow as TF
import numpy as NP
import core.Tool.BaseFunction as BF
import core.LinearModel.LogicRegression as LR

def polyBaseFunction(x):
    row=x.shape[0]
    powArg=TF.constant([[float(i)] for i in range(row)])
    return TF.math.pow(x,powArg)

def test():
    oData, oTarget = DataLoader.loadIris()
    data = BF.normalize(oData, axis=0)
    target = oTarget
    # print(data)
    # print(target)
    logicRegression = LR.LogicRegression()
    logicRegression.open()
    logicRegression.build(learningRate=0.05,baseFunc=polyBaseFunction,shape=[3, 4], reguArg=0.)
    logicRegression.printGraph("C://logs")
    logicRegression.train(data=data, target=target, batchSize=100, turn=10000)
    # print(data[0])
    print(logicRegression.predict([data[0]]))
    print(target[0])

    print(logicRegression.predict([data[35]]))
    print(target[35])

    print(logicRegression.predict([data[110]]))
    print(target[110])
    logicRegression.close()

test()