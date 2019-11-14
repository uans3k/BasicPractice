import core.Tool.BaseFunction as BF
import numpy as NP
def normTest():
    a=NP.array([[1,2,3],[2,5,4],[5,6,7]])
    aNorm=BF.normalize(a,axis=0)
    print(a)
    print(aNorm)
normTest()