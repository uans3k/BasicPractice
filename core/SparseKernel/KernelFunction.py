import  tensorflow as TF
from core.Exception.RuntimeException import  *

def kernelGaussian(gamma=-50.):
    if(gamma>0):
        raise ParamException()
    def func(xm,xn):
        xmxm = TF.reduce_sum(TF.square(xm), axis=1)
        xmxm=TF.reshape(xmxm,shape=[-1,1],name="xmxm")
        xnxn = TF.reduce_sum(TF.square(xn), axis=1)
        xnxn = TF.reshape(xnxn, shape=[-1, 1],name="xnxn")
        xmxn = TF.matmul(xm,TF.transpose(xn),name="xmxn")
        dist =TF.add(xmxm-2.*xmxn,TF.transpose(xnxn),name="dist")
        # d=TF.multiply(gamma,dist,name="d")
        kMatrix=TF.exp(gamma*dist)
        return  kMatrix
    return func
