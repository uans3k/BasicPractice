from core.SparseKernel.KernelFunction import *
import tensorflow as  TF
def gaussianTest():
    sess=TF.Session()
    kFunc=kernelGaussian(-50.)
    a=TF.constant([[1.,1.],[2.,2.]])
    kMatrix=kFunc(a,a)
    kMatrixValue=sess.run(kMatrix)
    print(kMatrixValue)
    sess.close()

gaussianTest()