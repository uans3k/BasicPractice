import tensorflow as TF
import numpy as NP
import matplotlib.pyplot as PLT
import scipy as SCI
import cv2 as CV
import os
ROOTDIR = os.path.dirname(os.path.abspath(__file__))

class GAN:
    # ganConfig={generator :(graph,inputName,outputShape,outputName,scope)=>input*output:
    #           ,discriminator : (graph, input, outputName, scope)=>output:
    #           ,cost(generator,discriminatorFake,discriminatorReal)=>dStep,dLoss,gStep,gLoss
    #           ,noiseDim
    #           ,outputShape
    #           ,pkgDir
    #           ,resultDir
    #           }
    def __init__(self):
        self._graph=TF.Graph()

        self._noiseDim = 0
        self._gOutputSahpe = None

        self._dRealInput=None
        self._dLoss=None
        self._gLoss=None
        self._dStep=None
        self._gStep=None
        self._globalStep=None

        self._hasBuild=False



    def _saveResult(self,imgs,name):
        imgs = NP.squeeze(imgs, -1)
        PLT.clf()
        for i in range(64):
            PLT.subplot(8, 8, i + 1)
            PLT.imshow(imgs[i], cmap='Greys_r')
            PLT.axis("off")
        PLT.savefig(self._resultDir+"/"+name+".png")

    # ganConfig={generator :(graph,inputName,outputShape,outputName,scope)=>input*output:
    #           ,discriminator : (graph, input, outputName, scope)=>output:
    #           ,lossStep(generator,dFake,dReal,dLossName
    #                    ,dStepName
    #                    ,gLossName
    #                    ,gStepName
    #                    ,dScope
    #                    ,vScope)=>dLoss,dStep,gLoss,gStep,
    #           ,noiseDim
    #           ,outputShape
    #           ,pkgDir
    #           ,resultDir
    #           }
    def build(self,ganConfig):
        self._noiseDim=ganConfig["noiseDim"]
        self._gOutputSahpe=ganConfig["outputShape"]
        self._pkgDir = ganConfig["pkgDir"]
        self._resultDir=ganConfig["resultDir"]

        if not os.path.isdir(self._resultDir):
            os.makedirs(self._resultDir)
        if not os.path.isdir(self._pkgDir):
            os.makedirs(self._pkgDir)


        with self._graph.as_default():
            #build generator
            discriminator=ganConfig["discriminator"]
            generator=ganConfig["generator"]
            lossStep=ganConfig["lossStep"]
            self._gInput,self._gOutput =generator(graph=self._graph
                                                  ,outputShape=self._gOutputSahpe
                                                  ,noiseDim=self._noiseDim
                                                  ,inputName="g_input"
                                                  ,outputName="g_output"
                                                  ,scope="generator"
                                                  )

            #build discriminator for fake and real
            self._dRealInput = TF.placeholder(shape=[None, self._gOutputSahpe[0], self._gOutputSahpe[1], self._gOutputSahpe[2]],
                                        dtype=TF.float32)

            dFakeOutput = discriminator(graph=self._graph,input=self._gOutput, outputName="d_fake_output", scope="discriminator")
            dRealOutput = discriminator(graph=self._graph,input=self._dRealInput, outputName="d_real_output", scope="discriminator")


            #build loss step
            self._globalStep=TF.Variable(0,trainable=False,name="global_step")
            self._dLoss,self._dStep,self._gLoss,self._gStep,=lossStep(graph=self._graph
                                                                      ,dReal=dRealOutput
                                                                      , dFake=dFakeOutput
                                                                      , dLossName="d_loss"
                                                                      , dStepName="d_step"
                                                                      , gLossName="d_loss"
                                                                      , gStepName="d_step"
                                                                      , dScope="discriminator"
                                                                      , gScope="generator"
                                                                      , globalStep=self._globalStep
                                                                      )



            self._hasBuild=True


    def train(self, data,batchSize=100,epoch=20000, dStepNum=1,gStepNum=1,resultName="default_" ,isSaveInnerRes=True, isRetrain=False):
        if self._hasBuild ==False :
            raise RuntimeError("model need build or load")

        with self._graph.as_default() :
            with TF.Session(graph=self._graph) as sess:
                if isRetrain == False:
                    sess.run(TF.initialize_all_variables())

                print("start::")

                # z = self._sampleZ(shape=[9, self._noiseDim])
                # fakeOutput = sess.run(self._gOutput, feed_dict={self._gInput: z})
                # xIndex = NP.random.choice(len(data), size=64)
                # x = data[xIndex]
                # self._saveResult(x,"o")



                # train discriminator pre

                # for j in range(20):
                #     z = self._sampleZ(shape=[batchSize, sampleDim])
                #     xIndex = NP.random.choice(len(data), size=batchSize)
                #     x = data[xIndex]
                #     dLoss = 0
                #     _, dLoss = sess.run([self._dStep, self._dLoss],
                #                         feed_dict={self._gInput: z, self._dRealInput: x})
                # print("discriminator loss pre: " + str(dLoss))

                zTest = self._sampleZ(shape=[64, self._noiseDim])

                for i in range(epoch):

                    # train discriminator
                    dLoss=0
                    z = self._sampleZ(shape=[batchSize, self._noiseDim])
                    for j in range(dStepNum):
                        # z = self._sampleZ(shape=[batchSize, self._noiseDim])
                        xIndex = NP.random.choice(len(data), size=batchSize)
                        x = data[xIndex]
                        _, dLoss = sess.run([self._dStep, self._dLoss],
                                            feed_dict={self._gInput: z, self._dRealInput: x,self._globalStep:i})


                    # train generator
                    gLoss=0
                    for j in range (gStepNum):
                        # z = self._sampleZ(shape=[batchSize, self._noiseDim])
                        _, gLoss = sess.run([self._gStep, self._gLoss], feed_dict={self._gInput: z,self._globalStep:i})

                    if ((i + 1) % 100) == 0:
                        print("epoch :" + str(i+1))
                        print("discriminator loss : " + str(dLoss))
                        print("generator loss : " + str(gLoss))

                    if isSaveInnerRes:
                        if ((i + 1) % 2000) == 0:
                            n = int((i + 1) / 2000)
                            fakeOutput = sess.run(self._gOutput, feed_dict={self._gInput: zTest})
                            self._saveResult(fakeOutput,name=resultName+str(n*2000))



    def genPic(self):
        pass

    def _sampleZ(self,shape):
       return NP.random.uniform(low=-NP.sqrt(3),high=NP.sqrt(3),size=shape)

    # def buildByFile(self,path):
    #     with self._graph.as_default:
    #         saver = TF.train.import_meta_graph(path)
    #         # saver.restore(self.sess, TF.train.latest_checkpoint('./'))
    #
    # def predict(self):
    #     pass
    #
    # def save(self,path):
    #     saver = TF.train.Saver()
    #     # saver.save(self.sess, path=path)


