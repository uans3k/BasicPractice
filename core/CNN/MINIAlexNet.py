import tensorflow as TF
import numpy as NP
import matplotlib.pyplot as PLT
import scipy as SCI
import cv2 as CV
import os
ROOTDIR = os.path.dirname(os.path.abspath(__file__))

class MINIAlexNet:
    def __init__(self):
        self._graph=TF.Graph()
        self._resultDir=ROOTDIR+"/model/"
        if not os.path.isdir(self._resultDir):
            os.makedirs(self._resultDir)

    def _mininet(self, graph,inputShape,inputName="mininet_input",outputName="mininet_output",scope="mininet", convFeatures1=32, convFeatures2=64, fcFeatures1=512, fcFeatures2=256, mkeepProb=0.5):
        with graph.as_default():
            with TF.variable_scope(scope,reuse=TF.AUTO_REUSE):
                input=TF.placeholder(shape=[None,inputShape[0],inputShape[1],inputShape[2]],dtype=TF.float32,name=inputName)
                inputShape=input.get_shape().as_list()
                channels=inputShape[3]
                # layer 1 conv relu maxpool LRN
                w1 = TF.get_variable(initializer=TF.truncated_normal([5, 5, channels,convFeatures1]
                                                                    ,stddev=0.1
                                                                    ,dtype=TF.float32)
                                     ,name="d_w_1")
                b1 = TF.get_variable(initializer=TF.zeros([convFeatures1], dtype=TF.float32), name="d_b_1")
                conv1 = TF.nn.conv2d(input, filter=w1, strides=[1, 1, 1, 1], padding="SAME")
                relu1 = TF.nn.relu(TF.nn.bias_add(conv1, b1))
                maxpool1 = TF.nn.max_pool2d(relu1, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
                output1 = TF.nn.lrn(maxpool1, depth_radius=5, bias=2.0, alpha=1e-3,
                                    beta=0.75, name='d_output_1')

                # layer 2 conv relu maxpool LRN reshape [batchSize,-1]
                w2 = TF.get_variable(initializer=TF.truncated_normal([5, 5, convFeatures1, convFeatures2]
                                                                     , stddev=0.1
                                                                     , dtype=TF.float32
                                                                     )
                                     ,name="d_w_2"
                                     )
                b2 = TF.get_variable(initializer=TF.zeros([convFeatures2], dtype=TF.float32), name="d_b_2")
                conv2 = TF.nn.conv2d(output1, filter=w2, strides=[1, 1, 1, 1], padding="SAME")
                relu2 = TF.nn.relu(TF.nn.bias_add(conv2, b2))
                maxpool2 = TF.nn.max_pool2d(relu2, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
                output2 = TF.nn.lrn(maxpool2, depth_radius=5, bias=2.0, alpha=1e-3,
                                    beta=0.75, name='d_output_1')

                # layer3 full-connection relu
                output2Shape=output2.get_shape().as_list()
                output2ReshapedDim = output2Shape[1] * output2Shape[2] * output2Shape[3]
                output2Reshaped=TF.reshape(output2,shape=[-1,output2ReshapedDim])

                w3 = TF.get_variable(initializer=TF.truncated_normal(shape=[output2ReshapedDim, fcFeatures1]
                                                                     , stddev=0.1
                                                                     , dtype=TF.float32
                                                                     )
                                     ,name="d_w_3"
                                     )
                b3 = TF.get_variable(initializer=TF.zeros(shape=[fcFeatures1], dtype=TF.float32), name="d_b_3")
                wb3 = TF.nn.bias_add(TF.matmul(output2Reshaped, w3), b3)
                output3 = TF.nn.relu(wb3, name="d_output_3")

                # layer4 full-connection
                w4 = TF.get_variable(initializer=TF.truncated_normal([fcFeatures1, fcFeatures2]
                                                                     , stddev=0.1
                                                                     , dtype=TF.float32
                                                                     )
                                     ,name="d_w_4"
                                     )
                b4 = TF.get_variable(initializer=TF.zeros(shape=[fcFeatures2], dtype=TF.float32), name="d_b_4")
                wb4 = TF.nn.bias_add(TF.matmul(output3, w4), b4)
                output4 = TF.nn.relu(wb4, name="d_output_4")

                # layer5 sigmoid logic-regression classify
                w5 = TF.get_variable(initializer=TF.truncated_normal([fcFeatures2, 1]
                                                                     , stddev=0.1
                                                                     , dtype=TF.float32
                                                                     )
                                     ,name="d_w_5"
                                     )

                b5 = TF.get_variable(initializer=TF.zeros(shape=[1], dtype=TF.float32), name="d_b_5")
                wb5 = TF.nn.bias_add(TF.matmul(output4, w5), b5)

                output = TF.sigmoid(wb5, name=outputName)
                # self._check = output
        return output
    def load(self):
        pass

    def build(self):


        # dLearnRateDecay = TF.train.exponential_decay
        with self._graph.as_default():
            #build generator
            self._gInput,self._gOutput = _mininet(self._graph,noiseDim=noiseDim,outputShape=gOutputShape)

            #build discriminator for fake and real
            self._dRealInput = TF.placeholder(shape=[None, gOutputShape[0], gOutputShape[1], gOutputShape[2]],
                                        dtype=TF.float32)
            dFakeOutput = self._discriminator(self._graph, input=self._gOutput)
            dRealOutput = self._discriminator(self._graph,input=self._dRealInput)


            #build loss function
            self._dLoss = -TF.reduce_mean(TF.log(dRealOutput+1e-8)+TF.log(1-dFakeOutput+1e-8))
            self._gLoss = -TF.reduce_mean(TF.log(dFakeOutput+1e-8))

            # vList=TF.trainable_variables()
            dVList = TF.trainable_variables(scope="discriminator")
            gVList = TF.trainable_variables(scope="generator")


            #build train step
            self._dStep=TF.train.GradientDescentOptimizer(dLearnRate,name="dStep").minimize(loss=self._dLoss,var_list=dVList)
            self._gStep=TF.train.GradientDescentOptimizer(gLearnRate,name="gStep").minimize(loss=self._gLoss,var_list=gVList)

            self._hasBuild=True


    def train(self, data,batchSize=128,epoch=50000, dStepNum=1,resultName="default_" ,isSaveInnerRes=True, isRetrain=False):
        if self._hasBuild ==False :
            raise RuntimeError("model need build or load")



        with self._graph.as_default() :
            with TF.Session(graph=self._graph) as sess:
                if isRetrain == False:
                    sess.run(TF.initialize_all_variables())

                print("start::")

                # z = self._sampleZ(shape=[9, self._noiseDim])
                # fakeOutput = sess.run(self._gOutput, feed_dict={self._gInput: z})
                # xIndex = NP.random.choice(len(data), size=9)
                # x = data[xIndex]
                # self._saveResult(x,"o")

                # train discriminator pre
                # z = self._sampleZ(shape=[batchSize, self._noiseDim])
                # xIndex = NP.random.choice(len(data), size=batchSize)
                # x = data[xIndex]
                # dLoss = 0
                # for j in range(dStepNum):
                #     _, dLoss = sess.run([self._dStep, self._dLoss],
                #                         feed_dict={self._gInput: z, self._dRealInput: x})
                # print("discriminator loss pre: " + str(dLoss))
                sampleDim = self._noiseDim * self._gOutputSahpe[2]
                for i in range(epoch):


                    # train discriminator
                    z = self._sampleZ(shape=[batchSize, sampleDim])
                    xIndex = NP.random.choice(len(data), size=batchSize)
                    x = data[xIndex]
                    dLoss=0
                    for j in range(dStepNum):
                        _, dLoss = sess.run([self._dStep, self._dLoss],
                                            feed_dict={self._gInput: z, self._dRealInput: x})


                    # train generator
                    z = self._sampleZ(shape=[batchSize, self._noiseDim])
                    _, gLoss = sess.run([self._gStep, self._gLoss], feed_dict={self._gInput: z})

                    if ((i + 1) % 100) == 0:
                        print("epoch :" + str(i+1))
                        print("discriminator loss : " + str(dLoss))
                        print("generator loss : " + str(gLoss))
                    #
                    if isSaveInnerRes:
                        if ((i + 1) % 2000) == 0:
                            n = (i + 1) / 2000
                            z = self._sampleZ(shape=[9, sampleDim])
                            fakeOutput = sess.run(self._gOutput, feed_dict={self._gInput: z})
                            self._saveResult(fakeOutput,name=resultName+str(n*2000))


