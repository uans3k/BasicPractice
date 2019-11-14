import tensorflow as TF


def miniAlexNetDiscriminator(graph, input, outputName, scope, convFeatures1=32, convFeatures2=64,
                   fcFeatures1=512, fcFeatures2=256, mkeepProb=0.5):
    with graph.as_default():
        with TF.variable_scope(scope, reuse=TF.AUTO_REUSE):
            # input=TF.placeholder(shape=[None,inputShape[0],inputShape[1],inputShape[2]],dtype=TF.float32,name=inputName)
            inputShape = input.get_shape().as_list()
            channels = inputShape[3]
            # layer 1 conv relu maxpool LRN
            w1 = TF.get_variable(initializer=TF.truncated_normal([5, 5, channels, convFeatures1]
                                                                 , stddev=0.1
                                                                 , dtype=TF.float32)
                                 , name="d_w_1")
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
                                 , name="d_w_2"
                                 )
            b2 = TF.get_variable(initializer=TF.zeros([convFeatures2], dtype=TF.float32), name="d_b_2")
            conv2 = TF.nn.conv2d(output1, filter=w2, strides=[1, 1, 1, 1], padding="SAME")
            relu2 = TF.nn.relu(TF.nn.bias_add(conv2, b2))
            maxpool2 = TF.nn.max_pool2d(relu2, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
            output2 = TF.nn.lrn(maxpool2, depth_radius=5, bias=2.0, alpha=1e-3,
                                beta=0.75, name='d_output_1')

            # layer3 full-connection relu
            output2Shape = output2.get_shape().as_list()
            output2ReshapedDim = output2Shape[1] * output2Shape[2] * output2Shape[3]
            output2Reshaped = TF.reshape(output2, shape=[-1, output2ReshapedDim])

            w3 = TF.get_variable(initializer=TF.truncated_normal(shape=[output2ReshapedDim, fcFeatures1]
                                                                 , stddev=0.05
                                                                 , dtype=TF.float32
                                                                 )
                                 , name="d_w_3"
                                 )
            b3 = TF.get_variable(initializer=TF.zeros(shape=[fcFeatures1], dtype=TF.float32), name="d_b_3")
            wb3 = TF.nn.bias_add(TF.matmul(output2Reshaped, w3), b3)
            output3 = TF.nn.relu(wb3, name="d_output_3")

            # layer4 full-connection
            w4 = TF.get_variable(initializer=TF.truncated_normal([fcFeatures1, fcFeatures2]
                                                                 , stddev=0.1
                                                                 , dtype=TF.float32
                                                                 )
                                 , name="d_w_4"
                                 )
            b4 = TF.get_variable(initializer=TF.zeros(shape=[fcFeatures2], dtype=TF.float32), name="d_b_4")
            wb4 = TF.nn.bias_add(TF.matmul(output3, w4), b4)
            output4 = TF.nn.relu(wb4, name="d_output_4")

            # layer5 sigmoid logic-regression classify
            w5 = TF.get_variable(initializer=TF.truncated_normal([fcFeatures2, 1]
                                                                 , stddev=0.1
                                                                 , dtype=TF.float32
                                                                 )
                                 , name="d_w_5"
                                 )

            b5 = TF.get_variable(initializer=TF.zeros(shape=[1], dtype=TF.float32), name="d_b_5")
            wb5 = TF.nn.bias_add(TF.matmul(output4, w5), b5)

            output = TF.sigmoid(wb5, name=outputName)
            # self._check = output
    return output

#maxout network   Goodfellow,2013
def originDiscriminator(graph, input, outputName, scope,keepProb=0.8):
    with graph.as_default():
        with TF.variable_scope(scope, reuse=TF.AUTO_REUSE):
            #layer0  input  reshape dropout
            inputShape= input.get_shape().as_list()
            reshapeDim=inputShape[1]*inputShape[2]*inputShape[3]
            reshapeInput=TF.reshape(tensor=input,shape=[-1,reshapeDim])
            reshapeInput=TF.nn.dropout(x=reshapeInput,keep_prob=keepProb)
            # layer 1 maxout x=[batchsize,d] w=[d,n,k]
            w1=TF.Variable(TF.truncated_normal(shape=[reshapeDim, 240,5]
                                               , stddev=0.005
                                               , dtype=TF.float32
                                               )
                           ,name="d_w_1"
                          )
            b1=TF.Variable(TF.zeros([240,5]),name="d_b_1")
            output1=TF.reduce_max(TF.tensordot(reshapeInput,w1,axes=1)+b1,axis=2)

            #layer 2  x=[batchsize,d] w=[d,n,k]
            w2 = TF.Variable(TF.truncated_normal(shape=[240, 240, 5]
                                                 , stddev=0.005
                                                 , dtype=TF.float32
                                                 )
                             , name="d_w_1"
                             )
            b2 = TF.Variable(TF.zeros([240, 5]), name="d_b_1")
            output2 = TF.reduce_max(TF.tensordot(output1, w2,axes=1) + b2, axis=2)

            #output layer3 sigmoid
            w3 = TF.Variable(TF.truncated_normal(shape=[240,1]
                                                 , stddev=0.005
                                                 , dtype=TF.float32
                                                 )
                             , name="d_w_1"
                             )
            b3 = TF.Variable(TF.zeros([1]), name="d_b_1")
            output = TF.sigmoid(TF.matmul(output2, w3) + b3,name=outputName)

    return output