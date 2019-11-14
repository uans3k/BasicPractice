import tensorflow as TF




def originGenrator(graph, outputShape,noiseDim ,inputName, outputName,scope
                   , featureDim=1200, isDropout=False, keepProb=0.8):
    with graph.as_default():
        with TF.variable_scope(scope, reuse=TF.AUTO_REUSE):
            input = TF.placeholder(shape=[None, noiseDim], dtype=TF.float32, name=inputName)
            # dropout layer
            if isDropout:
                input = TF.nn.dropout(input, keep_prob=0.8)

            # layer 1
            w1 = TF.get_variable(initializer=TF.truncated_normal(shape=[noiseDim, featureDim]
                                                                 , stddev=0.05
                                                                 , dtype=TF.float32
                                                                 )
                                 , name="g_w_1"
                                 )
            b1 = TF.get_variable(initializer=TF.zeros([featureDim], dtype=TF.float32), name="g_b_1")
            output1 = TF.nn.relu(TF.matmul(input, w1) + b1, name="g_output_1")

            # layer 2
            w2 = TF.get_variable(initializer=TF.truncated_normal([featureDim, featureDim]
                                                                 , stddev=0.05
                                                                 , dtype=TF.float32
                                                                 )
                                 , name="g_w_2"
                                 )

            b2 = TF.get_variable(initializer=TF.zeros([featureDim], dtype=TF.float32), name="g_b_2")
            output2 = TF.nn.relu(TF.matmul(output1, w2) + b2, name="g_output_2")

            # layer 3

            w3 = TF.get_variable(
                initializer=TF.truncated_normal([featureDim, outputShape[0] * outputShape[1] * outputShape[2]]
                                                , stddev=0.05
                                                , dtype=TF.float32
                                                )
                , name="g_w_3"
                )
            b3 = TF.get_variable(initializer=TF.zeros([outputShape[0] * outputShape[1] * outputShape[2]]), name="g_b_3")
            output3 = TF.nn.sigmoid(TF.matmul(output2, w3) + b3)

            # result
            output = TF.reshape(output3, shape=[-1, outputShape[0], outputShape[1], outputShape[2]], name=outputName)
    return input, output