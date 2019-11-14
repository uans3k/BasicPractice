import tensorflow as TF
from core.Exception.RuntimeException import *
import numpy as NP
import cv2 as CV

class UNet:
    def __init__(self):
        self._sess = None
        self._graph=TF.Graph()

        self._input=None
        self._label=None
        self._output=None
        self._loss=None
        self._accuracy=None
        self._predict=None

        self._w=-1
        self._h=-1
        self._channel = -1
        self._classNum = -1

        self._isRetrain=False
        self._isBuild = False
        pass

    def open(self):
        self._sess = TF.Session(graph=self._graph)

    def restore(self,ckptPath):
        with self._graph.as_default():
            saver = TF.train.Saver()
            saver.restore(self._sess, ckptPath)





    # input * shape * id ->  relu
    def _encoderLayer(self,input,convShape1,convShape2,id):
        with TF.compat.v1.variable_scope("encoder"):
            relu=self._doubleConv(input,convShape1,convShape2,id)
            pool = TF.nn.max_pool(value=relu
                                  , ksize=[1, 2, 2, 1]
                                  , strides=[1, 2, 2, 1]
                                  , padding='VALID'
                                  , name='pool_')


        return relu, pool

    # input*shape*id->upSample
    def _bottomLayer(self,input,convShape1,convShape2,id):
        with TF.compat.v1.variable_scope("encoder"):
            relu=self._doubleConv(input,convShape1,convShape2,id)
        return relu

    #input * shape * id -> upSample

    def _doubleConv(self,input,convShape1,convShape2,id):
        w1 = TF.Variable(TF.random.truncated_normal(shape=convShape1
                                                    , stddev=0.1
                                                    , dtype=TF.float32
                                                    , name="w1_dconv_" + id)
                         )

        b1 = TF.Variable(TF.random.truncated_normal(shape=[convShape1[3]]
                                                    , stddev=0.1
                                                    , dtype=TF.float32
                                                    , name="b1_dconv_" + id)
                         )

        conv1 = TF.nn.conv2d(input=input
                             , filter=w1
                             , strides=[1, 1, 1, 1]
                             , padding='SAME'
                             , name='conv1_dconv_' + id
                             )
        relu1 = TF.nn.relu(TF.nn.bias_add(conv1, b1), name='relu1_dconv_' + id)

        w2 = TF.Variable(TF.random.truncated_normal(shape=convShape2
                                                    , stddev=0.1
                                                    , dtype=TF.float32
                                                    , name="w2_dconv_" + id)
                         )

        b2 = TF.Variable(TF.random.truncated_normal(shape=[convShape2[3]]
                                                    , stddev=0.1
                                                    , dtype=TF.float32
                                                    , name="b2_dconv_" + id)
                         )

        conv2 = TF.nn.conv2d(input=relu1
                             , filter=w2
                             , strides=[1, 1, 1, 1]
                             , padding='SAME'
                             , name='conv2_dconv_' + id
                             )
        relu2 = TF.nn.relu(TF.nn.bias_add(conv2, b2), name='relu2_dconv_' + id)

        return relu2

    def _upSample(self,input,id):
        inputShape = input.get_shape().as_list()
        upSampleShapeW = inputShape[1] * 2
        upSampleShapeH = inputShape[2] * 2
        inputFeatureDim=inputShape[3]
        upFeatrureDim = int(inputFeatureDim / 2)

        w = TF.Variable(
            TF.random.truncated_normal(shape=[2,2,upFeatrureDim,inputFeatureDim]
                                       , stddev=0.1
                                       , dtype=TF.float32
                                       , name="w_up" + id)
        )

        b = TF.Variable(TF.random.truncated_normal(shape=[upFeatrureDim]
                                                    , stddev=0.1
                                                    , dtype=TF.float32
                                                    , name="b_up" + id)
                         )
        #if we use TF.shape,we should convert all to tensor,i.e.output_shape can be assigned a tensor
        upConv = TF.nn.conv2d_transpose(
            value=input
            , filter=w
            ,output_shape= TF.stack([TF.shape(input)[0], upSampleShapeW, upSampleShapeH, upFeatrureDim])
            ,strides=[1, 2, 2, 1], padding='VALID', name='conv_up_' + id)

        relu = TF.nn.relu(TF.nn.bias_add(upConv, b), name='relu_up_' + id)

        return relu


    #input * skip * shape * id -> upSample
    def _decoderLayer(self,input,skipConn,convShape1,convShape2,id):
        with TF.compat.v1.variable_scope("decoder"):
            # upSample
            relu = self._upSample(input=input, id=id)

            # concat
            concat = TF.concat(values=[skipConn, relu], axis=-1)

            #double conv
            relu2=self._doubleConv(concat,convShape1=convShape1,convShape2=convShape2,id=id)

        return  relu2

    def _outputLayer(self,input,convShape):
        with TF.compat.v1.variable_scope("decoder"):
            w = TF.Variable(
                TF.random.truncated_normal(shape=convShape
                                           , stddev=0.1
                                           , dtype=TF.float32
                                           , name="w_output")
            )

            b = TF.Variable(TF.random.truncated_normal(shape=[convShape[3]]
                                                       , stddev=0.1
                                                       , dtype=TF.float32
                                                       , name="b_output")
                            )
            conv = TF.nn.conv2d(input=input
                                 , filter=w
                                 , strides=[1, 1, 1, 1]
                                 , padding='SAME'
                                 , name='conv_output'
                                 )
            output=TF.nn.bias_add(conv,b, name='output')

        return output

    def _buildLoss(self,output,label):
        loss = TF.nn.sparse_softmax_cross_entropy_with_logits(logits=output,labels=label)
        loss = TF.reduce_mean(loss,name='loss')
        return loss

    def _buildAccuracy(self,output,label):
        accuracy=TF.equal(TF.argmax(input=output, axis=3, output_type=TF.int32),label)
        accuracy=TF.reduce_mean(TF.cast(accuracy,dtype=TF.float32), name='accuracy')
        return accuracy
    def _buildAccuracyIoU(self,output,label):
        pass

    def _buildPredict(self,output):
        predict=TF.argmax(input=output, axis=3,name="predict")
        predict=TF.cast(predict, TF.uint8)
        predict= TF.reshape(predict, [self._w, self._h,1])
        return predict

    def _buildStep(self,loss):
        return TF.compat.v1.train.AdamOptimizer(learning_rate=1e-4).minimize(loss,name="step")

    def build(self,inputW, inputH,inputChanel,classNum,isRegular=False):
        self._w=inputW
        self._h=inputH
        self._channel=inputChanel
        self._classNum=classNum

        with self._graph.as_default():
            self._input = TF.compat.v1.placeholder(shape=[None, inputW, inputH, inputChanel], dtype=TF.float32, name="input")
            self._label = TF.compat.v1.placeholder(shape=[None, inputW, inputH], dtype=TF.int32, name="label")
            # encoder
            # layer 1
            relu1, pool1 = self._encoderLayer(self._input, [3, 3, 1, 64], [3, 3, 64, 64], "layer1")

            # layer 2
            relu2, pool2 = self._encoderLayer(pool1, [3, 3, 64, 128], [3, 3, 128, 128], "layer2")

            # layer 3
            relu3, pool3 = self._encoderLayer(pool2, [3, 3, 128, 256], [3, 3, 256, 256], "layer3")

            # layer 4
            relu4, pool4 = self._encoderLayer(pool3, [3, 3, 256, 512], [3, 3, 512, 512], "layer4")

            # layer 5 bottom
            relu5 = self._bottomLayer(pool4, [3, 3, 512, 1024], [3, 3, 1024, 1024], "layer5")
            # decoder
            # layer 6
            relu6 = self._decoderLayer(input=relu5
                                       , skipConn=relu4
                                       , convShape1=[3, 3, 1024, 512]
                                       , convShape2=[3, 3, 512, 512]
                                       , id="layer6")

            # layer 7
            relu7 = self._decoderLayer(input=relu6
                                       , skipConn=relu3
                                       , convShape1=[3, 3, 512, 256]
                                       , convShape2=[3, 3, 256, 256]
                                       , id="layer7")

            # layer 8
            relu8 = self._decoderLayer(input=relu7
                                       , skipConn=relu2
                                       , convShape1=[3, 3, 256, 128]
                                       , convShape2=[3, 3, 128, 128]
                                       , id="layer8")

            # layer 9
            relu9 = self._decoderLayer(input=relu8
                                       , skipConn=relu1
                                       , convShape1=[3, 3, 128, 64]
                                       , convShape2=[3, 3, 64, 64]
                                       , id="layer9")

            # output
            self._output = self._outputLayer(input=relu9, convShape=[1, 1, 64, classNum])

            # loss
            self._loss = self._buildLoss(self._output, self._label)

            # step
            self._step = self._buildStep(self._loss)

            # accuracy
            self._accuracy = self._buildAccuracy(output=self._output, label=self._label)

            # predict
            self._predict = self._buildPredict(output=self._output)

        self._isBuild = True

    def predict(self, imagePath):
        if not self._isBuild:
            raise IllegalOperationException("Please build or load a model !")

        img=CV.imread(imagePath)
        img = NP.asarray(a=img[:, :, 0], dtype=NP.uint8)
        img = NP.reshape(a=img,newshape=[self._w,self._h,1])
        imgs=[img]
        with self._graph.as_default():
            rPredicts= self._sess.run(
                [self._predict],
                feed_dict={self._input:imgs}
            )
            CV.imshow('predict',mat= rPredicts[0] * 255)
            CV.waitKey(0)


    def close(self):
        self._sess.close()


    def train(self, tfRecordPaths,summaryPath,ckptPath,batchSize=3, epochNum=8, checkTurn=10,saveTurn=100):
        if not self._isBuild :
            raise IllegalOperationException("Please build or load a model !")

        with self._graph.as_default():
            summary_writer = TF.summary.FileWriter(summaryPath, flush_secs=60)
            summary_writer.add_graph(self._graph)

            lossSummary = TF.summary.scalar('loss', self._loss)
            accuracySummary = TF.summary.scalar('accuracy', self._accuracy)

            saver = TF.train.Saver(max_to_keep=20)

            nextOP = self._readTFRecordAsBatch(tfRecordPaths=tfRecordPaths
                                               , shape=[self._w, self._h, self._channel]
                                               , batchSize=batchSize
                                               , epochNum=epochNum)

            #init
            if not self._isRetrain:
                self._sess.run(TF.global_variables_initializer())
                self._sess.run(TF.local_variables_initializer())

            #train
            try:
                step=1
                while True:
                    rImgBatch, rLabelBatch = self._sess.run(nextOP)
                    self._sess.run(
                        [self._step],
                        feed_dict={self._input: rImgBatch, self._label: rLabelBatch}
                    )
                    if (step % checkTurn) == 0:
                        rLoss, rAccuracy, rLossSummary, rAccuracySummary = self._sess.run(
                            [self._loss, self._accuracy, lossSummary, accuracySummary],
                            feed_dict={self._input: rImgBatch, self._label: rLabelBatch}
                        )
                        summary_writer.add_summary(rLossSummary, step)
                        summary_writer.add_summary(rAccuracySummary, step)
                        print('steps %d -- loss: %.6f and accuracy: %.6f' % (step, rLoss, rAccuracy))
                    if (step % saveTurn) == 0:
                        saver.save(sess=self._sess, save_path=ckptPath, write_meta_graph=False, global_step=step)
                        pass
                    step += 1


            except TF.errors.OutOfRangeError:
                print('Done training')
            finally:
                saver.save(sess=self._sess, save_path=ckptPath, write_meta_graph=False)
            self._isRetrain = True


    def _readTFRecordAsBatch(self,tfRecordPaths,shape,batchSize,epochNum):
        dataset = TF.data.TFRecordDataset(tfRecordPaths)
        
        def parser(record):
            features = {
                'label': TF.FixedLenFeature([], TF.string),
                'image_raw': TF.FixedLenFeature([], TF.string)
            }
            example = TF.parse_single_example(record, features)

            image = TF.decode_raw(example['image_raw'], TF.uint8)
            image = TF.reshape(image, shape)

            label = TF.decode_raw(example['label'], TF.uint8)
            label = TF.reshape(label, [shape[0], shape[1]])

            return image, label

        dataset = dataset.map(parser)
        dataset = dataset.batch(batchSize)
        dataset = dataset.repeat(epochNum)
        dataset = dataset.shuffle(batchSize*10)
        nextOP = dataset.make_one_shot_iterator().get_next()

        return nextOP