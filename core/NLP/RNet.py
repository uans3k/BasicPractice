import tensorflow as TF
from core.Exception.RuntimeException import *
import core.NLP.NLPComponent as NLPC
import numpy as NP
import cv2 as CV


class RNet:
    def __init__(self):
        self._sess = None
        self._graph = TF.Graph()

        self._input = None
        self._label = None
        self._wordMat=None
        self._output = None
        self._loss = None
        self._accuracy = None
        self._predict = None

        self._isRetrain = False
        self._isBuild = False
        pass

    def open(self):
        self._sess = TF.Session(graph=self._graph)

    def loadWeigth(self, ckptPath):
        with self._graph.as_default():
            saver = TF.train.Saver()
            saver.restore(self._sess, ckptPath)


    def build(self,pNum,qNum,dicVol,embedDim,dimModel,wordMat=None):
        with self._graph.as_default():
            #[batchSize,pNum],值为词id,词id=0表示为空
            p = TF.placeholder(shape=[None,pNum],dtype=TF.int32)
            q = TF.placeholder(shape=[None, qNum], dtype=TF.int32)

            #embedding layer
            #[batchSize, qNum,embedDim]
            pEmbeded, qEmbeded=self._embeddingLayer(p,q,dicVol,embedDim)

            #q p encoder by BiGru
            uQ=NLPC.biGRU(unitNum=dimModel)(pEmbeded)
            uP=NLPC.biGRU(unitNum=dimModel)(pEmbeded)

            #q&p attention layer
            vP=NLPC.gatedAttentionRNN(q=uQ,p=uP,unitNum=dimModel,hidden=dimModel)

            #self-match attention layer
            hP=NLPC.selfMatchAttentionRNN(v=vP,unitNum=dimModel,hidden=dimModel)

            #out layer



    def _outLayer(self):
        pass

    def _embeddingLayer(self,p,q,dicVol,embedDim,wordMat):
        with TF.compat.v1.variable_scope("embedding"):
            if wordMat is None:
                self._wordMat = TF.Variable(TF.random.truncated_normal(shape=[dicVol, embedDim]
                                                                       , stddev=0.1
                                                                       , dtype=TF.float32
                                                                       )
                                            )
            else:
                self._wordMat = TF.Variable(initial_value=wordMat, trainable=False, dtype=TF.float32)
                pEmbeded = TF.nn.embedding_lookup(self._wordMat, p)
                qEmbeded = TF.nn.embedding_lookup(self._wordMat, q)
                return  pEmbeded,qEmbeded






    def predict(self, imagePath):
        pass

    def close(self):
        self._sess.close()

    def train(self, tfRecordPaths, summaryPath, ckptPath, batchSize=3, epochNum=8, checkTurn=10, saveTurn=100):
        if not self._isBuild:
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

            # init
            if not self._isRetrain:
                self._sess.run(TF.global_variables_initializer())
                self._sess.run(TF.local_variables_initializer())

            # train
            try:
                step = 1
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

    def _readTFRecordAsBatch(self, tfRecordPaths, shape, batchSize, epochNum):
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
        dataset = dataset.shuffle(batchSize * 10)
        nextOP = dataset.make_one_shot_iterator().get_next()

        return nextOP