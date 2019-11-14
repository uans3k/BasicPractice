import tensorflow as TF
import core.Exception.RuntimeException as RE

GATE_VALUE_0 = 0.00001
GATE_VALUE_1 = 0.99999


class SVM:
    def __init__(self):
        self.sess = None
        self.graph = TF.Graph()
        self.hasBuild = False
        self.dataSize = 0
        self.dim = 0
        self.data = None
        self.target = None
        pass

    def open(self):
        self.sess = TF.Session(graph=self.graph)

    def build(self, data, target, kernelFunc, slackScale,penaltyFactor=0.05, learningRate=0.001):
        self.dataSize = data.shape[0]
        self.dim = data.shape[1]
        self.data = data
        self.target = target
        with self.graph.as_default():
            input = TF.constant(data, dtype=TF.float32, name="input")
            predInput = TF.placeholder(shape=[None, self.dim], dtype=TF.float32, name="predInput")
            label = TF.constant(target, dtype=TF.float32, name="label")
            lamb = TF.Variable(TF.random_normal(shape=[self.dataSize, 1]), name="lambda")
            mu = TF.Variable(TF.random_normal(shape=[1], dtype=TF.float32), name="mu")
            alphaTrans = TF.Variable(TF.random_uniform(shape=[1,self.dataSize], name="alphaTrans"))
            alphaTransRelu=TF.nn.relu(alphaTrans)
            betaTrans = TF.Variable(TF.random_uniform(shape=[1,self.dataSize],name="betaTrans"))
            betaTransRelu=TF.nn.relu(betaTrans)
            t = label * lamb
            tTrans = TF.transpose(t)
            # build loss function
            firstTerm = TF.reduce_sum(lamb)
            kMatrix = kernelFunc(input, input)
            secondTerm = TF.matmul(TF.matmul(tTrans, kMatrix), t)
            sign0=TF.cast(TF.less(lamb,0),dtype=TF.float32)
            signC=TF.cast(TF.greater(lamb,slackScale),dtype=TF.float32)
            penalty=TF.square(TF.matmul(TF.transpose(lamb), label))+TF.reduce_sum(TF.square(lamb)*sign0)+TF.reduce_sum(TF.square(lamb-slackScale)*signC)
            penaltyTerm=penaltyFactor*penalty
            # subjectTerm = mu * (TF.matmul(TF.transpose(lamb), label)) \
            #               + TF.matmul(alphaTransRelu, -lamb) \
            #               + TF.matmul(betaTransRelu,lamb-slackScale)
            loss = TF.add(secondTerm-firstTerm, penaltyTerm, name="loss")
            # loss=TF.loss = TF.subtract( secondTerm,firstTerm ,name="loss")
            opt = TF.train.GradientDescentOptimizer(learning_rate=learningRate)
            # stepForSubject = opt.minimize(loss=-loss, var_list=[mu, alphaTrans, betaTrans], name="stepForSubject")
            stepForLambda = opt.minimize(loss=loss, var_list=[lamb], name="stepForLambda")
            # eavaluate b
            sBool = TF.greater(TF.abs(lamb), GATE_VALUE_0)
            s = TF.cast(sBool, dtype=TF.float32)
            hBool = TF.greater(lamb,GATE_VALUE_0) & TF.less(lamb, GATE_VALUE_1*slackScale)
            h = TF.cast(hBool, dtype=TF.float32)
            hTrans = TF.transpose(h)
            hNum = TF.reduce_sum(h)
            sLamb = TF.multiply(s,lamb,name="sLamb")
            svTrans=TF.transpose(sLamb*label,name="svTrans")
            b = TF.multiply(1 / hNum, TF.matmul(hTrans, label) - TF.matmul(TF.matmul(svTrans, kMatrix), h))
            # build prediction
            predKMatrix = kernelFunc(input, predInput)
            prediction = TF.add(TF.matmul(svTrans, predKMatrix), b, name="prediction")
            # init model
            init = TF.global_variables_initializer()
            self.sess.run(init)
        self.hasBuild = True

    def buildByFile(self, path):
        with self.graph.as_default:
            saver = TF.train.import_meta_graph(path)
            saver.restore(self.sess, TF.train.latest_checkpoint('./'))

    def close(self):
        self.sess.close()

    def predict(self, data):
        predInput = self.graph.get_tensor_by_name("predInput:0")
        prediction = self.graph.get_tensor_by_name("prediction:0")
        return self.sess.run(prediction, feed_dict={input: predInput})

    def test(self, data, target):
        pass

    def reduce(self):
        pass

    def save(self, path):
        saver = TF.train.Saver()
        saver.save(self.sess, path=path)

    def train(self,turn=1000, checkTurn=25):
        if self.hasBuild:
            input = self.graph.get_tensor_by_name("input:0")
            label = self.graph.get_tensor_by_name("label:0")
            loss = self.graph.get_tensor_by_name("loss:0")
            # stepForSubject = self.graph.get_operation_by_name("stepForSubject")
            stepForLambda = self.graph.get_operation_by_name("stepForLambda")
            sLamb=self.graph.get_tensor_by_name("sLamb:0")
            for i in range(turn):
                # self.sess.run(stepForSubject)
                self.sess.run(stepForLambda)
                if i % checkTurn == 0:
                    print("Turn " + str(i) + "# loss:" + str(self.sess.run(loss)))
                else:
                    pass
            sLambValue=self.sess.run(sLamb)
            print("SuportVector : \n"+str(sLambValue))
        else:
            pass
