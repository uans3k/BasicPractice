import tensorflow as TF
import numpy as NP
class LinearRegression:
    def __init__(self):
        self.sess=None
        self.graph=TF.Graph()
        self.hasBuild=False
        pass
    def open(self):
        self.sess=TF.Session(graph=self.graph)

    def build(self,shape,baseFunc=lambda x: x,reguFunc=lambda w: TF.reduce_sum(TF.square(w)),reguArg=0.1,learningRate=0.05,checkTurn = 25):
        with self.graph.as_default():
            w=TF.Variable(TF.random.normal(shape=shape),name="w")
            b=TF.Variable(TF.random.normal(shape=[shape[0],1]),name="b")
            label=TF.placeholder(shape=[w.shape[0], None], dtype=TF.float32,name="label")
            input=TF.placeholder(shape=[w.shape[1], None], dtype=TF.float32,name="input")
            out=TF.add(TF.matmul(w,baseFunc(input)),b,name="out")
            regu=TF.multiply(TF.add(reguFunc(w), reguFunc(b)),reguArg,name="regu")
            loss=TF.add(TF.reduce_mean(TF.square(out-label)),regu,name="loss")
            opt = TF.train.GradientDescentOptimizer(learningRate,name="opt")
            opt.minimize(loss,name="step")
            init = TF.global_variables_initializer()
            self.sess.run(init)
        self.hasBuild=True
    def buildByFile(self,path):
        with self.graph.as_default:
            saver = TF.train.import_meta_graph(path)
            saver.restore(self.sess, TF.train.latest_checkpoint('./'))
    def close(self):
        self.sess.close()
    def predict(self):
        pass
    def close(self):
        self.sess.close()
    def save(self,path):
        saver = TF.train.Saver()
        saver.save(self.sess, path=path)

    def train(self,data,target,batchSize=2,turn=10,checkTurn=5):
        if self.hasBuild:
            w=self.graph.get_tensor_by_name("w:0")
            b=self.graph.get_tensor_by_name("b:0")
            input=self.graph.get_tensor_by_name("input:0")
            label=self.graph.get_tensor_by_name("label:0")
            loss=self.graph.get_tensor_by_name("loss:0")
            step=self.graph.get_operation_by_name("step")
            for i in range(turn):
                rI = NP.random.choice(len(data), size=batchSize)
                rX = NP.transpose(data[rI])
                rY = NP.transpose(target[rI])
                self.sess.run(step, feed_dict={input: rX, label: rY})
                if i % checkTurn == 0:
                    print("Turn " + str(i) + ":" + "W=" + str(self.sess.run(w)) + " , b=" + str(self.sess.run(
                        b)) + " , loss:" + str(self.sess.run(loss, feed_dict={input: rX, label: rY})))
                else:
                    pass
        else:
            pass


