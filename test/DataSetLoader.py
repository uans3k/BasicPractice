from sklearn import datasets
import numpy as NP
import requests
import os
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import matplotlib.pyplot as PLT



def loadIris():
    iris = datasets.load_iris()
    # Sepal length, Sepal width, Petal length,Petal width
    # print(iris.data)
    #I. setosa, I. virginica, I. versicolor
    # print(iris.target)
    data=NP.array(iris.data)
    target=[]
    for y in iris.target:
        t=[0,0,0]
        t[y]=1
        target.append(t)
    target=NP.array(target)
    return data,target

def  loadBirthdate():
    url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
    birth_file = requests.get(url)
    print(birth_file.text)
    # birth_data = birth_file.text.split('\r\n')[5:]
    # birth_header = [x for x in birth_data[0].split('') if len(x) >= 1]
    # birth_data = [[float(x) for x in y.split('') if len(x) >= 1] for y
    #               in birth_data[1:] if len(y) >= 1]
    # y_vals = np.array([x[1] for x in birth_data])
    # x_vals = np.array([x[2:9] for x in birth_data])

def loadCircle(samples=500, factor=.5,noise=.1):
    return datasets.make_circles(n_samples=500, factor=.5,noise=.1)

def loadMINIST():
    ROOTDIR = os.path.dirname(os.path.abspath(__file__))
    dataChacheDir = ROOTDIR+"/DataSetCache/MINIST"
    mnist = read_data_sets(dataChacheDir)
    trainData = NP.array([NP.reshape(x, [28,28,1]) for x in mnist.
                           train.images])
    testData = NP.array([NP.reshape(x, [28,28,1]) for x in mnist.test.
                          images])
    trainLables = mnist.train.labels
    testLables = mnist.test.labels
    return trainData,trainLables,testData,testLables

# trainD,testD,trainL,testL=loadMINIST()
#
# PLT.figure(figsize=(10,6))
# PLT.ion()
# PLT.subplots_adjust( hspace =0.7)
# for i in range(50):
#     pic=NP.reshape(trainD[i], [28, 28])
#     PLT.subplot(5, 10, i + 1)
#     PLT.imshow(pic, cmap='Greys_r')
#     PLT.title('p: '+str(i+1))
#     PLT.axis("off")
#     PLT.pause(0.05)
#     # frame = PLT.gca()
#     # frame.axes.get_xaxis().set_visible(False)
#     # frame.axes.get_yaxis().set_visible(False)
#
# PLT.ioff()
# PLT.show()
