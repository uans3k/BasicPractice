import tensorflow as TF
from tensorflow import keras as Keras
from core.Exception.RuntimeException import *
import numpy as NP
import cv2 as CV

class ResUNet:
    def __init__(self):
        self._w=-1
        self._h=-1
        self._channel = -1
        self._classNum = -1
        self._model=None


    def _readTFRecord(self, tfRecordPaths, shape, batchSize):
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
            label = TF.reshape(label,shape)

            return image, label

        dataset = dataset.map(parser).batch(batchSize).shuffle(batchSize * 10).repeat()

        return dataset

    def loadWeight(self,weightPath):
        if self._model==None :
            raise IllegalOperationException("Please build or load a model !")
        self._model.load_weights(weightPath)
        print("load weights")

    def train(self,tfRecordPaths,savePath,logPath,batchSize=3, epochs=20,stepsPerEpoch=1000):
        if self._model==None :
            raise IllegalOperationException("Please build or load a model !")


        dataset=self._readTFRecord(tfRecordPaths
                                   ,shape=[self._w,self._h,self._channel]
                                   ,batchSize=batchSize)

        savCallback = Keras.callbacks.ModelCheckpoint(filepath=savePath,save_weights_only=True,save_freq=1,)
        # tbCallback = Keras.callbacks.TensorBoard(log_dir=logPath)
        reduceCallback = Keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3)
        callbacks=[savCallback,reduceCallback]

        self._model.fit(x=dataset
                        ,epochs=epochs
                        ,verbose=1
                        ,steps_per_epoch=stepsPerEpoch
                        ,callbacks=callbacks)

    def predict(self,imgPath):
        if self._model==None :
            raise IllegalOperationException("Please build or load a model !")
        img = CV.imread(imgPath)
        img = NP.asarray(a=img[:, :, 0], dtype=NP.uint8)
        img = NP.reshape(a=img, newshape=[self._w, self._h, self._channel])
        imgs = NP.asarray(a=[img])
        rPred=self._model.predict(imgs, batch_size=1)
        rPred=NP.argmax(rPred,axis=3)[0]
        rPred=NP.reshape(rPred,newshape=[self._w, self._h,1])
        rPred=NP.asarray(rPred,dtype="float32")
        CV.imshow('predict', mat=rPred)
        CV.waitKey(0)



    def build(self,inputW, inputH,inputChanel,classNum):
       self._w = inputW
       self._h = inputH
       self._channel = inputChanel
       self._classNum=classNum
       input = Keras.layers.Input((inputW, inputH, inputChanel),name="input")
       #input layer
       x = self._inputLayer(input=input,filters=64,id="layer0")
       #encoder layer res block (3,4,6,3)
       _,x = self._encoderLayer(input=x,filters=64,blockNum=3,id="layer1",isDown=False)
       actBeforDown1,x = self._encoderLayer(input=x,filters=128,blockNum=4,id="layer2")
       actBeforDown2,x = self._encoderLayer(input=x, filters=256, blockNum=6, id="layer3")
       actBeforDown3,x = self._encoderLayer(input=x, filters=512, blockNum=3, id="layer4")
       #bottom layer
       x = self._bottomLayer(input=x,id="layer5")
       #decoder layer
       x= self._decoderLayer(input=x,skipConn=actBeforDown3,filters=256,id="layer6")
       x = self._decoderLayer(input=x, skipConn=actBeforDown2, filters=128, id="layer7")
       x = self._decoderLayer(input=x, skipConn=actBeforDown1, filters=64, id="layer8")
       #output layer
       x = self._outPutLayer(input=x,filters1=32,filters2=16,classNum=classNum)
       #modelS
       self._model = Keras.models.Model(inputs=input, outputs=x)

       accuracy=Keras.metrics.sparse_categorical_accuracy
       loss= Keras.losses.sparse_categorical_crossentropy

       # def accuracy(y_true, y_pred):
       #     accuracy=Keras.backend.equal(Keras.backend.argmax(y_pred,axis=3),Keras.backend.cast(y_true,"int64"))
       #     accuracy=Keras.backend.mean(x=Keras.backend.cast(accuracy,"float32"),axis=[0,1,2])
       #     return accuracy

       self._model.compile(optimizer=Keras.optimizers.Adam(lr=1e-4)
                     , loss=loss
                     , metrics=[accuracy]
                     )

       # self._model.fit()
       print('model compile complete')

    def _outPutLayer(self,input,filters1,filters2,classNum):
        x=input
        # upsample 1
        x=self._upSample(input=x, filters=filters1, kernelSize=(3, 3), id="up1_output")
        # double conv 1
        x = self._doubleConv(input=x, filters=filters1, id="dconv1_output")

        # upsample 2
        x = self._upSample(input=x, filters=filters2, kernelSize=(3, 3), id="up2_output")
        # double conv 2
        x = self._doubleConv(input=x, filters=filters2, id="dconv2_output")

        #conv 1*1
        x = Keras.layers.Convolution2D(filters=classNum
                                           ,kernel_size=(1,1)
                                           ,strides=(1,1)
                                           ,padding="same"
                                           ,name="conv1_output")(x)
        #softmax act
        x = Keras.layers.Softmax(name="output")(x)

        return x


    def _resBlock(self,input,filters,id):

        x = Keras.layers.BatchNormalization(name="bn1_"+id)(input)
        x = Keras.layers.Activation('relu', name="relu1_" + id)(x)
        x = Keras.layers.Convolution2D(filters=filters,kernel_size=(3,3),name="conv1_"+id,padding="same")(x)

        x = Keras.layers.BatchNormalization(name="bn2_"+id)(x)
        x = Keras.layers.Activation('relu', name="relu2_" + id)(x)
        x = Keras.layers.Convolution2D(filters=filters,kernel_size=(3,3),name="conv2_"+id,padding="same")(x)

        x = Keras.layers.Add(name="add_"+id)([input,x])

        return x

    #...->actBeforDown
    def _resBlockDown(self, input, filters, id):
        x = Keras.layers.BatchNormalization(name="bn1_" + id)(input)
        x = Keras.layers.Activation('relu', name="relu1_" + id)(x)
        actBeforDown=x
        #downsample
        x = Keras.layers.Convolution2D(filters=filters
                                       , kernel_size=(3, 3)
                                       , name="conv1_" + id
                                       , padding="same"
                                       ,strides=(2,2))(x)

        x = Keras.layers.BatchNormalization(name="bn2_" + id)(x)
        x = Keras.layers.Activation('relu', name="relu2_" + id)(x)
        x = Keras.layers.Convolution2D(filters=filters
                                       , kernel_size=(3, 3)
                                       , name="conv2_" + id
                                       , padding="same")(x)

        skipConn=Keras.layers.Convolution2D(filters=filters
                                            , kernel_size=(1, 1)
                                            , name="conv3_" + id
                                            , padding="valid"
                                            ,strides=(2,2))(input)

        x = Keras.layers.Add(name='add_'+id)([skipConn, x])

        return  actBeforDown,x


    def _inputLayer(self,input,filters,id):
        x=input
        # x = Keras.layers.BatchNormalization(name="bn1_" + id)(x)
        x = Keras.layers.Convolution2D(filters=filters
                                       , kernel_size=(7, 7)
                                       , name="conv1_" + id
                                       , strides=(2,2)
                                       , padding="same")(x)

        x = Keras.layers.BatchNormalization(name="bn1_" + id)(x)

        x = Keras.layers.Activation('relu', name="relu1_" + id)(x)

        x = Keras.layers.MaxPooling2D(pool_size=(2, 2)
                                     ,name="pool1_"
                                     ,strides=(2, 2)
                                     ,padding='valid')(x)
        return x

    # ...->actBeforDown,x
    def _encoderLayer(self,input,filters,blockNum,id,isDown=True):
        if blockNum < 1:
            raise ParamException("blckNum must be greater than 1！")
        x=input
        actBeforDown=None
        if isDown :
            actBeforDown,x=self._resBlockDown(input=x,filters=filters,id="block1_"+id)
        else:
            x=self._resBlock(input=x,filters=filters,id="block1_"+id)

        for i in range(blockNum-1):
            x=self._resBlock(input=x, filters=filters, id="block"+str(i+2)+"_"+id)

        return actBeforDown,x

    def _decoderLayer(self,input,skipConn,filters,id):
        x=input
        #upsample
        x=self._upSample(input=x,filters=filters,kernelSize=(3,3),id=id)
        #concate
        x=Keras.layers.Concatenate(name="concate_"+id)([skipConn,x])
        #double conv
        x=self._doubleConv(input=x,filters=filters,id=id)

        return x

    def _bottomLayer(self, input, id):
        x = input
        x = Keras.layers.BatchNormalization(name="bn1_" + id)(x)
        x = Keras.layers.Activation('relu', name="relu1_" + id)(x)
        return x


    def _upSample(self,input,filters,kernelSize,id):
        x=input
        x=Keras.layers.Convolution2DTranspose(filters=filters
                                              ,kernel_size=kernelSize
                                              ,strides=(2,2)
                                              ,padding='same'
                                              ,name="convTrans1_up_"+id)(x)
        x=Keras.layers.Activation(activation="relu",name="relu1_up_" + id)(x)
        return x

    def _doubleConv(self,input,filters,id):
        x=input
        x = Keras.layers.Convolution2D(filters=filters
                                       , kernel_size=(3, 3)
                                       , name="conv1_" + id
                                       , padding="same")(x)

        x = Keras.layers.Activation('relu', name="relu1_dconv" + id)(x)

        x = Keras.layers.Convolution2D(filters=filters
                                       , kernel_size=(3, 3)
                                       , name="conv2_dconv" + id
                                       , padding="same")(x)

        x = Keras.layers.Activation('relu', name="relu2_dconv" + id)(x)

        return x

# class My_Custom_Generator(eras.utils.Sequence):
#     def __init__(self, sess, data_set, batch_size, dataset_size):
#         self._sess = sess
#         self._data_set = data_set
#         self._batch_size = batch_size
#         self._dataset_size = dataset_size
#
#     def __len__(self):
#         return (NP.ceil(self.dataset_size / float(self.batch_size))).astype(np.int)
#
#     def __getitem__(self, idx):
#         data, label = self.sess.run(self.data_set)
#         # label = np.squeeze(label)
#         return data, label



