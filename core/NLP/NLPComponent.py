import tensorflow as TF

def gruCell(unitNum):
    return TF.keras.layers.GRUCell(units=unitNum)

def biGRU(unitNum):
    return TF.keras.layers.Bidirectional(layer=TF.keras.layers.GRU(units=unitNum,return_sequences=True)
                                         ,backward_layer=TF.keras.layers.GRU(units=unitNum
                                                                             ,return_sequences=True
                                                                             ,go_backwards=True))

#Scaled Dot-Product Attention
#q,k[batchSize,vNum,kDim] self attention kdim=vdim
#v [batchSize,vNum,vDim]]
#mask [vNum]
def dotAttention(q,k,v,mask=None):
    # outshape=[batchSize,vNum,vNum]
    qk=TF.matmul(q,k,transpose_b=True)
    qkScale=TF.div(qk/q.shape()[2])

    #prob [batchSize,vNum,vNum]
    vNum=v.shape()[1]
    if mask is not None:
        mask = TF.tile(TF.reshape(mask,shape=[1,1,vNum]), [1, vNum, 1])
        qkScale=TF.multiply(qkScale,mask)
    prob=TF.nn.softmax(qkScale)

    #gatedV =[batchSize,vNum,vDim]
    gatedV = TF.matmul(prob, v)

    return  gatedV

#Multi-Head Attention
#q,k[batchSize,vNum,dimModel] self attention kdim=vdim
#v [batchSize,vNum,dimModel]]
def multiHeadAttention(q,k,v,dimModel,head,mask=None):
    dv=int(dimModel/head)
    concated=[]
    for i in range(head):

        wQ = _variable(shape=[dimModel, dv])
        wK = _variable(shape=[dimModel, dv])
        wV = _variable(shape=[dimModel, dv])
        pQ = TF.matmul(q,wQ)
        pK = TF.matmul(k, wK)
        pV = TF.matmul(v, wV)
        # shape=[batchSize,vNum,dv]
        gatedV=dotAttention(q=pQ,k=pK,v=pV,mask=mask)
        concated.append(gatedV)
    #shape=[batchSize, vNum, head*dv=dimModel]
    concat=TF.concat(concated,axis=2)

    wO=_variable(shape=[dimModel, dimModel])

    #shape = [batchSize,vNum,dimModel]
    vAttenion=TF.matmul(concat,wO)


#GATED ATTENTION-BASED RECURRENT NETWORKS R-Net
#p=[batchSize,pNum,pDim] q=[batchSize,qNum,qDim]
#return v=[batchSize,vNum,vDim]
#hidden for attention dim,unitNum for RNN dim
def gatedAttentionRNN(q, p, unitNum, hidden):

    # qUnstack qNum [batchSize,qdim] pUnSatck qNum [batchSize,qdim]
    pUnstack = TF.unstack(p, axis=1)

    qDim=q.get_shape().as_list()[2]
    pDim=p.get_shape().as_list()[2]

    wQ=_variable(shape=[qDim,hidden])
    wP=_variable(shape=[pDim,hidden])
    wV=_variable(shape=[hidden,hidden])
    vQP=_variable(shape=[hidden,1])
    wG = _variable(shape=[qDim+pDim,qDim+pDim])

    #ct = [batchSize,qDim]
    ct = None
    # vts  vNum [batchSize,vDim]
    vts = []
    # keras GRU will broadcast it at the axis of BatchSize
    vt =TF.zeros(shape=[1,unitNum])
    for i in range(pUnstack):

        if i == 0:
            ct = _attentionPool(q, wQ, pUnstack[i], wP, None, wV, vQP)
        else:
            ct = _attentionPool(q, wQ, pUnstack[i], wP, vts[i - 1], wV, vQP)
        #uPtCONct=[batchSize,qDim+pDim]
        uPtCONct=TF.concat(pUnstack[i], ct)
        gate=TF.sigmoid(TF.matmul(uPtCONct,wG))
        uPtCONctGated=TF.multiply(uPtCONct,gate)
        vt=gruCell(unitNum=unitNum)(inputs=uPtCONctGated,states=[vt])
        vts.append(vt)

    v=TF.stack(vts,axis=1)
    return v

#R-Net
#v [batchSize,vNum,vDim]
def selfMatchAttentionRNN(v, unitNum, hidden):
    vUnStack = TF.unstack(v, 1)
    vDim = v.get_shape().as_list()[2]
    wV = _variable(shape=[vDim, hidden])
    wV2 = _variable(shape=[vDim, hidden])
    vPP = _variable(shape=[hidden, 1])
    wG = _variable(shape=[vDim + vDim, vDim + vDim])

    # ct = [batchSize,vDim]
    ct = None
    # list of [batchSize,vdim+vdim]
    vsCONctGatedList = []
    for i in range(vUnStack):
        ct = _attentionPool(v, wV, vUnStack[i], wV2, None, None, vPP)
        # vsCONct=[batchSize,vDim+vDim]
        vsCONct = TF.concat(vUnStack[i], ct)
        gate = TF.sigmoid(TF.matmul(vsCONct, wG))
        vsCONctGated = TF.multiply(vsCONct, gate)
        vsCONctGatedList.append(vsCONctGated)

    #vsCONctStack [batchSize,timeStep,vdim+vdim]
    vsCONctStack=TF.stack(vsCONctGatedList, 1)

    # h[batchSize,hNum,unitNum]
    h=biGRU(unitNum=unitNum)(vsCONctStack)

    return h



#return  ct = [batchSize,qDim]
def _attentionPool(q, wQ, pt, wP, vt, wV, vQP):
    qNum=q.get_shape().as_list()[1]
    #uP [batchSize,pDim] utTiled=[batchSize,qNum,pdim]
    pDim=pt.get_shape().as_list()[1]
    ptTiled=TF.tile(TF.reshape(pt, shape=[-1, 1, pDim]), [1, qNum, 1])


    # [batchSize,qNum,hidden]
    actTanh = None
    if vt is None:
        actTanh = TF.tanh(TF.matmul(q, wQ), TF.matmul(ptTiled, wP))
    else:
        # vP [batchSize,vDim] uPtiled=[batchSize,qNum,vdim]
        vDim = vt.get_shape().as_list()[1]
        vtTiled = TF.tile(TF.reshape(vt, shape=[-1, 1, vDim]), [1, qNum, 1])
        actTanh=TF.tanh(TF.matmul(q, wQ) + TF.matmul(ptTiled, wP) + TF.matmul(vtTiled, wV))

    # st prob [batchSize,qNum,1]
    # q [batchSize,qNum,pdim]
    # ct = [batchSize,qDim]

    st=TF.matmul(actTanh,vQP)
    prob=TF.nn.softmax(st,axis=1)
    ct=TF.squeeze(TF.matmul(TF.transpose(q, [0, 2, 1]), prob), axis=2)
    return ct

def _variable(shape):
    return  TF.Variable(TF.random.truncated_normal(shape=shape, stddev=0.1,dtype=TF.float32))


