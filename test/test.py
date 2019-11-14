import tensorflow as TF
import numpy as NP
import core.NLP.NLPComponent as NC

# print(NP.random.uniform(low=-1,high=1,size=[4,4]))
vt =TF.zeros(shape=[2,2])
input=TF.constant([[1,2,3]
                    ,[4,5,6]
                  ],dtype=TF.float32)
vt=NC.gruCell(unitNum=2)(inputs=input, states=[vt])
# cell=NC.gruCell(unitNum=2)
#[2,1,2]
# a=TF.constant([[1,2]
#                ,[3,4]
#               ],dtype=TF.uint8)
#[2,4]
# b=TF.constant([
#                 [[1,2],[3,4]]
#                ,[[1,2],[3,4]]
#                ,[[1,2],[3,4]]
#                ,[[1,2],[3,4]]
#               ],dtype=TF.float32)
# # c=TF.tensordot(a,b,axes=1)
# c=TF.one_hot(a,axis=2,depth=3)

# c=TF.tile(a,a)

sess=TF.Session()
sess.run(TF.global_variables_initializer())
sess.run(TF.local_variables_initializer())
print(sess.run(vt))

# d=TF.reduce_max(c,axis=2)
# print(sess.run(d))

# g1=TF.Graph()
# baseFunc = lambda x: x
# with g1.as_default():
#     a=TF.constant(1.,name="a")
#     b=baseFunc(a)
# sess = TF.Session(graph=g1)
# print(sess.run(b))
# sess.close()
# m=NP.array([[1,2,3],[2,1,3],[1,3,7]])
# d=m.max(axis=0)
# print(m)
#  print(d)
# print(m/d)

