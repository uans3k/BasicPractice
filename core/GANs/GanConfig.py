import core.GANs.Generator as Gen
import core.GANs.Discriminator as Dis
import core.GANs.LossStep as LS
import os
# ganConfig={generator :(graph,inputName,outputShape,outputName,scope)=>input*output:
#           ,discriminator : (graph, input, outputName, scope)=>output:
#           ,lossStep(generator,discriminatorFake,discriminatorReal)=>dLoss,dStep,gLoss,gStep
#           ,noiseDim
#           ,outputShape
#           ,pkgDir
#           ,resultDir
#           }
ROOTDIR=os.path.dirname(os.path.abspath(__file__))
RESULT_DIR=ROOTDIR+"/Result"
PKG_DIR=ROOTDIR+"/Pkg"


originDis=lambda graph, input, outputName,scope: \
        Dis.originDiscriminator(graph=graph, input=input,outputName=outputName, scope=scope,keepProb=0.8)

originGen=lambda graph, noiseDim ,outputShape,inputName, outputName,scope: \
        Gen.originGenrator(graph=graph
                           ,noiseDim=noiseDim
                           , outputShape=outputShape
                           , inputName=inputName
                           , outputName=outputName
                           , scope=scope
                           , featureDim=1200
                           , isDropout=False
                           , keepProb=0.8)

originLossStep=lambda graph,dReal,dFake,dLossName,dStepName,gLossName,gStepName,dScope,gScope,globalStep:    \
                    LS.originLossStep(graph=graph,dReal=dReal,dFake=dFake,dLossName=dLossName,dStepName=dStepName
                              ,gLossName=gLossName,gStepName=gStepName,dScope=dScope,gScope=gScope
                              ,globalStep=globalStep
                              ,dLearnRate=0.0001,gLearnRate=0.0006)

miniAlexNetDis=lambda graph, input, outputName,scope: \
        Dis.miniAlexNetDiscriminator(graph=graph, input=input,outputName=outputName, scope=scope)


defaultConfig= \
{
"generator":originGen

,"discriminator":originDis
,"lossStep":originLossStep
,"noiseDim":50
,"outputShape":[28,28,1]
,"pkgDir":PKG_DIR+"/Origin"
,"resultDir":RESULT_DIR+"/Origin"
}

miniAlexNetConfig= \
{
"generator":originGen
,"discriminator":miniAlexNetDis
,"lossStep":originLossStep
,"noiseDim":100
,"outputShape":[28,28,1]
,"pkgDir":PKG_DIR+"/Origin"
,"resultDir":RESULT_DIR+"/Origin"
}
