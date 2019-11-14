import tensorflow as TF

def originLossStep(graph
                   ,dReal
                   , dFake
                   , dLossName
                   , dStepName
                   , gLossName
                   , gStepName
                   , dScope
                   , gScope
                   , globalStep
                   , dLearnRate
                   , gLearnRate
                   ,decayRate=0.96
                   ,decaySteps=1000
                   ):
    with graph.as_default():
        dDecayLearnRate = TF.train.exponential_decay(dLearnRate, globalStep, decaySteps, decayRate, staircase=False)
        gDecayLearnRate = TF.train.exponential_decay(gLearnRate, globalStep, decaySteps, decayRate, staircase=False)

        # build loss function
        d=TF.reduce_mean(TF.log(dReal + 1e-8)) + TF.reduce_mean(TF.log(1 - dFake + 1e-8))
        dLoss = TF.negative(d,name=dLossName)

        g = TF.reduce_mean(TF.log(dFake + 1e-8))
        gLoss=TF.negative(g,name=gLossName)

        # vList=TF.trainable_variables()
        dVList = TF.trainable_variables(scope=dScope)
        gVList = TF.trainable_variables(scope=gScope)

        # build train step
        dStep = TF.train.GradientDescentOptimizer(dDecayLearnRate, name=dStepName).minimize(loss=dLoss,
                                                                                            var_list=dVList)
        gStep = TF.train.GradientDescentOptimizer(gDecayLearnRate, name=gStepName).minimize(loss=gLoss,
                                                                                            var_list=gVList)
    return  dLoss,dStep,gLoss,gStep