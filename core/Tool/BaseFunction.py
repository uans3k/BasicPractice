import numpy as NP
SOFTING=0.000000001

def normalize(m,axis):
    max=m.max(axis=axis)
    min=m.min(axis=axis)
    d=max-min
    d=[(x if x>0 else SOFTING) for x in d]
    return (m-min)/d

