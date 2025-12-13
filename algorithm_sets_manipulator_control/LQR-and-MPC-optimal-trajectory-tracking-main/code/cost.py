import numpy as np
import control as ctrl
import fs_dynamics_staticps as fsd

ns = fsd.ns
ni = fsd.ni

state_weights = [100.0, 100.0, 100.0, 100.0,  # position states
                1.0, 1.0, 1.0, 1.0]       # velocity states
QQt = 10.0*np.diag(state_weights)
RRt = 0.1*np.eye(ni)

def stagecost(xx,uu, xx_ref, uu_ref):
    xx = xx[:,None]
    uu = uu[:,None]

    xx_ref = xx_ref[:,None]
    uu_ref = uu_ref[:,None]

    ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

    lx = QQt@(xx - xx_ref)
    lu = RRt@(uu - uu_ref)

    return ll.squeeze(), lx, lu

def termcost(xT,xT_ref, QQT = QQt):

    llT = 0.5*(xT - xT_ref).T@QQT@(xT - xT_ref)
    lTx = QQT@(xT - xT_ref)
    
    return llT.squeeze(), lTx
